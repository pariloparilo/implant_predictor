"""First-visit inventory. Provider partitions are preserved; no training/split creation."""
import collections
import copy
import html
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .audit import audit, discover
from .common import config_path, file_sha256, new_output, read_csv, token, write_csv, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def setup(config):
    root = config_path(config, config['data_root'])
    if not root.is_dir():
        raise ValueError('data_root가 없습니다. 01.데이터의 실제 절대경로를 확인하세요.')
    partitions = config['provider_partitions']
    if not isinstance(partitions, list) or len(partitions) != 2:
        raise ValueError('provider_partitions에 제공 Training과 Validation 두 영역을 지정하세요.')
    ids, paths = set(), []
    for partition in partitions:
        name = partition['id']
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('영역 id는 영문 소문자·숫자·밑줄·하이픈으로 지정하세요.')
        if name in ids:
            raise ValueError('중복된 영역 id입니다.')
        ids.add(name)
        for key in ('label_root', 'image_root'):
            rel = Path(partition[key])
            path = (root / rel).resolve()
            if rel.is_absolute() or not path.is_relative_to(root) or path == root:
                raise ValueError('영역 경로는 data_root 내부 상대경로여야 합니다.')
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths):
                raise ValueError('영역 경로가 중복되거나 서로 포함됩니다.')
            paths.append(path)
    return root, partitions


def environment(config, check_ml):
    packages = {}
    for package in ('Pillow', 'torch', 'torchvision', 'numpy', 'matplotlib'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'NOT_INSTALLED'
    result = dict(python=platform.python_version(), os=platform.system(), machine=platform.machine(),
                  packages=packages, gpu_check='NOT_REQUESTED', weights_check='NOT_CONFIGURED')
    if check_ml:
        try:
            import torch
            import torchvision
            result['gpu_check'] = 'CUDA_OK' if torch.cuda.is_available() else 'CUDA_UNAVAILABLE'
            result['torch_cuda_build'] = torch.version.cuda
            if torch.cuda.is_available():
                x = torch.ones((8, 8), device='cuda')
                result['cuda_operation_ok'] = bool((x @ x).sum().item() == 512)
                gpu = torch.cuda.get_device_properties(0)
                result.update(gpu_name=gpu.name, gpu_memory_gib=round(gpu.total_memory / 2**30, 2))
        except (ImportError, OSError, RuntimeError, AssertionError, ValueError, AttributeError) as exc:
            result['gpu_check'] = 'ML_IMPORT_OR_GPU_ERROR'
            result['ml_error_type'] = type(exc).__name__
    settings = config.get('training', {})
    if settings.get('weights_path'):
        try:
            path = config_path(config, settings['weights_path'])
            if not path.is_file():
                result['weights_check'] = 'FILE_MISSING'
            elif not settings.get('weights_sha256'):
                result['weights_check'] = 'EXPECTED_HASH_MISSING'
            else:
                result['weights_check'] = 'HASH_MATCH' if file_sha256(path) == settings['weights_sha256'].lower() else 'HASH_MISMATCH'
        except OSError:
            result['weights_check'] = 'READ_ERROR'
    return result


def inventory(root, partition, extensions):
    labels, images = root / partition['label_root'], root / partition['image_root']
    if not labels.is_dir() or not images.is_dir():
        raise ValueError('해당 영역의 라벨링데이터/원천데이터 경로가 없습니다.')
    label_files = discover(labels, {'.json'})
    image_files = discover(images, set(extensions))
    folders = collections.Counter()
    for files, base, kind in [(label_files, labels, 'json'), (image_files, images, 'image')]:
        for path in files:
            rel = path.relative_to(base)
            folders[(rel.parts[0] if len(rel.parts) > 1 else '(root)', kind)] += 1
    return dict(json_files=len(label_files), image_files=len(image_files),
                source_folders=len({folder for folder, kind in folders if kind == 'json'})), [
        dict(provider_partition=partition['id'], source_folder=folder, file_kind=kind, count=count)
        for (folder, kind), count in sorted(folders.items())]


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(min=values[0], median=values[len(values)//2], max=values[-1])


def aggregate(records, names, complete, out):
    classes = {name: set() for name in names}
    pixels = collections.defaultdict(list)
    counts, folder_hospital = collections.Counter(), collections.Counter()
    for row in records:
        name = row['provider_partition']
        if row['status'] == 'eligible':
            classes[name].add(row['class_key'])
        if row['pixel_sha256']:
            pixels[row['pixel_sha256']].append(row)
        counts[(name, row['class_key'], row['hospital'], row['modality'], row['status'])] += 1
        rel = Path(row['label_relpath'])
        folder_hospital[(name, rel.parts[0] if len(rel.parts) > 1 else '(root)', row['hospital'])] += 1
    overlap, conflicts, duplicate_members = 0, 0, []
    for digest, same in pixels.items():
        if len({r['provider_partition'] for r in same}) > 1:
            overlap += 1
            conflict = len({r['class_key'] for r in same if r['class_key']}) > 1
            conflicts += conflict
            for row in same:
                duplicate_members.append(dict(provider_partition=row['provider_partition'],
                    qualified_sample_id=row['qualified_sample_id'], pixel_sha256=digest,
                    label_conflict=conflict, label_relpath=row['label_relpath']))
    write_csv(out / 'cross_partition_duplicates.csv', duplicate_members,
              ['provider_partition', 'qualified_sample_id', 'pixel_sha256', 'label_conflict', 'label_relpath'])
    columns = ['provider_partition', 'class_key', 'hospital', 'modality', 'status', 'count']
    write_csv(out / 'class_hospital_modality_counts.csv', [dict(zip(columns, (*key, count)))
        for key, count in sorted(counts.items())], columns)
    columns = ['provider_partition', 'source_folder', 'hospital', 'count']
    write_csv(out / 'folder_hospital_counts.csv', [dict(zip(columns, (*key, count)))
        for key, count in sorted(folder_hospital.items())], columns)
    first, second = names
    return dict(scope='FULL' if complete else 'PARTIAL_OBSERVED_ONLY',
        cross_partition_exact_pixel_clusters=overlap,
        cross_partition_label_conflict_clusters=conflicts,
        observed_classes_only_in_first=len(classes[first] - classes[second]),
        observed_classes_only_in_second=len(classes[second] - classes[first]),
        first_partition=first, second_partition=second,
        warning='부분 점검의 0은 전체 중복 없음·제품 없음의 증거가 아닙니다. 정확한 픽셀 중복 검사로 환자 독립성은 확인할 수 없습니다.')


def contact_sheet(records, root, partitions, out):
    lookup = {p['id']: p for p in partitions}
    # records are already balanced across first-level source folders within each partition.
    buckets = {p['id']: [r for r in records if r['provider_partition'] == p['id'] and r['status'] == 'eligible']
               for p in partitions}
    selected = []
    while any(buckets.values()) and len(selected) < 30:
        for name in buckets:
            if buckets[name] and len(selected) < 30:
                selected.append(buckets[name].pop(0))
    if not selected:
        return
    sheet = Image.new('RGB', (1000, 40 + ((len(selected) + 4)//5)*220), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), 'INTERNAL ONLY - visual crop review; not model predictions', fill='black')
    mapping = []
    for i, row in enumerate(selected):
        image_root = (root / lookup[row['provider_partition']]['image_root']).resolve()
        path = (image_root / row['image_relpath']).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError('미리보기 경로가 영상 루트를 벗어납니다.')
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert('RGB'), (180, 180))
        x, y = (i % 5)*200 + 10, (i//5)*220 + 40
        sheet.paste(thumb, (x + (180-thumb.width)//2, y))
        draw.text((x, y+185), f"{i+1:02d} {row['provider_partition']}", fill='black')
        mapping.append(dict(review_number=i+1, **row))
    sheet.save(out / 'contact_sheet.png')
    write_csv(out / 'visual_review_internal.csv', mapping, list(mapping[0]))


def human_report(summary, out):
    stage = summary['stage']
    env = summary['environment']
    lines = ['현장 점검 화면 — 모든 파일은 안심존 내부 전용',
        '구두·메모 전달도 기관이 허용한 항목에 한합니다. 자동 반출 승인 자료가 아닙니다.',
        f"실행 단계: {stage} / 상태: {summary['status']}",
        f"환경: Python {env['python']} / {env['os']} {env['machine']}",
        f"GPU: {env['gpu_check']} / 가중치: {env['weights_check']}",
        f"패키지: {json.dumps(env['packages'], ensure_ascii=False)}",
        f"완료까지 경과: {summary['elapsed_seconds']}초 / 작업 디스크 여유: {summary['free_disk_gib']} GiB", '']
    for name, result in summary['partitions'].items():
        lines.append(f"[{name}] 상태={result['status']}")
        if 'inventory' in result:
            inv = result['inventory']
            lines.append(f"JSON {inv['json_files']} / 영상 {inv['image_files']} / 라벨 상위폴더 {inv['source_folders']}")
        if 'audit' in result:
            a = result['audit']
            lines.extend([f"점검 {a['label_files_scanned']} / 사용가능 {a['eligible_rows']} / 제외 {a['excluded_rows']} / 관측 제품 {a['class_count_eligible']}",
                f"단일 대응 영상이 확인된 JSON 수: {result['matched_rows']}",
                f"오류 종류: {json.dumps(a['issue_counts'], ensure_ascii=False)}",
                f"필드 누락: {json.dumps(a['missing_fields'], ensure_ascii=False)}",
                f"영상 폭/높이: {json.dumps(result['dimensions'], ensure_ascii=False)}",
                f"병원별 그룹키 확인: {a['grouping_kind']} / verified={a['grouping_verified']}"])
        if 'error_type' in result:
            lines.append(f"오류 유형: {result['error_type']} (내부 details.json 참조)")
        lines.append('')
    if 'cross_partition' in summary:
        cross = summary['cross_partition']
        lines.extend([f"영역 간 점검 범위: {cross['scope']}",
            f"서로 겹치는 동일 픽셀 군집: {cross['cross_partition_exact_pixel_clusters']}",
            f"그중 라벨 충돌 군집: {cross['cross_partition_label_conflict_clusters']}",
            f"앞 영역에서만 관측된 제품: {cross['observed_classes_only_in_first']}",
            f"뒤 영역에서만 관측된 제품: {cross['observed_classes_only_in_second']}", cross['warning']])
    lines.extend(['', 'G01: 환자/원촬영 그룹키 의미·대응표 확인 필요',
        'S01: 제공 Training/Validation의 원래 분할 기준·재분할 허용 범위 확인 필요',
        'V01: sample의 contact_sheet.png와 visual_review_internal.csv로 매칭·크롭 수동 검토',
        '이 점검 성공은 학습 승인·분할 독립성·의료 성능 검증을 의미하지 않습니다.',
        '방문 후 전달 양식: FIELD_FEEDBACK_TEMPLATE.md. 허용된 상태·수치만 전사하세요.'])
    (out / 'READ_ON_SCREEN.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    body = html.escape('\n'.join(lines))
    preview = '<p>아래 영상은 내부 검토 전용입니다.</p><img src="contact_sheet.png" alt="내부 크롭 점검">' if (out/'contact_sheet.png').exists() else ''
    (out / 'dashboard.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>현장 점검</title>'
        '<style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;background:#f5f7fa;color:#172334}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px;border-radius:10px}img{max-width:100%}</style>'
        '<h1>현장 점검</h1><pre>'+body+'</pre>'+preview+'</html>', encoding='utf-8')


def visit(config, run_id, stage='preflight', check_ml=False):
    if stage not in ('preflight', 'sample', 'full'):
        raise ValueError('stage는 preflight, sample, full 중 하나여야 합니다.')
    root, partitions = setup(config)
    start = time.monotonic()
    out = new_output(config, 'visit', run_id)
    summary = dict(stage=stage, status='RUNNING', environment=environment(config, check_ml),
                   partitions={}, all_outputs_internal_only=True, creates_training_split=False)
    write_json(out / 'summary.json', summary)
    records, folders, errors = [], [], []
    extensions = [x.lower() for x in config['image_extensions']]
    limit = config.get('visit', {}).get('sample_per_partition', 150)
    if type(limit) is not int or limit < 1:
        raise ValueError('sample_per_partition은 양의 정수여야 합니다.')
    for partition in partitions:
        name = partition['id']
        result = summary['partitions'][name] = {'status': 'RUNNING'}
        try:
            inv, folder_rows = inventory(root, partition, extensions)
            result['inventory'] = inv
            folders.extend(folder_rows)
            if not inv['json_files'] or not inv['image_files']:
                raise ValueError('JSON 또는 지원 영상 파일이 없습니다. 압축 해제 상태·확장자를 확인하세요.')
            if stage != 'preflight':
                cfg = copy.deepcopy(config)
                cfg.pop('provider_partitions')
                cfg.update(data_root=str(root), label_root=partition['label_root'],
                           image_root=partition['image_root'], work_dir=str(out / 'internal'),
                           audit_sampling='balanced_folder')
                cfg['pairing'] = partition.get('pairing', config['pairing'])
                cfg['grouping'] = partition.get('grouping', config['grouping'])
                folder = audit(cfg, name, limit if stage == 'sample' else None)
                result['audit'] = read_json(folder / 'summary.json')
                rows = read_csv(folder / 'manifest.csv')
                result['matched_rows'] = sum(bool(r['image_relpath']) for r in rows)
                for row in rows:
                    row['provider_partition'] = name
                    row['qualified_sample_id'] = token(name, row['sample_id'])
                records.extend(rows)
                result['dimensions'] = {key: distribution([int(r[key]) for r in rows if r[key]])
                                        for key in ('width', 'height')}
            result['status'] = 'COMPLETED'
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result.update(status='ERROR', error_type=type(exc).__name__)
            errors.append(dict(provider_partition=name, type=type(exc).__name__, detail=str(exc)))
        write_json(out / 'summary.json', summary)
    write_csv(out / 'folder_inventory.csv', folders, ['provider_partition', 'source_folder', 'file_kind', 'count'])
    write_json(out / 'details.json', errors)
    if stage != 'preflight':
        complete = len(summary['partitions']) == len(partitions) and all(
            r.get('status') == 'COMPLETED' and r.get('audit', {}).get('full_scan') for r in summary['partitions'].values())
        summary['cross_partition'] = aggregate(records, [p['id'] for p in partitions], complete, out)
        if records:
            write_csv(out / 'provider_manifest_internal.csv', records, list(records[0]))
        if stage == 'sample':
            try:
                contact_sheet(records, root, partitions, out)
            except (ValueError, OSError) as exc:
                errors.append(dict(type=type(exc).__name__, detail=str(exc), step='contact_sheet'))
                write_json(out / 'details.json', errors)
    summary.update(status='COMPLETED_WITH_ERRORS' if errors else 'COMPLETED',
                   elapsed_seconds=round(time.monotonic()-start, 2),
                   free_disk_gib=round(shutil.disk_usage(out).free / 2**30, 1))
    write_json(out / 'summary.json', summary)
    human_report(summary, out)
    return out
