"""Reviewed split boundary, independent of the optional learning packages."""
import collections
import json
from pathlib import Path

from .common import config_path, file_sha256, new_output, read_csv, write_csv, write_json
from .split import PARTITIONS


def validate_rows(rows):
    if not rows or {r['split'] for r in rows} != set(PARTITIONS):
        raise ValueError('네 분할이 모두 비어 있지 않아야 합니다.')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('중복 sample_id가 있습니다.')
    seen = {field: {} for field in ('group_id', 'source_image_group_id',
                                   'pixel_sha256', 'leakage_group_id', 'image_relpath')}
    classes = {}
    for row in rows:
        if row['status'] != 'eligible' or any(not row[k] for k in
                ('group_id', 'leakage_group_id', 'pixel_sha256', 'image_sha256', 'image_relpath')):
            raise ValueError('사용 가능 여부·그룹키·영상 해시가 누락되었습니다.')
        key = int(row['class_id'])
        if key in classes and classes[key] != row['class_key']:
            raise ValueError('class_id와 class_key가 일치하지 않습니다.')
        classes[key] = row['class_key']
        for field, values in seen.items():
            value = row[field]
            if value and value in values and values[value] != row['split']:
                raise ValueError(f'분할 누수 발견: {field}')
            if value:
                values[value] = row['split']
    if sorted(classes) != list(range(len(classes))) or len(set(classes.values())) != len(classes):
        raise ValueError('클래스 번호는 0부터 연속이며 라벨과 일대일이어야 합니다.')
    if len(classes) < 2:
        raise ValueError('분류에는 두 제품 이상이 필요합니다.')
    for split in ('train', 'dev'):
        present = {int(r['class_id']) for r in rows if r['split'] == split}
        if present != set(classes):
            raise ValueError(f'{split}에 없는 제품이 있습니다. 학습 전 제품 범위·분할을 검토하세요.')
    return [{'class_id': k, 'class_key': classes[k]} for k in sorted(classes)]


def freeze_split(config, split_dir, run_id, review_note):
    if not review_note or not review_note.strip():
        raise ValueError('분할·제품 범위 검토 근거인 review-note가 필요합니다.')
    source = Path(split_dir).resolve()
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'DRAFT_REQUIRES_REVIEW_BEFORE_TRAINING':
        raise ValueError('split-draft 결과를 사용하세요.')
    manifest = source / 'split_manifest.csv'
    if file_sha256(manifest) != summary['split_manifest_sha256']:
        raise ValueError('분할 초안 manifest가 변경되었습니다.')
    rows = read_csv(manifest)
    classes = validate_rows(rows)
    source_config = json.loads((source / 'config_snapshot.json').read_text(encoding='utf-8'))
    for key in ('data_root', 'image_root', 'label_root', 'pairing', 'grouping'):
        if config[key] != source_config[key]:
            raise ValueError(f'분할 때와 현재 설정이 다릅니다: {key}')
    out = new_output(config, 'frozen', run_id)
    write_csv(out / 'split_manifest.csv', rows, list(rows[0]))
    write_json(out / 'class_map.json', classes)
    write_json(out / 'study.json', {
        'status': 'FROZEN_CLOSED_SET_BASELINE',
        'review_note': review_note.strip(),
        'source_split_summary_sha256': file_sha256(source / 'summary.json'),
        'manifest_sha256': file_sha256(out / 'split_manifest.csv'),
        'class_map_sha256': file_sha256(out / 'class_map.json'),
        'data_root': str(config_path(config, config['data_root'])),
        'image_root': config['image_root'],
        'grouping_kind': summary['grouping_kind'],
        'class_scope': 'All classes in the reviewed draft; no implicit rare-class removal',
        'selection_metric': 'dev_macro_f1_all_classes',
        'counts': dict(collections.Counter(r['split'] for r in rows)),
        'all_outputs_internal_only': True,
    })
    return out


def load_study(directory):
    directory = Path(directory).resolve()
    study = json.loads((directory / 'study.json').read_text(encoding='utf-8'))
    if study.get('status') != 'FROZEN_CLOSED_SET_BASELINE':
        raise ValueError('freeze-split로 먼저 검토 기록을 남기세요.')
    for name, field in [('split_manifest.csv', 'manifest_sha256'), ('class_map.json', 'class_map_sha256')]:
        if file_sha256(directory / name) != study[field]:
            raise ValueError(f'확정된 분할 파일이 변경되었습니다: {name}')
    rows = read_csv(directory / 'split_manifest.csv')
    classes = validate_rows(rows)
    if classes != json.loads((directory / 'class_map.json').read_text(encoding='utf-8')):
        raise ValueError('클래스 표가 manifest와 일치하지 않습니다.')
    study['study_sha256'] = file_sha256(directory / 'study.json')
    return study, rows, classes


def image_path(study, row):
    root = (Path(study['data_root']) / study['image_root']).resolve()
    relative = Path(row['image_relpath'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError('영상 경로가 승인된 영상 루트 밖에 있습니다.')
    return path


def verify_images(study, rows):
    for row in rows:
        if file_sha256(image_path(study, row)) != row['image_sha256']:
            raise ValueError('감사 이후 영상이 변경되었습니다. 학습/평가를 중단합니다.')
