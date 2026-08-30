import collections
import hashlib
import json
import platform
import struct
from pathlib import Path

from PIL import Image, ImageStat

from .common import (config_path, file_sha256, new_output, normalized, read_csv,
                     token, write_csv, write_json)


COLUMNS = ["sample_id", "label_relpath", "image_relpath", "label_sha256", "status",
           "serial", "manufacturer_raw", "model_raw", "manufacturer", "model", "class_key",
           "modality", "hospital", "diameter", "length", "tooth_position", "implant_date",
           "capture_date", "width", "height", "image_mode", "intensity_std", "image_sha256",
           "pixel_sha256", "group_id", "source_image_group_id"]


def strict_object(pairs):
    obj = {}
    for key, value in pairs:
        key = normalized(key)
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def get_field(record, path):
    if not path:
        return ""
    current = record
    for key in path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            return ""
    if current is None:
        return ""
    if isinstance(current, (list, dict)):
        raise ValueError("Expected scalar field")
    return normalized(current)


def unique_mapping(rows, key_name):
    result = {}
    for row in rows:
        key = normalized(row.get(key_name, ""))
        if not key or key in result:
            raise ValueError(f"CSV의 {key_name} 값이 비었거나 중복입니다.")
        result[key] = row
    return result


def discover(root, suffixes):
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
                   key=lambda p: str(p.relative_to(root)))
    # Do not let symlinks silently expand the approved dataset boundary.
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("데이터 루트 밖으로 연결된 파일이 있습니다.")
    return paths


def audit(config, run_id, limit=None):
    data = config_path(config, config["data_root"])
    labels = data / config["label_root"]
    images = data / config["image_root"]
    if not labels.is_dir() or not images.is_dir():
        raise ValueError("설정한 라벨·영상 루트 폴더가 없습니다. data_root와 두 상대경로를 확인하세요.")
    label_files = discover(labels, {".json"})
    image_files = discover(images, {s.lower() for s in config["image_extensions"]})
    if not label_files or not image_files:
        raise ValueError("JSON 또는 영상 파일이 없습니다.")
    if limit is not None and limit < 1:
        raise ValueError("limit는 1 이상이어야 합니다.")

    image_by_stem = collections.defaultdict(list)
    image_by_rel = collections.defaultdict(list)
    for path in image_files:
        rel = path.relative_to(images)
        image_by_stem[normalized(rel.with_suffix("").as_posix())].append(path)
        image_by_rel[normalized(rel.as_posix())].append(path)
    label_keys = [normalized(p.relative_to(labels).as_posix()) for p in label_files]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("Unicode 정규화 후 같은 경로가 되는 JSON이 있습니다. 매칭을 중단합니다.")

    pairing = config["pairing"]
    if pairing["mode"] not in ("relative_stem", "mapping_csv"):
        raise ValueError("pairing.mode는 relative_stem 또는 mapping_csv여야 합니다.")
    pairs = unique_mapping(read_csv(config_path(config, pairing["csv"])), "label_relpath") \
        if pairing["mode"] == "mapping_csv" else {}
    grouping = config["grouping"]
    groups = unique_mapping(read_csv(config_path(config, grouping["csv"])), "label_relpath") \
        if grouping.get("csv") else {}
    aliases = {}
    if config.get("aliases_csv"):
        for row in read_csv(config_path(config, config["aliases_csv"])):
            key = (normalized(row["manufacturer_raw"]), normalized(row["model_raw"]))
            value = (normalized(row["manufacturer"]), normalized(row["model"]))
            if key in aliases or not all(key + value):
                raise ValueError("별칭 표에 중복 또는 빈 값이 있습니다.")
            aliases[key] = value

    out = new_output(config, "audit", run_id)
    rows, issues, used_images = [], [], set()
    root_types, field_missing, key_counts = collections.Counter(), collections.Counter(), collections.Counter()
    selected = label_files if limit is None else label_files[:limit]

    def issue(sample_id, kind, rel):
        issues.append({"sample_id": sample_id, "issue": kind, "label_relpath": rel})

    for index, path in enumerate(selected):
        rel = path.relative_to(labels).as_posix()
        key = normalized(rel)
        row = {k: "" for k in COLUMNS}
        row.update(sample_id=token(key), label_relpath=rel, status="excluded")
        rows.append(row)
        try:
            row["label_sha256"] = file_sha256(path)
            raw = json.loads(path.read_text(encoding=config.get("json_encoding", "utf-8-sig")),
                             object_pairs_hook=strict_object)
            root_types[type(raw).__name__] += 1
            record = raw
            for part in config.get("record_path", []):
                record = record[part]
            if not isinstance(record, dict):
                raise ValueError("JSON record must be an object")
            key_counts.update(normalized(k) for k in record)
            record = {normalized(k): v for k, v in record.items()}
            for name, field in config["fields"].items():
                row[name] = get_field(record, field)
                if not row[name]:
                    field_missing[name] += 1
            row["manufacturer_raw"], row["model_raw"] = row["manufacturer"], row["model"]
            row["manufacturer"], row["model"] = aliases.get(
                (row["manufacturer"], row["model"]), (row["manufacturer"], row["model"]))
            if row["manufacturer"] and row["model"]:
                row["class_key"] = json.dumps([row["manufacturer"], row["model"]], ensure_ascii=False)
            else:
                issue(row["sample_id"], "missing_class_label", rel)
        except (ValueError, UnicodeError, OSError, KeyError, IndexError, TypeError):
            issue(row["sample_id"], "json_parse_or_schema_error", rel)
            continue

        if pairing["mode"] == "relative_stem":
            candidates = image_by_stem.get(normalized(Path(rel).with_suffix("").as_posix()), [])
        else:
            target = normalized(pairs.get(key, {}).get("image_relpath", ""))
            candidates = image_by_rel.get(target, [])
        if len(candidates) != 1:
            issue(row["sample_id"], "image_missing" if not candidates else "image_ambiguous", rel)
            continue
        image_path = candidates[0]
        row["image_relpath"] = image_path.relative_to(images).as_posix()
        used_images.add(row["image_relpath"])
        try:
            row["image_sha256"] = file_sha256(image_path)
            with Image.open(image_path) as img:
                row["width"], row["height"] = img.size
                row["image_mode"] = img.mode
                img.load()
                # Decoder pixels, not compressed bytes: catch identical exports with different metadata.
                rgb = img.convert("RGB")
                digest = hashlib.sha256(struct.pack("!II", *rgb.size))
                digest.update(rgb.tobytes())
                row["pixel_sha256"] = digest.hexdigest()
                gray = img.convert("L")
                gray.thumbnail((128, 128))
                row["intensity_std"] = round(ImageStat.Stat(gray).stddev[0], 4)
                if img.mode not in ("L", "RGB"):
                    issue(row["sample_id"], "review_image_mode", rel)
            if row["class_key"] and row["image_mode"] in ("L", "RGB"):
                row["status"] = "eligible"
        except (OSError, ValueError, Image.DecompressionBombError):
            issue(row["sample_id"], "image_decode_error", rel)
            continue

        group = groups.get(key, {})
        if normalized(group.get("group_key", "")):
            row["group_id"] = token("verified-group", normalized(group["group_key"]))
        if normalized(group.get("source_image_key", "")):
            row["source_image_group_id"] = token("source-image", normalized(group["source_image_key"]))
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{len(selected)} JSON files", flush=True)

    # A single image referenced by multiple labels needs review; never silently count it twice.
    by_image = collections.defaultdict(list)
    by_pixel = collections.defaultdict(list)
    for row in rows:
        if row["image_relpath"]:
            by_image[row["image_relpath"]].append(row)
        if row["pixel_sha256"]:
            by_pixel[row["pixel_sha256"]].append(row)
    for same in by_image.values():
        if len(same) > 1:
            for row in same:
                row["status"] = "excluded"
                issue(row["sample_id"], "image_reused_by_multiple_labels", row["label_relpath"])
    duplicates = []
    for digest, same in by_pixel.items():
        if len(same) < 2:
            continue
        conflict = len({r["class_key"] for r in same if r["class_key"]}) > 1
        for row in same:
            duplicates.append({"pixel_sha256": digest, "sample_id": row["sample_id"],
                               "class_key": row["class_key"], "label_conflict": conflict})
            if conflict:
                row["status"] = "excluded"
                issue(row["sample_id"], "duplicate_image_label_conflict", row["label_relpath"])

    write_csv(out / "manifest.csv", rows, COLUMNS)
    write_csv(out / "issues.csv", issues, ["sample_id", "issue", "label_relpath"])
    write_csv(out / "duplicate_pixels.csv", duplicates,
              ["pixel_sha256", "sample_id", "class_key", "label_conflict"])
    unmatched = [{"image_relpath": p.relative_to(images).as_posix()} for p in image_files
                 if p.relative_to(images).as_posix() not in used_images]
    write_csv(out / "unmatched_images.csv", unmatched, ["image_relpath"])
    counts = collections.Counter((r["manufacturer"], r["model"], r["hospital"],
                                  r["modality"], r["status"]) for r in rows)
    write_csv(out / "class_hospital_modality_counts.csv", [dict(zip(
        ["manufacturer", "model", "hospital", "modality", "status", "image_count"], (*key, n)))
        for key, n in sorted(counts.items())],
        ["manufacturer", "model", "hospital", "modality", "status", "image_count"])
    eligible = [r for r in rows if r["status"] == "eligible"]
    full = len(selected) == len(label_files)
    write_json(out / "summary.json", {
        "code_version": "0.2.0", "python": platform.python_version(),
        "full_scan": full, "label_files_total": len(label_files), "label_files_scanned": len(rows),
        "image_files_total": len(image_files), "eligible_rows": len(eligible),
        "excluded_rows": len(rows) - len(eligible),
        "class_count_eligible": len({r["class_key"] for r in eligible}),
        "eligible_rows_without_group": sum(not r["group_id"] for r in eligible),
        "grouping_verified": bool(grouping.get("verified")), "grouping_kind": grouping["kind"],
        "json_root_types": root_types, "json_key_counts": key_counts, "missing_fields": field_missing,
        "issue_counts": collections.Counter(i["issue"] for i in issues),
        "manifest_sha256": file_sha256(out / "manifest.csv"),
        "all_outputs_internal_only": True,
        "warnings": ["일련번호는 환자키로 간주하지 않음", "병원별데이터_01은 기관 ID로 간주하지 않음",
                     "unmatched_images는 부분 감사일 때 아직 읽지 않은 JSON의 영상도 포함",
                     "픽셀 해시는 정확히 같은 디코딩 영상만 탐지하며 유사 크롭·재촬영 탐지는 아님"]})
    return out
