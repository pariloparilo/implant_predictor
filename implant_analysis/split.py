import collections
import json
import random
from pathlib import Path

from .common import file_sha256, new_output, read_csv, token, write_csv, write_json


PARTITIONS = ("train", "dev", "calibration", "test")


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def allocate_groups(count, ratios):
    if count < 4:
        raise ValueError("연결된 독립 그룹이 4개 미만이라 네 집합으로 나눌 수 없습니다.")
    # Keep each partition nonempty; quotas concern components, not image counts.
    desired = {p: (count - 4) * ratios[p] for p in PARTITIONS}
    result = {p: 1 + int(desired[p]) for p in PARTITIONS}
    order = sorted(PARTITIONS, key=lambda p: (-(desired[p] - int(desired[p])), p))
    for p in order[:count - sum(result.values())]:
        result[p] += 1
    return result


def split_draft(config, audit_dir, run_id):
    audit_dir = Path(audit_dir).resolve()
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    audited = json.loads((audit_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    grouping = audited["grouping"]
    if not summary["full_scan"]:
        raise ValueError("부분 감사(--limit) 결과로는 분할할 수 없습니다. 전체 감사를 실행하세요.")
    if not audited["pairing"].get("verified"):
        raise ValueError("영상-JSON 매칭 규칙을 현장에서 검토한 뒤 pairing.verified를 설정하고 다시 감사하세요.")
    if not grouping.get("verified") or grouping["kind"] not in ("patient", "source_image") \
            or not grouping.get("verification_note", "").strip():
        raise ValueError("기관에서 확인한 환자/원촬영 그룹키와 확인 근거가 필요합니다. 일련번호로 자동 대체하지 않습니다.")
    if file_sha256(audit_dir / "manifest.csv") != summary["manifest_sha256"]:
        raise ValueError("감사 manifest가 변경되었습니다. 원본 감사 결과를 사용하세요.")
    all_rows = read_csv(audit_dir / "manifest.csv")
    rows = sorted([r for r in all_rows if r["status"] == "eligible"], key=lambda r:r["sample_id"])
    if not rows or any(not r["group_id"] or not r["pixel_sha256"] for r in rows):
        raise ValueError("사용 가능한 영상이 없거나 일부 영상의 그룹키/픽셀 해시가 없습니다.")
    ratios = config["split"]["ratios"]
    if set(ratios) != set(PARTITIONS) or any(ratios[p] <= 0 for p in PARTITIONS) \
            or abs(sum(ratios.values()) - 1) > 1e-8:
        raise ValueError("train/dev/calibration/test 비율은 양수이고 합계가 1이어야 합니다.")

    # Unite all relationships before removing repeated images. This also joins
    # distinct supplied group keys connected by the very same decoded image.
    union = UnionFind(r["sample_id"] for r in rows)
    for field in ("group_id", "source_image_group_id", "pixel_sha256"):
        first = {}
        for row in rows:
            value = row[field]
            if not value:
                continue
            if value in first:
                union.union(first[value], row["sample_id"])
            first[value] = row["sample_id"]
    members = collections.defaultdict(list)
    for row in rows:
        members[union.find(row["sample_id"])].append(row["sample_id"])
    component_id = {root: token("component", sorted(ids)) for root, ids in members.items()}
    for row in rows:
        row["leakage_group_id"] = component_id[union.find(row["sample_id"])]

    kept, duplicate_records, seen = [], [], {}
    for row in rows:
        digest = row["pixel_sha256"]
        if digest in seen:
            if row["class_key"] != seen[digest]["class_key"]:
                raise ValueError("동일 픽셀에 서로 다른 라벨이 있습니다. 먼저 감사 결과를 확인하세요.")
            duplicate_records.append({"removed_sample_id": row["sample_id"],
                                      "kept_sample_id": seen[digest]["sample_id"]})
        else:
            seen[digest] = row
            kept.append(row)
    components = collections.defaultdict(list)
    for row in kept:
        components[row["leakage_group_id"]].append(row)
    ids = sorted(components)
    quotas = allocate_groups(len(ids), ratios)
    classes = sorted({r["class_key"] for r in kept})
    group_labels = {group: {r["class_key"] for r in records} for group, records in components.items()}
    totals = collections.Counter(label for labels in group_labels.values() for label in labels)
    candidate_count = config["split"].get("candidate_count", 64)
    if not isinstance(candidate_count, int) or not 1 <= candidate_count <= 1000:
        raise ValueError("분할 초안 후보 수는 1~1000이어야 합니다.")

    best = None
    for candidate in range(candidate_count):
        shuffled = ids.copy()
        random.Random(config["seed"] + candidate).shuffle(shuffled)
        assignment, offset = {}, 0
        counts = {p: collections.Counter() for p in PARTITIONS}
        image_counts = collections.Counter()
        for partition in PARTITIONS:
            for group in shuffled[offset:offset + quotas[partition]]:
                assignment[group] = partition
                counts[partition].update(group_labels[group])
                image_counts[partition] += len(components[group])
            offset += quotas[partition]
        missing_train = sum(counts["train"][label] == 0 for label in classes)
        score = 1000 * missing_train + sum(
            (counts[p][label] / totals[label] - ratios[p]) ** 2
            for p in PARTITIONS for label in classes)
        score += sum((image_counts[p] / len(kept) - ratios[p]) ** 2 for p in PARTITIONS)
        choice = (score, candidate)
        if best is None or choice < best[0]:
            best = choice, assignment
    assignment = best[1]
    class_ids = {label: i for i, label in enumerate(classes)}
    for row in kept:
        row["split"] = assignment[row["leakage_group_id"]]
        row["class_id"] = class_ids[row["class_key"]]
    for field in ("group_id", "source_image_group_id", "pixel_sha256", "leakage_group_id"):
        partitions = collections.defaultdict(set)
        for row in kept:
            if row[field]:
                partitions[row[field]].add(row["split"])
        if any(len(values) != 1 for values in partitions.values()):
            raise RuntimeError(f"분할 누수 검사 실패: {field}")

    by_class = []
    minimums = config["split"]["core_min_groups"]
    for label in classes:
        entry = {"class_id": class_ids[label], "class_key": label}
        for p in PARTITIONS:
            subset = [r for r in kept if r["class_key"] == label and r["split"] == p]
            entry[p + "_images"] = len(subset)
            entry[p + "_groups"] = len({r["leakage_group_id"] for r in subset})
        entry["primary_candidate"] = all(entry[p + "_groups"] >= minimums[p] for p in PARTITIONS)
        by_class.append(entry)
    out = new_output(config, "split_draft", run_id)
    write_csv(out / "split_manifest.csv", kept, list(kept[0]))
    write_csv(out / "class_split_counts.csv", by_class, list(by_class[0]))
    write_csv(out / "removed_exact_duplicates.csv", duplicate_records, ["removed_sample_id", "kept_sample_id"])
    write_json(out / "class_map.json", [{"class_id": class_ids[label], "class_key": label} for label in classes])
    write_json(out / "summary.json", {
        "status": "DRAFT_REQUIRES_REVIEW_BEFORE_TRAINING", "grouping_kind": grouping["kind"],
        "grouping_verification_note": grouping["verification_note"],
        "audit_manifest_sha256": summary["manifest_sha256"],
        "split_manifest_sha256": file_sha256(out / "split_manifest.csv"),
        "eligible_rows_before_dedup": len(rows), "rows_after_dedup": len(kept),
        "exact_duplicate_rows_removed": len(duplicate_records), "connected_groups": len(ids),
        "requested_ratios": ratios, "actual_group_counts": quotas,
        "actual_image_counts": collections.Counter(r["split"] for r in kept),
        "class_count": len(classes), "train_missing_classes": [r["class_key"] for r in by_class if not r["train_groups"]],
        "primary_candidate_classes": sum(r["primary_candidate"] for r in by_class),
        "selected_candidate_seed": config["seed"] + best[0][1],
        "candidate_selection_uses": "group/label counts only; no model predictions or image features",
        "leakage_checks": "passed for supplied group keys, supplied source-image keys, exact decoded pixels",
        "warnings": ["원촬영 단위 분할을 환자 독립 검증이라고 표현하지 않음",
                     "병원 간 같은 환자의 연결 누락·유사 크롭은 이 검사로 배제할 수 없음",
                     "주평가 최소 그룹 수는 운영 기준이며 검정력 보장이 아님",
                     "최초 학습 전에 분할·평가층·미학습 제외 제품을 확정하고 이후 변경하지 않음"],
        "all_outputs_internal_only": True})
    return out
