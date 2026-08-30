import collections
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from implant_analysis.audit import audit
from implant_analysis.common import load_config, read_csv, write_csv
from implant_analysis.split import split_draft


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_file = self.root / "site.json"
        template = Path(__file__).parents[1] / "configs/site.example.json"
        self.config = json.loads(template.read_text())
        self.config.update(data_root=str(self.root / "data"), work_dir=str(self.root / "work"))
        self.labels = self.root / "data" / self.config["label_root"]
        self.images = self.root / "data" / self.config["image_root"]
        self.labels.mkdir(parents=True)
        self.images.mkdir(parents=True)
        self.group_rows = []

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self):
        self.config_file.write_text(json.dumps(self.config, ensure_ascii=False))
        return load_config(self.config_file)

    def add(self, name, number, model="M", group=None, source=None, maker="Maker"):
        rel = Path("병원별데이터_01") / (name + ".json")
        label = self.labels / rel
        image = self.images / rel.with_suffix(".png")
        label.parent.mkdir(exist_ok=True)
        image.parent.mkdir(exist_ok=True)
        record = {"일련번호": f"{number:04}", "촬영방법": "파노라마", "제조사": maker,
                  "모델명": model, "직경": "4.3", "길이": "10.0", "식립위치": "26",
                  "식립일자": "20200101", "촬영일자": "20200201", "촬영병원": "가상병원"}
        label.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8-sig")
        pixels = bytes((number * 7 + x * 3) % 256 for x in range(32 * 48))
        Image.frombytes("L", (32, 48), pixels).save(image)
        self.group_rows.append({"label_relpath": rel.as_posix(), "group_key": group or f"P{number}",
                                "source_image_key": source or f"S{number}"})
        return label, image

    def verified_groups(self):
        path = self.root / "groups.csv"
        write_csv(path, self.group_rows, ["label_relpath", "group_key", "source_image_key"])
        self.config["pairing"]["verified"] = True
        self.config["grouping"] = {"csv": str(path), "kind": "patient", "verified": True,
                                   "verification_note": "SYNTHETIC ONLY: known generated groups"}

    def test_pdf_fields_bom_and_no_implicit_patient_id(self):
        self.add("a", 1)
        out = audit(self.cfg(), "first")
        row = read_csv(out / "manifest.csv")[0]
        self.assertEqual(row["status"], "eligible")
        self.assertEqual(row["serial"], "0001")
        self.assertEqual(row["hospital"], "가상병원")
        self.assertEqual(row["group_id"], "")
        self.assertEqual(json.loads(row["class_key"]), ["Maker", "M"])

    def test_missing_and_ambiguous_images_are_not_guessed(self):
        _, a = self.add("a", 1)
        a.rename(a.with_name("different.png"))
        _, b = self.add("b", 2)
        Image.open(b).save(b.with_suffix(".jpg"))
        out = audit(self.cfg(), "ambiguity")
        kinds = {r["issue"] for r in read_csv(out / "issues.csv")}
        self.assertIn("image_missing", kinds)
        self.assertIn("image_ambiguous", kinds)
        self.assertTrue(all(r["status"] == "excluded" for r in read_csv(out / "manifest.csv")))

    def test_duplicate_pixels_with_conflicting_labels_excluded(self):
        self.add("a", 1, model="A")
        self.add("b", 1, model="B")
        out = audit(self.cfg(), "conflict")
        self.assertTrue(all(r["status"] == "excluded" for r in read_csv(out / "manifest.csv")))
        self.assertEqual(len(read_csv(out / "duplicate_pixels.csv")), 2)

    def test_corrupt_image_and_duplicate_json_keys_reported(self):
        _, image = self.add("a", 1)
        image.write_bytes(b"not an image")
        label, _ = self.add("b", 2)
        label.write_text('{"제조사":"A", "제조사":"B", "모델명":"M"}')
        out = audit(self.cfg(), "corrupt")
        kinds = {r["issue"] for r in read_csv(out / "issues.csv")}
        self.assertIn("image_decode_error", kinds)
        self.assertIn("json_parse_or_schema_error", kinds)

    def test_explicit_mapping_handles_different_names(self):
        label, image = self.add("a", 1)
        moved = image.with_name("different.png")
        image.rename(moved)
        path = self.root / "pairing.csv"
        write_csv(path, [{"label_relpath": label.relative_to(self.labels).as_posix(),
                          "image_relpath": moved.relative_to(self.images).as_posix()}],
                  ["label_relpath", "image_relpath"])
        self.config["pairing"] = {"mode": "mapping_csv", "csv": str(path), "verified": True}
        out = audit(self.cfg(), "mapping")
        self.assertEqual(read_csv(out / "manifest.csv")[0]["status"], "eligible")

    def test_unverified_grouping_refuses_split(self):
        for i in range(8):
            self.add(str(i), i)
        self.config["pairing"]["verified"] = True
        out = audit(self.cfg(), "unverified")
        with self.assertRaisesRegex(ValueError, "그룹키"):
            split_draft(self.cfg(), out, "no")

    def test_partial_audit_refuses_split(self):
        for i in range(8):
            self.add(str(i), i)
        self.verified_groups()
        out = audit(self.cfg(), "partial", limit=3)
        with self.assertRaisesRegex(ValueError, "부분 감사"):
            split_draft(self.cfg(), out, "no")

    def test_missing_group_key_refuses_split(self):
        for i in range(8):
            self.add(str(i), i)
        self.verified_groups()
        self.group_rows.pop()
        self.verified_groups()
        out = audit(self.cfg(), "missinggroup")
        with self.assertRaisesRegex(ValueError, "그룹키"):
            split_draft(self.cfg(), out, "no")

    def test_grouped_deduplicated_split_and_reproducibility(self):
        for i in range(24):
            self.add(str(i), i, model="A" if i % 2 else "B", group=f"P{i // 2}")
        self.add("same_pixels_new_group", 1, model="A", group="linked-via-duplicate")
        self.add("linked_source", 90, model="A", group="another-group", source="S1")
        self.verified_groups()
        out = audit(self.cfg(), "all")
        first = split_draft(self.cfg(), out, "one")
        second = split_draft(self.cfg(), out, "two")
        a, b = read_csv(first / "split_manifest.csv"), read_csv(second / "split_manifest.csv")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 25)
        self.assertEqual(len(read_csv(first / "removed_exact_duplicates.csv")), 1)
        for key in ("group_id", "source_image_group_id", "pixel_sha256", "leakage_group_id"):
            seen = collections.defaultdict(set)
            for row in a:
                if row[key]:
                    seen[row[key]].add(row["split"])
            self.assertTrue(all(len(parts) == 1 for parts in seen.values()))
        self.assertEqual({r["split"] for r in a}, {"train", "dev", "calibration", "test"})

    def test_same_model_name_different_manufacturers_stay_distinct(self):
        self.add("a", 1, maker="A")
        self.add("b", 2, maker="B")
        out = audit(self.cfg(), "manufacturers")
        self.assertEqual(len({r["class_key"] for r in read_csv(out / "manifest.csv")}), 2)

    def test_audit_outputs_are_not_overwritten(self):
        self.add("a", 1)
        audit(self.cfg(), "once")
        with self.assertRaises(FileExistsError):
            audit(self.cfg(), "once")

    def test_work_directory_inside_raw_data_is_rejected(self):
        self.add("a", 1)
        self.config["work_dir"] = str(self.root / "data" / "results")
        with self.assertRaisesRegex(ValueError, "원본 데이터 루트 밖"):
            audit(self.cfg(), "bad_output")


if __name__ == "__main__":
    unittest.main()
