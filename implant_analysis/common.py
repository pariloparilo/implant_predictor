import csv
import hashlib
import json
import unicodedata
from pathlib import Path


def normalized(value):
    return unicodedata.normalize("NFC", str(value)).strip()


def token(*values):
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    config["_config_file"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def config_path(config, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_config_dir"]) / path).resolve()


def new_output(config, command, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run-id에는 영문·숫자·밑줄·하이픈만 사용하세요.")
    work_root = config_path(config, config["work_dir"])
    data_root = config_path(config, config["data_root"])
    if work_root.is_relative_to(data_root):
        raise ValueError("work_dir는 원본 데이터 루트 밖의 별도 작업 폴더로 지정하세요.")
    out = work_root / command / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    write_json(out / "config_snapshot.json", snapshot)
    return out
