"""Copy completed generation files to a durable backup, then verify hashes and a restore.

Every file in data/manifests/generations.json is copied to --dest, its SHA-256 is checked against
the manifest, and the backup is restored into a scratch directory and checked again. The result
is appended to the manifest's "backups" list.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hdetect.generation.report import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--manifest", default="data/manifests/generations.json", type=Path)
    args = parser.parse_args()

    record = json.loads(args.manifest.read_text())
    if not record["complete"]:
        raise SystemExit("generation is not complete; back up only finished, read-only files")
    files = record["files"]
    args.dest.mkdir(parents=True, exist_ok=True)

    checks = {}
    for split, meta in files.items():
        src = Path(meta["path"])
        if sha256_file(src) != meta["sha256"]:
            raise SystemExit(f"{src} no longer matches the manifest")
        dst = args.dest / src.name
        if dst.exists() and sha256_file(dst) == meta["sha256"]:
            pass
        else:
            if dst.exists():
                dst.chmod(0o644)
            shutil.copy2(src, dst)
            dst.chmod(0o444)
        checks[split] = sha256_file(dst) == meta["sha256"]
    shutil.copy2(args.manifest, args.dest / args.manifest.name)

    with tempfile.TemporaryDirectory() as scratch:
        restored = {}
        for split, meta in files.items():
            target = Path(scratch) / Path(meta["path"]).name
            shutil.copy2(args.dest / Path(meta["path"]).name, target)
            restored[split] = sha256_file(target) == meta["sha256"]

    ok = all(checks.values()) and all(restored.values())
    record.setdefault("backups", []).append({
        "dest": str(args.dest),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hashes_match": checks,
        "restore_matches": restored,
        "ok": ok,
    })
    args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    shutil.copy2(args.manifest, args.dest / args.manifest.name)
    print(json.dumps(record["backups"][-1], indent=2))
    if not ok:
        raise SystemExit("backup verification failed")


if __name__ == "__main__":
    main()
