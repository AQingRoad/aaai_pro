#!/usr/bin/env python3
"""为 CDs_and_Vinyl 归档生成逐文件大小、行数和 SHA256 清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix not in {".jsonl", ".txt", ".md", ".log"}:
        return None
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manu-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = [
        args.manu_src / "datas/CDs_and_Vinyl",
        args.manu_src / "eval_results/CDs_and_Vinyl",
        args.manu_src / "experiments/CDs_and_Vinyl",
    ]
    output_resolved = args.output.resolve()
    files = []
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() != output_resolved):
            stat = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(args.manu_src)),
                    "bytes": stat.st_size,
                    "lines": line_count(path),
                    "sha256": sha256(path),
                }
            )

    payload = {
        "archive_date": "2026-07-16",
        "scope": "CDs_and_Vinyl existing data, prompts, reports, audits and frozen aligned run",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
