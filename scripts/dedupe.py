"""raw/ 文件 SHA256 去重与 manifest 对比。

用途：被 wiki-ingest skill 调用，扫描 raw/ 下所有文件，计算 SHA256，
与 raw/_manifest.jsonl 对比，输出新增/修改/已存在分类。

遵守 CLAUDE.md 分层契约：只读 raw，不修改。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path


CST = timezone(timedelta(hours=8))  # 中国标准时间


@dataclass(frozen=True)
class RawFile:
    path: Path
    sha256: str
    size: int


def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    """流式计算文件 SHA256。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_raw(raw_root: Path) -> list[RawFile]:
    """扫描 raw/ 下所有文件（递归，跳过隐藏文件和 _manifest.jsonl）。"""
    if not raw_root.exists():
        return []

    files: list[RawFile] = []
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.name == "_manifest.jsonl":
            continue
        files.append(RawFile(
            path=p.relative_to(raw_root.parent),
            sha256=compute_sha256(p),
            size=p.stat().st_size,
        ))
    return files


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """读取 manifest.jsonl，返回 {path_str: record} 字典。"""
    if not manifest_path.exists():
        return {}

    records: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["path"]] = rec
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[warn] manifest line {line_num} invalid: {e}", file=sys.stderr)
    return records


def diff(raw_files: list[RawFile], manifest: dict[str, dict]) -> dict[str, list]:
    """输出分类：new/modified/unchanged。"""
    new, modified, unchanged = [], [], []
    for rf in raw_files:
        key = str(rf.path).replace("\\", "/")
        rec = manifest.get(key)
        if rec is None:
            new.append(rf)
        elif rec.get("sha256") != rf.sha256:
            modified.append(rf)
        else:
            unchanged.append(rf)
    return {"new": new, "modified": modified, "unchanged": unchanged}


def format_manifest_entry(rf: RawFile, source: str = "local",
                          wiki_pages: list[str] | None = None) -> str:
    """生成一行 manifest JSON。"""
    entry = {
        "path": str(rf.path).replace("\\", "/"),
        "sha256": rf.sha256,
        "size": rf.size,
        "source": source,
        "ingested_at": datetime.now(CST).isoformat(timespec="seconds"),
        "wiki_pages": wiki_pages or [],
    }
    return json.dumps(entry, ensure_ascii=False)


def main() -> int:
    """CLI 入口：python dedupe.py <vault_root>"""
    if len(sys.argv) < 2:
        print("Usage: python dedupe.py <vault_root>", file=sys.stderr)
        return 2

    vault = Path(sys.argv[1]).resolve()
    raw_root = vault / "raw"
    manifest_path = raw_root / "_manifest.jsonl"

    if not raw_root.exists():
        print(f"[error] raw/ not found at {raw_root}", file=sys.stderr)
        return 1

    raw_files = scan_raw(raw_root)
    manifest = load_manifest(manifest_path)
    result = diff(raw_files, manifest)

    print(f"=== Raw scan at {vault} ===")
    print(f"Total: {len(raw_files)} files")
    print(f"New: {len(result['new'])}")
    print(f"Modified: {len(result['modified'])}  (CLAUDE.md 违规：raw 应只追加！)")
    print(f"Unchanged: {len(result['unchanged'])}")

    if result["new"]:
        print("\n--- New ---")
        for rf in result["new"]:
            print(f"  + {rf.path}  ({rf.sha256[:12]}... {rf.size} bytes)")

    if result["modified"]:
        print("\n--- Modified (警告) ---")
        for rf in result["modified"]:
            print(f"  ! {rf.path}  ({rf.sha256[:12]}...)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
