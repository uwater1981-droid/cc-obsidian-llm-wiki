#!/usr/bin/env python3
"""
Mirror uwater Obsidian vault into raw/notes/public/uwater/ with kebab-case + pinyin slugs.
Append to raw/_manifest.jsonl per CLAUDE.md constitution (section 3 & 10).

Usage:
  python scripts/mirror_uwater.py --dry-run [--sample 20]
  python scripts/mirror_uwater.py            # full execution
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pypinyin import lazy_pinyin, Style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("F:/Icloud/iCloudDrive/uwater")
TARGET_ROOT = PROJECT_ROOT / "raw" / "notes" / "public" / "uwater"
MANIFEST = PROJECT_ROOT / "raw" / "_manifest.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs" / "ingest"

EXCLUDE_DIR_NAMES = {".trash", ".obsidian", "node_modules", ".git"}
INCLUDE_EXTS = {".md"}
TZ = timezone(timedelta(hours=8))


def slugify(text: str) -> str:
    """Convert Chinese/mixed text to kebab-case slug."""
    parts: list[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            py = lazy_pinyin(ch, style=Style.NORMAL)
            parts.append(py[0] if py else "")
        else:
            parts.append(ch.lower())
    s = "".join(parts)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded(rel: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in rel.parts)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for p in SOURCE.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in INCLUDE_EXTS:
            continue
        rel = p.relative_to(SOURCE)
        if is_excluded(rel):
            continue
        files.append(p)
    files.sort()
    return files


def plan_copies(files: list[Path]) -> tuple[list[tuple[Path, Path]], list[tuple[str, str]]]:
    """Return (actions, collisions). Actions is [(src, target)]."""
    actions: list[tuple[Path, Path]] = []
    used: set[str] = set()
    collisions: list[tuple[str, str]] = []

    for src in files:
        rel = src.relative_to(SOURCE)
        dir_parts = [slugify(p) for p in rel.parent.parts]
        stem = slugify(rel.stem)
        target_dir = TARGET_ROOT.joinpath(*dir_parts) if dir_parts else TARGET_ROOT
        base_target = target_dir / f"{stem}.md"

        target = base_target
        n = 1
        while str(target).lower() in used:
            target = target_dir / f"{stem}-{n}.md"
            n += 1
        if n > 1:
            collisions.append((str(src), str(target)))
        used.add(str(target).lower())
        actions.append((src, target))

    return actions, collisions


def main() -> int:
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    sample = 0
    if "--sample" in argv:
        idx = argv.index("--sample")
        sample = int(argv[idx + 1]) if idx + 1 < len(argv) else 0

    if not SOURCE.exists():
        print(f"ERROR: source does not exist: {SOURCE}", file=sys.stderr)
        return 2

    print(f"Source : {SOURCE}")
    print(f"Target : {TARGET_ROOT}")
    files = collect_files()
    print(f"Found  : {len(files)} .md files (excluding .trash, .obsidian)")

    if sample > 0:
        files = files[:sample]
        print(f"Sample : first {sample}")

    actions, collisions = plan_copies(files)
    print(f"Plan   : {len(actions)} copies, {len(collisions)} collisions resolved")

    if dry_run:
        print("\n=== DRY RUN — first 20 mappings ===")
        for src, tgt in actions[:20]:
            rel_src = src.relative_to(SOURCE)
            rel_tgt = tgt.relative_to(PROJECT_ROOT)
            print(f"  {rel_src}")
            print(f"    -> {rel_tgt}")
        if collisions[:5]:
            print("\n=== Sample collisions ===")
            for s, t in collisions[:5]:
                print(f"  {s}\n    -> {t}")
        return 0

    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    errors: list[tuple[str, str]] = []

    for i, (src, tgt) in enumerate(actions, 1):
        try:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            sha = sha256_file(tgt)
            rel_tgt = tgt.relative_to(PROJECT_ROOT)
            entry = {
                "path": str(rel_tgt).replace("\\", "/"),
                "sha256": sha,
                "source": "local",
                "original": str(src).replace("\\", "/"),
                "ingested_at": datetime.now(TZ).isoformat(timespec="seconds"),
            }
            manifest_lines.append(json.dumps(entry, ensure_ascii=False))
            if i % 100 == 0:
                print(f"  copied {i}/{len(actions)}")
        except Exception as exc:
            errors.append((str(src), repr(exc)))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        for line in manifest_lines:
            fh.write(line + "\n")

    report = run_dir / "report.md"
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(f"# uwater ingest — {stamp}\n\n")
        fh.write(f"- Source: `{SOURCE}`\n")
        fh.write(f"- Target: `{TARGET_ROOT.relative_to(PROJECT_ROOT)}`\n")
        fh.write(f"- Manifest appended: `{MANIFEST.relative_to(PROJECT_ROOT)}`\n")
        fh.write(f"- Files planned: {len(actions)}\n")
        fh.write(f"- Files copied:  {len(manifest_lines)}\n")
        fh.write(f"- Collisions:    {len(collisions)}\n")
        fh.write(f"- Errors:        {len(errors)}\n\n")
        if collisions:
            fh.write("## Collisions (first 50)\n\n")
            for s, t in collisions[:50]:
                fh.write(f"- `{s}` -> `{t}`\n")
            fh.write("\n")
        if errors:
            fh.write("## Errors (first 50)\n\n")
            for s, e in errors[:50]:
                fh.write(f"- `{s}`: {e}\n")

    print(f"\nDone. Copied {len(manifest_lines)}/{len(actions)}, errors {len(errors)}.")
    print(f"Report  : {report.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
