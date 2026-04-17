"""识别 wiki/ 中的陈旧内容。

检查两类：
1. 未勾选的 TODO：`- [ ] ...` 所在文件 frontmatter updated 或 mtime 超过 90 天
2. 陈旧页面：frontmatter updated 字段超过 180 天

安全性：
- FRONTMATTER_PATTERN 支持 CRLF（Windows iCloud 常见）
- UnicodeDecodeError 和 OSError 都记录 warning
- 统一使用 CST 时区（与 icloud-lock.py 保持一致）

遵守 CLAUDE.md：只读 wiki/。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal


CST = timezone(timedelta(hours=8))
TODO_PATTERN = re.compile(r"^\s*-\s*\[\s*\]\s+(.+)$", re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r"^---\r?\n(.*?)\r?\n---", re.DOTALL)
UPDATED_PATTERN = re.compile(r"^updated:\s*(\S+)", re.MULTILINE)

STALE_TODO_DAYS = 90
STALE_PAGE_DAYS = 180


@dataclass
class StaleItem:
    path: str
    kind: Literal["todo", "page"]
    detail: str
    age_days: int


def parse_frontmatter_updated(text: str) -> datetime | None:
    """提取 frontmatter 的 updated: 字段，返回 CST-aware datetime。"""
    fm_match = FRONTMATTER_PATTERN.match(text)
    if not fm_match:
        return None
    fm = fm_match.group(1)
    upd_match = UPDATED_PATTERN.search(fm)
    if not upd_match:
        return None
    try:
        naive = datetime.strptime(upd_match.group(1), "%Y-%m-%d")
        return naive.replace(tzinfo=CST)
    except ValueError:
        return None


def file_age_days(md: Path, updated: datetime | None) -> int:
    """优先用 frontmatter updated，否则用文件 mtime。"""
    now = datetime.now(CST)
    if updated is not None:
        return (now - updated).days
    mtime_naive = datetime.fromtimestamp(md.stat().st_mtime)
    mtime = mtime_naive.replace(tzinfo=CST)
    return (now - mtime).days


def _read_md(md: Path) -> str | None:
    """读取 md 文件，处理所有常见异常。"""
    try:
        return md.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[warn] cannot decode {md}, skipping", file=sys.stderr)
        return None
    except OSError as e:
        print(f"[warn] cannot read {md}: {e}", file=sys.stderr)
        return None


def scan_stale_todos(wiki_root: Path, threshold_days: int) -> list[StaleItem]:
    """扫描所有 `- [ ] ...` 项，按 frontmatter updated 或文件 mtime 判断陈旧。"""
    items: list[StaleItem] = []

    for md in wiki_root.rglob("*.md"):
        text = _read_md(md)
        if text is None:
            continue
        updated = parse_frontmatter_updated(text)
        age = file_age_days(md, updated)
        if age < threshold_days:
            continue

        for todo_match in TODO_PATTERN.finditer(text):
            items.append(StaleItem(
                path=str(md.relative_to(wiki_root.parent)).replace("\\", "/"),
                kind="todo",
                detail=todo_match.group(1)[:80],
                age_days=age,
            ))
    return items


def scan_stale_pages(wiki_root: Path, threshold_days: int) -> list[StaleItem]:
    """扫描 frontmatter updated 超过阈值的页。"""
    items: list[StaleItem] = []

    for md in wiki_root.rglob("*.md"):
        if md.name.startswith("_"):
            continue
        text = _read_md(md)
        if text is None:
            continue

        updated = parse_frontmatter_updated(text)
        if updated is None:
            continue
        age = (datetime.now(CST) - updated).days
        if age >= threshold_days:
            items.append(StaleItem(
                path=str(md.relative_to(wiki_root.parent)).replace("\\", "/"),
                kind="page",
                detail=f"updated: {updated.date()}",
                age_days=age,
            ))
    return items


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python stale-check.py <vault_root>", file=sys.stderr)
        return 2

    vault = Path(sys.argv[1]).resolve()
    wiki_root = vault / "wiki"

    if not wiki_root.exists():
        print(f"[error] wiki/ not found at {wiki_root}", file=sys.stderr)
        return 1

    stale_todos = scan_stale_todos(wiki_root, STALE_TODO_DAYS)
    stale_pages = scan_stale_pages(wiki_root, STALE_PAGE_DAYS)

    print(f"=== Stale check at {vault} ===")
    print(f"Stale TODOs (>{STALE_TODO_DAYS}d): {len(stale_todos)}")
    for item in stale_todos:
        print(f"  - [{item.age_days}d] {item.path}: {item.detail}")

    print(f"\nStale pages (updated >{STALE_PAGE_DAYS}d ago): {len(stale_pages)}")
    for item in stale_pages:
        print(f"  - [{item.age_days}d] {item.path}  ({item.detail})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
