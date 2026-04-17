"""构建 wiki/ 有向链接图，识别死链与孤岛。

用途：被 wiki-lint skill 调用。
- Phase 1 死链扫描：检查每个 [[link]] 目标是否存在
- Phase 2 孤岛检测：indegree=0 AND outdegree=0 AND mtime > 30d

遵守 CLAUDE.md：只读 wiki/，不修改。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path


WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")
ORPHAN_DAYS = 30


@dataclass
class WikiPage:
    path: Path
    slug: str
    outbound: list[str] = field(default_factory=list)
    inbound: list[str] = field(default_factory=list)
    mtime: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": str(self.path).replace("\\", "/"),
            "slug": self.slug,
            "outbound": self.outbound,
            "inbound": self.inbound,
            "mtime": self.mtime,
        }


def slug_from_path(path: Path, wiki_root: Path) -> str:
    """path = wiki/concepts/llm-wiki.md → slug = 'llm-wiki'（文件名去扩展）。"""
    return path.stem


def resolve_link(link: str, all_slugs: dict[str, Path]) -> Path | None:
    """把 [[foo]] 或 [[concepts/foo]] 解析为实际路径，None 表示死链。"""
    link = link.strip()
    bare = link.split("/")[-1]
    return all_slugs.get(bare) or all_slugs.get(link.split("/")[-1])


def extract_links(text: str) -> list[str]:
    """提取一页中所有 [[wiki-link]]。"""
    return [m.group(1).strip() for m in WIKI_LINK_PATTERN.finditer(text)]


def scan_wiki(wiki_root: Path) -> dict[str, WikiPage]:
    """扫描 wiki/ 所有 .md 文件，构建 WikiPage 字典（key=slug）。"""
    pages: dict[str, WikiPage] = {}

    for md in wiki_root.rglob("*.md"):
        if md.name.startswith("_"):
            continue
        slug = slug_from_path(md, wiki_root)
        pages[slug] = WikiPage(
            path=md.relative_to(wiki_root.parent),
            slug=slug,
            mtime=md.stat().st_mtime,
        )

    for slug, page in pages.items():
        full_path = wiki_root.parent / page.path
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[warn] cannot decode {page.path}, skipping", file=sys.stderr)
            continue

        for link in extract_links(text):
            page.outbound.append(link)
            bare = link.split("/")[-1]
            if bare in pages and bare != slug:
                pages[bare].inbound.append(slug)

    return pages


def find_dead_links(pages: dict[str, WikiPage], wiki_root: Path) -> list[tuple[str, str]]:
    """返回 [(source_slug, dead_link_target), ...]

    规则：
    - 简单 slug 链接 [[foo]]：在 pages 字典中查找
    - 路径式链接 [[../../raw/...]] 或 [[topic/foo]]：通过文件系统解析
    - 跨层引用 raw/ 是合法的，不计死链（只要文件存在）
    """
    dead: list[tuple[str, str]] = []
    vault = wiki_root.parent
    for slug, page in pages.items():
        page_dir = (vault / page.path).parent
        for link in page.outbound:
            # 路径式链接（含 / 或 .md 后缀）
            if "/" in link or link.endswith(".md"):
                target_rel = link if link.endswith(".md") else f"{link}.md"
                resolved = (page_dir / target_rel).resolve()
                if not resolved.exists():
                    dead.append((slug, link))
                continue
            # 简单 slug 链接
            if link not in pages:
                dead.append((slug, link))
    return dead


def find_orphans(pages: dict[str, WikiPage], days: int = ORPHAN_DAYS) -> list[str]:
    """返回孤岛页的 slug 列表。"""
    now = datetime.now().timestamp()
    threshold = now - days * 86400
    return [
        slug for slug, p in pages.items()
        if not p.inbound and not p.outbound and p.mtime < threshold
    ]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python link-graph.py <vault_root> [--json]", file=sys.stderr)
        return 2

    vault = Path(sys.argv[1]).resolve()
    wiki_root = vault / "wiki"
    json_mode = "--json" in sys.argv

    if not wiki_root.exists():
        print(f"[error] wiki/ not found at {wiki_root}", file=sys.stderr)
        return 1

    pages = scan_wiki(wiki_root)
    dead = find_dead_links(pages, wiki_root)
    orphans = find_orphans(pages)

    if json_mode:
        out = {
            "pages": {s: p.to_dict() for s, p in pages.items()},
            "dead_links": [{"source": s, "target": t} for s, t in dead],
            "orphans": orphans,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"=== Wiki link graph at {vault} ===")
        print(f"Pages: {len(pages)}")
        print(f"Dead links: {len(dead)}")
        for source, target in dead:
            print(f"  ! [[{target}]] in wiki/.../{source}.md")
        print(f"\nOrphans (>{ORPHAN_DAYS}d, no links): {len(orphans)}")
        for slug in orphans:
            print(f"  - {slug}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
