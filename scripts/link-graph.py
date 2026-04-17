"""构建 wiki/ 有向链接图，识别死链与孤岛。

用途：被 wiki-lint skill 调用。
- Phase 1 死链扫描：检查每个 [[link]] / [text](path.md) 目标是否存在
- Phase 2 孤岛检测：indegree=0 AND outdegree=0 AND mtime > 30d

安全性：
- 所有路径解析都做 vault containment check（防目录遍历）
- 页面以 vault-relative 路径作 key（避免同名文件冲突）
- 处理 UnicodeDecodeError 不沉默

遵守 CLAUDE.md：只读 wiki/，不修改。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
# 排除 http(s) 外链
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((?!https?://)([^)]+\.md)\)")
ORPHAN_DAYS = 30


@dataclass
class WikiPage:
    rel_path: str  # vault-relative posix path, e.g. "wiki/concepts/llm-wiki.md"
    slug: str      # file stem for display
    outbound: list[str] = field(default_factory=list)
    inbound: list[str] = field(default_factory=list)
    mtime: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "rel_path": self.rel_path,
            "slug": self.slug,
            "outbound": self.outbound,
            "inbound": self.inbound,
            "mtime": self.mtime,
        }


def page_key(md_path: Path, wiki_root: Path) -> str:
    """vault-relative posix path — unique even when stems collide."""
    return str(md_path.relative_to(wiki_root.parent)).replace("\\", "/")


def extract_links(text: str) -> list[str]:
    """提取所有 wiki-link 和 md-link（统一返回字符串列表）。"""
    wiki_links = [m.group(1).strip() for m in WIKI_LINK_PATTERN.finditer(text)]
    md_links = [m.group(2).strip() for m in MD_LINK_PATTERN.finditer(text)]
    return wiki_links + md_links


def scan_wiki(wiki_root: Path) -> dict[str, WikiPage]:
    """扫描 wiki/ 所有 .md 文件，key=vault-relative path。"""
    pages: dict[str, WikiPage] = {}
    stem_index: dict[str, list[str]] = {}  # stem → list of keys (for fuzzy resolve)

    for md in wiki_root.rglob("*.md"):
        if md.name.startswith("_"):
            continue
        key = page_key(md, wiki_root)
        pages[key] = WikiPage(
            rel_path=key,
            slug=md.stem,
            mtime=md.stat().st_mtime,
        )
        stem_index.setdefault(md.stem, []).append(key)

    for key, page in pages.items():
        full_path = wiki_root.parent / page.rel_path
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[warn] cannot decode {page.rel_path}, skipping", file=sys.stderr)
            continue
        except OSError as e:
            print(f"[warn] cannot read {page.rel_path}: {e}", file=sys.stderr)
            continue

        for link in extract_links(text):
            page.outbound.append(link)

    # 二次遍历建 inbound，用 stem fuzzy 匹配
    for key, page in pages.items():
        for link in page.outbound:
            target_key = _resolve_link_to_key(link, page, pages, stem_index, wiki_root)
            if target_key and target_key in pages and target_key != key:
                pages[target_key].inbound.append(key)

    return pages


def _resolve_link_to_key(link: str, source: WikiPage, pages: dict[str, WikiPage],
                          stem_index: dict[str, list[str]], wiki_root: Path) -> str | None:
    """把一个 link 字符串解析为 pages 字典的 key（vault-relative path）。

    返回 None 表示死链 / 跨层到 raw（raw 通过文件系统判断，不在此处）。
    """
    vault = wiki_root.parent

    # 路径式链接
    if "/" in link or link.endswith(".md"):
        target_rel = link if link.endswith(".md") else f"{link}.md"
        source_dir = (vault / source.rel_path).parent
        try:
            resolved = (source_dir / target_rel).resolve()
            resolved.relative_to(vault.resolve())  # containment check
        except (ValueError, OSError):
            return None
        try:
            target_rel_posix = str(resolved.relative_to(vault.resolve())).replace("\\", "/")
        except ValueError:
            return None
        return target_rel_posix

    # 纯 slug，fuzzy match
    candidates = stem_index.get(link, [])
    if len(candidates) == 1:
        return candidates[0]
    return None  # 0 or >1 matches — ambiguous


def find_dead_links(pages: dict[str, WikiPage], wiki_root: Path) -> list[tuple[str, str]]:
    """返回 [(source_key, dead_link_target), ...]

    对于路径式链接，通过文件系统 + vault containment check 判断存在性。
    对于 slug 链接，通过 stem_index 查找（0 或多匹配都视为死链）。
    """
    vault = wiki_root.parent.resolve()
    stem_index: dict[str, list[str]] = {}
    for key, p in pages.items():
        stem_index.setdefault(p.slug, []).append(key)

    dead: list[tuple[str, str]] = []
    for key, page in pages.items():
        source_dir = (vault / page.rel_path).parent
        for link in page.outbound:
            if "/" in link or link.endswith(".md"):
                target_rel = link if link.endswith(".md") else f"{link}.md"
                try:
                    resolved = (source_dir / target_rel).resolve()
                    resolved.relative_to(vault)
                except (ValueError, OSError):
                    dead.append((key, link))
                    continue
                if not resolved.exists():
                    dead.append((key, link))
                continue
            # slug 链接
            candidates = stem_index.get(link, [])
            if len(candidates) == 0:
                dead.append((key, link))
            elif len(candidates) > 1:
                dead.append((key, f"{link} (ambiguous: {len(candidates)} matches)"))
    return dead


def find_orphans(pages: dict[str, WikiPage], days: int = ORPHAN_DAYS) -> list[str]:
    """返回孤岛页的 key 列表。"""
    now = datetime.now().timestamp()
    threshold = now - days * 86400
    return [
        key for key, p in pages.items()
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
            "pages": {k: p.to_dict() for k, p in pages.items()},
            "dead_links": [{"source": s, "target": t} for s, t in dead],
            "orphans": orphans,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"=== Wiki link graph at {vault} ===")
        print(f"Pages: {len(pages)}")
        print(f"Dead links: {len(dead)}")
        for source, target in dead:
            print(f"  ! [[{target}]] in {source}")
        print(f"\nOrphans (>{ORPHAN_DAYS}d, no links): {len(orphans)}")
        for key in orphans:
            print(f"  - {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
