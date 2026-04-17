#!/usr/bin/env python3
"""
Fetch Andrej Karpathy canonical outputs into raw/ with manifest append.

Handlers:
- blogs    -> WebFetch+markdownify -> raw/articles/karpathy/<slug>.md
- youtube  -> youtube-transcript-api -> raw/transcripts/karpathy/_raw/<slug>.md
             (LLM post-processing happens in a separate step outside this script)
- gists    -> gh gist view --raw     -> raw/gists/karpathy/<slug>.md
- threads  -> WebFetch text          -> raw/notes/public/karpathy-threads.md (aggregated)

Usage:
  python scripts/fetch_karpathy.py --dry-run                  # print plan
  python scripts/fetch_karpathy.py                            # run all
  python scripts/fetch_karpathy.py --filter blogs             # only blogs
  python scripts/fetch_karpathy.py --filter youtube --limit 2 # first 2 videos
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify as md

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    _YT_OK = True
except ImportError:
    _YT_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_YAML = PROJECT_ROOT / "scripts" / "karpathy_sources.yaml"
MANIFEST = PROJECT_ROOT / "raw" / "_manifest.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs" / "ingest"

ART_DIR = PROJECT_ROOT / "raw" / "articles" / "karpathy"
YT_RAW_DIR = PROJECT_ROOT / "raw" / "transcripts" / "karpathy" / "_raw"
GIST_DIR = PROJECT_ROOT / "raw" / "gists" / "karpathy"
THREADS_FILE = PROJECT_ROOT / "raw" / "notes" / "public" / "karpathy-threads.md"

TZ = timezone(timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (cc+obsidian learning vault) Karpathy Fetcher"}


@dataclass(frozen=True)
class FetchResult:
    kind: str
    slug: str
    target: Path
    ok: bool
    error: str = ""


@dataclass
class RunLog:
    results: list[FetchResult] = field(default_factory=list)

    def add(self, r: FetchResult) -> None:
        self.results.append(r)

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "ok": sum(1 for r in self.results if r.ok),
            "fail": sum(1 for r in self.results if not r.ok),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_paths() -> set[str]:
    if not MANIFEST.exists():
        return set()
    paths: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
                paths.add(entry.get("path", ""))
            except json.JSONDecodeError:
                continue
    return paths


def append_manifest(path: Path, source: str, extra: dict | None = None) -> None:
    rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    entry = {
        "path": rel,
        "sha256": sha256_file(path),
        "source": source,
        "ingested_at": datetime.now(TZ).isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_frontmatter(fh, meta: dict) -> None:
    fh.write("---\n")
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, list):
            fh.write(f"{k}: [{', '.join(json.dumps(x, ensure_ascii=False) for x in v)}]\n")
        else:
            fh.write(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, str) and (':' in v or v.startswith('-')) else v}\n")
    fh.write("---\n\n")


# ----- handlers -----


def fetch_blog(url: str, slug: str) -> Path:
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(post|content|article|entry)", re.I))
        or soup.body
    )
    if main is None:
        main = soup

    for tag in main.select("script, style, nav, header, footer, .site-header, .site-footer"):
        tag.decompose()

    # Title extraction priority:
    # 1. First h1 inside main (real post title on Jekyll/Bearblog)
    # 2. Open Graph og:title meta (fallback)
    # 3. <title> (often site-wide generic name — last resort)
    title = ""
    h1 = main.find("h1") if hasattr(main, "find") else None
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()
    if not title:
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else slug

    body_md = md(str(main), heading_style="ATX").strip()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    target = ART_DIR / f"{slug}.md"
    with target.open("w", encoding="utf-8") as fh:
        write_frontmatter(
            fh,
            {
                "title": title,
                "url": url,
                "author": "Andrej Karpathy",
                "slug": slug,
                "fetched_at": datetime.now(TZ).isoformat(timespec="seconds"),
                "type": "blog-post",
            },
        )
        fh.write(body_md)
        fh.write("\n")
    return target


def _fetch_youtube_metadata(video_id: str) -> dict:
    """Fetch YT page HTML and extract title/description/publish date/chapters.

    Returns {} if anything fails. Used when transcript API is blocked.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        resp = requests.get(url, headers=UA, timeout=20)
        resp.raise_for_status()
    except Exception:
        return {}
    html = resp.text

    meta = {"url": url}
    m_title = re.search(r'"title":"((?:[^"\\]|\\.)*?)","lengthSeconds"', html)
    if m_title:
        meta["page_title"] = bytes(m_title.group(1), "utf-8").decode("unicode_escape")
    m_len = re.search(r'"lengthSeconds":"(\d+)"', html)
    if m_len:
        meta["length_seconds"] = int(m_len.group(1))
    m_date = re.search(r'"publishDate":"(\d{4}-\d{2}-\d{2})"', html)
    if m_date:
        meta["publish_date"] = m_date.group(1)
    m_desc = re.search(
        r'"shortDescription":"((?:[^"\\]|\\.)*?)","isCrawlable"', html
    )
    if m_desc:
        desc_raw = m_desc.group(1)
        # Unescape common sequences
        desc = desc_raw.replace("\\n", "\n").replace('\\"', '"').replace("\\u0026", "&").replace("\\/", "/")
        meta["description"] = desc
    return meta


def fetch_youtube_raw(video_id: str, slug: str, title: str) -> Path:
    """Write a raw transcript file.

    Strategy:
    1. Try youtube-transcript-api (usually IP-blocked in this env)
    2. On failure, fall back to metadata-only (title/desc/chapters/length from page HTML)

    The output always lives under _raw/ and is processed by a later LLM pass.
    """
    YT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = YT_RAW_DIR / f"{slug}.md"

    segments: list[tuple[float, str]] = []
    transcript_error = ""
    if _YT_OK:
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
            segments = [(s.start, s.text) for s in fetched]
        except Exception as e:
            transcript_error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    meta = _fetch_youtube_metadata(video_id) if not segments else {}

    frontmatter = {
        "title": title,
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "author": "Andrej Karpathy",
        "slug": slug,
        "fetched_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "type": "youtube-transcript-raw" if segments else "youtube-metadata-only",
        "segments": len(segments),
    }
    if meta.get("length_seconds"):
        frontmatter["length_seconds"] = meta["length_seconds"]
    if meta.get("publish_date"):
        frontmatter["publish_date"] = meta["publish_date"]

    with target.open("w", encoding="utf-8") as fh:
        write_frontmatter(fh, frontmatter)
        fh.write(f"# {title}\n\n")
        fh.write(f"> Source: https://www.youtube.com/watch?v={video_id}\n\n")
        if segments:
            for start, text in segments:
                mm = int(start) // 60
                ss = int(start) % 60
                fh.write(f"[{mm:02d}:{ss:02d}] {text}\n")
        else:
            fh.write("## Transcript status\n\n")
            fh.write(
                f"> 字幕抓取失败:{transcript_error or '无可用字幕'}。\n"
                "> 已落盘 metadata + 视频说明,后续可人工补字幕或用外部工具补录。\n\n"
            )
            if meta.get("publish_date"):
                fh.write(f"- Publish date: {meta['publish_date']}\n")
            if meta.get("length_seconds"):
                mm = meta["length_seconds"] // 60
                ss = meta["length_seconds"] % 60
                fh.write(f"- Length: {mm}m{ss:02d}s\n")
            fh.write("\n## Description\n\n")
            fh.write(meta.get("description", "_(页面未返回说明栏,可能 YT UI 变更)_") + "\n")
    return target


def fetch_gist(gist_id: str, slug: str, note: str) -> Path:
    """Fetch a public GitHub gist.

    Primary: unauthenticated REST API (returns file list + content).
    Fallback: `gist.githubusercontent.com/<user>/<id>/raw` (works when API 502s,
    but only returns the single-file content without filename metadata).
    """
    api_url = f"https://api.github.com/gists/{gist_id}"
    body = ""
    try:
        resp = requests.get(
            api_url, headers={**UA, "Accept": "application/vnd.github+json"}, timeout=30
        )
        resp.raise_for_status()
        payload = resp.json()
        files = payload.get("files") or {}
        if not files:
            raise RuntimeError(f"gist {gist_id} has no files")
        parts: list[str] = []
        for fname, fmeta in files.items():
            content = fmeta.get("content") or ""
            if fmeta.get("truncated"):
                raw_url = fmeta.get("raw_url")
                if raw_url:
                    r2 = requests.get(raw_url, headers=UA, timeout=30)
                    if r2.ok:
                        content = r2.text
            parts.append(f"## `{fname}`\n\n```\n{content}\n```")
        body = "\n\n".join(parts)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in {502, 503, 504}:
            raw_url = f"https://gist.githubusercontent.com/karpathy/{gist_id}/raw"
            r = requests.get(raw_url, headers=UA, timeout=30)
            r.raise_for_status()
            body = f"## `(raw fallback)`\n\n```\n{r.text}\n```"
        else:
            raise

    GIST_DIR.mkdir(parents=True, exist_ok=True)
    target = GIST_DIR / f"{slug}.md"
    with target.open("w", encoding="utf-8") as fh:
        write_frontmatter(
            fh,
            {
                "title": slug,
                "url": f"https://gist.github.com/karpathy/{gist_id}",
                "author": "Andrej Karpathy",
                "gist_id": gist_id,
                "slug": slug,
                "fetched_at": datetime.now(TZ).isoformat(timespec="seconds"),
                "type": "gist",
                "note": note,
            },
        )
        fh.write(body)
        if not body.endswith("\n"):
            fh.write("\n")
    return target


def fetch_threads_aggregated(threads: list[dict]) -> Path:
    THREADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with THREADS_FILE.open("w", encoding="utf-8") as fh:
        write_frontmatter(
            fh,
            {
                "title": "Karpathy 精选 tweets / X threads",
                "author": "Andrej Karpathy",
                "fetched_at": datetime.now(TZ).isoformat(timespec="seconds"),
                "type": "curated-threads",
                "count": len(threads),
            },
        )
        fh.write("# Karpathy Curated Threads\n\n")
        fh.write(
            "> X 网站反爬严格,本文件仅登记 URL + 人工备注 + 标签。\n"
            "> 读原文请直接访问链接,或事后人工粘贴全文到本文件对应条目下方。\n\n"
        )
        for i, t in enumerate(threads, 1):
            url = t.get("url", "")
            note = t.get("note", "")
            tags = t.get("tags", [])
            fh.write(f"## {i}. {note}\n\n")
            fh.write(f"- URL: {url}\n")
            if tags:
                fh.write(f"- Tags: {', '.join(tags)}\n")
            fh.write("- Text: _(待人工填入)_\n\n")
    return THREADS_FILE


# ----- orchestration -----


def run(args: argparse.Namespace) -> int:
    if not SOURCES_YAML.exists():
        print(f"ERROR: sources YAML missing: {SOURCES_YAML}", file=sys.stderr)
        return 2

    sources = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8"))
    selected = set(args.filter) if args.filter else {"blogs", "youtube", "gists", "threads"}
    limit = args.limit

    log = RunLog()
    seen_paths = manifest_paths()

    # Plan summary
    plan = {k: len(sources.get(k, []) or []) for k in ("blogs", "youtube", "gists", "threads")}
    if args.dry_run:
        print("=== DRY RUN PLAN ===")
        for kind in ("blogs", "youtube", "gists", "threads"):
            if kind not in selected:
                continue
            items = (sources.get(kind) or [])[:limit] if limit else (sources.get(kind) or [])
            print(f"\n[{kind}] {len(items)} items:")
            for it in items:
                if kind == "blogs":
                    print(f"  - {it['slug']}  <- {it['url']}")
                elif kind == "youtube":
                    print(f"  - {it['slug']}  <- {it['video_id']}  ({it.get('title','')})")
                elif kind == "gists":
                    print(f"  - {it['slug']}  <- {it['id']}")
                elif kind == "threads":
                    print(f"  - {it.get('note','')}  <- {it['url']}")
        print(f"\nTotals: {plan}")
        return 0

    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # blogs
    if "blogs" in selected:
        for it in (sources.get("blogs") or [])[:limit] if limit else (sources.get("blogs") or []):
            slug = it["slug"]
            target_rel = f"raw/articles/karpathy/{slug}.md"
            if target_rel in seen_paths:
                print(f"[skip blog] already in manifest: {slug}")
                log.add(FetchResult("blog", slug, ART_DIR / f"{slug}.md", True, error="skipped"))
                continue
            try:
                path = fetch_blog(it["url"], slug)
                append_manifest(path, source="web", extra={"origin_url": it["url"]})
                print(f"[ok blog] {slug}")
                log.add(FetchResult("blog", slug, path, True))
            except Exception as e:
                print(f"[fail blog] {slug}: {e}", file=sys.stderr)
                log.add(FetchResult("blog", slug, ART_DIR / f"{slug}.md", False, str(e)))

    # youtube (raw only; LLM post-processing is a separate step)
    if "youtube" in selected:
        for it in (sources.get("youtube") or [])[:limit] if limit else (sources.get("youtube") or []):
            slug = it["slug"]
            target_rel = f"raw/transcripts/karpathy/{slug}.md"  # final output path (after LLM pass)
            raw_rel = f"raw/transcripts/karpathy/_raw/{slug}.md"
            if target_rel in seen_paths or raw_rel in seen_paths:
                print(f"[skip yt] already in manifest: {slug}")
                log.add(FetchResult("youtube", slug, YT_RAW_DIR / f"{slug}.md", True, error="skipped"))
                continue
            try:
                path = fetch_youtube_raw(it["video_id"], slug, it.get("title", slug))
                append_manifest(path, source="youtube", extra={"video_id": it["video_id"], "stage": "raw"})
                print(f"[ok yt-raw] {slug}")
                log.add(FetchResult("youtube", slug, path, True))
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                print(f"[fail yt] {slug}: {type(e).__name__}", file=sys.stderr)
                log.add(FetchResult("youtube", slug, YT_RAW_DIR / f"{slug}.md", False, type(e).__name__))
            except Exception as e:
                print(f"[fail yt] {slug}: {e}", file=sys.stderr)
                log.add(FetchResult("youtube", slug, YT_RAW_DIR / f"{slug}.md", False, str(e)))

    # gists
    if "gists" in selected:
        for it in (sources.get("gists") or [])[:limit] if limit else (sources.get("gists") or []):
            slug = it["slug"]
            target_rel = f"raw/gists/karpathy/{slug}.md"
            if target_rel in seen_paths:
                print(f"[skip gist] already in manifest: {slug}")
                log.add(FetchResult("gist", slug, GIST_DIR / f"{slug}.md", True, error="skipped"))
                continue
            try:
                path = fetch_gist(it["id"], slug, it.get("note", ""))
                append_manifest(path, source="github-gist", extra={"gist_id": it["id"]})
                print(f"[ok gist] {slug}")
                log.add(FetchResult("gist", slug, path, True))
            except Exception as e:
                print(f"[fail gist] {slug}: {e}", file=sys.stderr)
                log.add(FetchResult("gist", slug, GIST_DIR / f"{slug}.md", False, str(e)))

    # threads (aggregated, always overwrite)
    if "threads" in selected:
        try:
            path = fetch_threads_aggregated(sources.get("threads") or [])
            # Append manifest only if path wasn't there; if there, still append a new entry (different sha256 acceptable)
            append_manifest(path, source="curated", extra={"count": len(sources.get("threads") or [])})
            print(f"[ok threads] aggregated -> {path.name}")
            log.add(FetchResult("threads", "karpathy-threads", path, True))
        except Exception as e:
            print(f"[fail threads] {e}", file=sys.stderr)
            log.add(FetchResult("threads", "karpathy-threads", THREADS_FILE, False, str(e)))

    # write run report
    counts = log.counts()
    report = run_dir / "karpathy-report.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write(f"# Karpathy fetch — {stamp}\n\n")
        fh.write(f"- ok: {counts['ok']}\n")
        fh.write(f"- fail: {counts['fail']}\n")
        fh.write(f"- total: {counts['total']}\n\n")
        by_kind: dict[str, list[FetchResult]] = {}
        for r in log.results:
            by_kind.setdefault(r.kind, []).append(r)
        for kind, items in by_kind.items():
            fh.write(f"## {kind}\n\n")
            for r in items:
                status = "✅" if r.ok else "❌"
                tail = f" — {r.error}" if r.error else ""
                fh.write(f"- {status} `{r.slug}`{tail}\n")
            fh.write("\n")
    print(f"\n=== done ===")
    print(f"ok={counts['ok']} fail={counts['fail']} total={counts['total']}")
    print(f"report: {report.relative_to(PROJECT_ROOT)}")
    return 0 if counts["fail"] == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Andrej Karpathy public outputs.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--filter", nargs="+", choices=["blogs", "youtube", "gists", "threads"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
