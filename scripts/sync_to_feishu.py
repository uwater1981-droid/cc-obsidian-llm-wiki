#!/usr/bin/env python3
"""
Sync Karpathy 学习资料到飞书 Docs(docx)。

工作流:
  1. 用 FEISHU_APP_ID + FEISHU_APP_SECRET 换 tenant_access_token
  2. 把每个 .md 上传到飞书 Drive(目标文件夹 token 通过 FEISHU_FOLDER_TOKEN 指定)
  3. 创建 import_task 把 .md 转成 docx
  4. 轮询 task 直到成功,记录 docx token + URL
  5. 写状态文件 scripts/feishu_sync_state.json(sha256 幂等:内容未变不重传)
  6. 生成飞书"导读"页,列出所有 docx 链接
  7. 写 run 报告到 runs/feishu/<ts>/

Env vars(必需):
  FEISHU_APP_ID        自建应用 App ID
  FEISHU_APP_SECRET    自建应用 App Secret
  FEISHU_FOLDER_TOKEN  目标飞书文件夹 token(见文件夹 URL 最后一段)

可选 env:
  FEISHU_USER_OPEN_ID  若设置则用 user_access_token(需额外 OAuth),默认走 tenant

Usage:
  python scripts/sync_to_feishu.py --dry-run
  python scripts/sync_to_feishu.py
  python scripts/sync_to_feishu.py --filter blogs
  python scripts/sync_to_feishu.py --filter landing  # 只重建导读
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "scripts" / "feishu_sync_state.json"
RUNS_DIR = PROJECT_ROOT / "runs" / "feishu"

FEISHU_API = "https://open.feishu.cn/open-apis"
TZ = timezone(timedelta(hours=8))

# --------- 同步范围(文件集合 + 显示名前缀) ---------

SYNC_TARGETS: list[tuple[str, str, str]] = [
    # (kind, glob_pattern, display_prefix)
    ("blogs", "raw/articles/karpathy/*.md", "博客"),
    ("transcripts", "raw/transcripts/karpathy/*.md", "视频"),
    ("gists", "raw/gists/karpathy/*.md", "Gist"),
    ("threads", "raw/notes/public/karpathy-threads.md", "Tweets 精选"),
    ("dashboard", "wiki/entities/andrej-karpathy.md", "Karpathy 学习仪表盘"),
]

# 导读页特殊名称,放在文件夹首
LANDING_NAME = "00 Karpathy 学习 — 导读"


@dataclass
class SyncResult:
    kind: str
    rel_path: str
    display_name: str
    ok: bool
    doc_token: str = ""
    doc_url: str = ""
    error: str = ""
    skipped: bool = False


@dataclass
class RunState:
    results: list[SyncResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "new": sum(1 for r in self.results if r.ok and not r.skipped),
            "skipped": sum(1 for r in self.results if r.skipped),
            "failed": sum(1 for r in self.results if not r.ok),
        }


# ---------- Feishu client ----------


class FeishuClient:
    def __init__(self) -> None:
        self.app_id = os.environ.get("FEISHU_APP_ID")
        self.app_secret = os.environ.get("FEISHU_APP_SECRET")
        self.folder_token = os.environ.get("FEISHU_FOLDER_TOKEN")
        if not (self.app_id and self.app_secret and self.folder_token):
            raise RuntimeError(
                "缺环境变量:FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_FOLDER_TOKEN"
            )
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expires_at - 60:
            return self._token
        r = requests.post(
            f"{FEISHU_API}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"token error: {data}")
        self._token = data["tenant_access_token"]
        self._token_expires_at = now + int(data.get("expire", 7200))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def upload_markdown(self, local_path: Path) -> str:
        """Upload .md as a regular file to drive. Returns file_token."""
        url = f"{FEISHU_API}/drive/v1/files/upload_all"
        size = local_path.stat().st_size
        with local_path.open("rb") as fh:
            files = {"file": (local_path.name, fh, "text/markdown")}
            data = {
                "file_name": local_path.name,
                "parent_type": "explorer",
                "parent_node": self.folder_token,
                "size": str(size),
            }
            r = requests.post(url, headers=self._headers(), data=data, files=files, timeout=60)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"upload failed: {resp}")
        return resp["data"]["file_token"]

    def create_import_task(
        self, file_token: str, file_name: str, target_type: str = "docx"
    ) -> str:
        """Create import task, returns ticket."""
        url = f"{FEISHU_API}/drive/v1/import_tasks"
        payload = {
            "file_extension": "md",
            "file_token": file_token,
            "type": target_type,
            "file_name": file_name,
            "point": {
                "mount_type": 1,  # 1 = explorer folder
                "mount_key": self.folder_token,
            },
        }
        r = requests.post(url, headers={**self._headers(), "Content-Type": "application/json"},
                          json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"create import task failed: {resp}")
        return resp["data"]["ticket"]

    def poll_import_task(self, ticket: str, timeout_s: int = 90) -> dict:
        """Poll until done. Returns {token, url, type}."""
        url = f"{FEISHU_API}/drive/v1/import_tasks/{ticket}"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            r = requests.get(url, headers=self._headers(), timeout=15)
            r.raise_for_status()
            resp = r.json()
            if resp.get("code") != 0:
                raise RuntimeError(f"poll failed: {resp}")
            result = resp["data"]["result"]
            job_status = result.get("job_status")
            # 0 = success; 1/2 = running; >=3 = fail
            if job_status == 0:
                return {
                    "token": result.get("token", ""),
                    "url": result.get("url", ""),
                    "type": result.get("type", ""),
                }
            if job_status is not None and job_status >= 3:
                raise RuntimeError(f"import task failed: {result}")
            time.sleep(2)
        raise RuntimeError(f"import task timeout after {timeout_s}s")

    def delete_drive_file(self, file_token: str) -> None:
        """Delete the temporary uploaded .md from drive (best-effort)."""
        url = f"{FEISHU_API}/drive/v1/files/{file_token}"
        try:
            requests.delete(
                url,
                headers=self._headers(),
                params={"type": "file"},
                timeout=15,
            )
        except Exception:
            pass


# ---------- helpers ----------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "files": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_targets(filter_kinds: set[str] | None) -> list[tuple[str, Path, str]]:
    """Return [(kind, abs_path, display_name), ...]."""
    out: list[tuple[str, Path, str]] = []
    for kind, pattern, prefix in SYNC_TARGETS:
        if filter_kinds and kind not in filter_kinds:
            continue
        matches = sorted(PROJECT_ROOT.glob(pattern))
        for p in matches:
            stem = p.stem
            display = f"{prefix} - {stem}" if kind != "threads" and kind != "dashboard" else prefix
            out.append((kind, p, display))
    return out


def display_to_filename(display: str) -> str:
    """Keep display name ascii-safe-ish for upload (Feishu tolerates unicode but avoid slashes)."""
    return display.replace("/", "_").replace("\\", "_").strip() + ".md"


def build_landing(state: dict, targets: list[tuple[str, Path, str]]) -> str:
    """Render a markdown index of all synced docs grouped by kind."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for kind, path, display in targets:
        entry = state["files"].get(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        if not entry or not entry.get("doc_url"):
            continue
        groups.setdefault(kind, []).append((display, entry["doc_url"]))

    lines: list[str] = []
    lines.append("# Karpathy 学习合集 · 飞书导读")
    lines.append("")
    lines.append("> 由 `scripts/sync_to_feishu.py` 自动生成。点任一链接进入独立飞书文档。")
    lines.append("")
    lines.append(f"- 生成时间:{datetime.now(TZ).isoformat(timespec='seconds')}")
    total = sum(len(v) for v in groups.values())
    lines.append(f"- 文档总数:{total}")
    lines.append("")

    section_order = [
        ("dashboard", "🎯 学习仪表盘(从这里开始)"),
        ("blogs", "📚 博客原文"),
        ("transcripts", "🎥 视频字幕(Whisper · 带时间戳)"),
        ("gists", "💻 代码 Gists"),
        ("threads", "💬 Tweets 精选"),
    ]
    for key, title in section_order:
        items = groups.get(key) or []
        if not items:
            continue
        lines.append(f"## {title}({len(items)})")
        lines.append("")
        for display, url in items:
            lines.append(f"- [{display}]({url})")
        lines.append("")

    lines.append("## 推荐阅读顺序")
    lines.append("")
    lines.append("1. 先打开 “学习仪表盘”(第一节)纵览全貌")
    lines.append("2. 按博客时间线读代表作(2015 RNN / 2019 训练菜谱 / 2025 Verifiability)")
    lines.append("3. 结合视频字幕 + Gists 代码动手实现 micrograd / makemore")
    lines.append("4. Tweets 作为碎片补充")
    return "\n".join(lines)


# ---------- main ----------


def run(args: argparse.Namespace) -> int:
    filter_kinds = set(args.filter) if args.filter else None
    targets = collect_targets(filter_kinds)

    print(f"[plan] {len(targets)} 个文件待同步,filter={filter_kinds or 'all'}")

    if args.dry_run:
        for kind, path, display in targets[:50]:
            rel = path.relative_to(PROJECT_ROOT)
            print(f"  [{kind}] {rel}  ->  {display}")
        if len(targets) > 50:
            print(f"  ... 共 {len(targets)} 条")
        return 0

    client = FeishuClient()
    # Early token check
    try:
        _ = client.token
        print("[auth] tenant_access_token ok")
    except Exception as e:
        print(f"[auth fail] {e}", file=sys.stderr)
        return 2

    state = load_state()
    run_log = RunState()
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sync non-landing items first
    for kind, path, display in targets:
        if kind == "landing":
            continue
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        sha = sha256_file(path)
        prev = state["files"].get(rel) or {}
        if prev.get("sha256") == sha and prev.get("doc_token"):
            print(f"[skip] {rel} (sha 未变)")
            run_log.results.append(
                SyncResult(kind, rel, display, True, prev["doc_token"], prev.get("doc_url", ""), skipped=True)
            )
            continue
        try:
            # Copy to a temp file with display-named filename (Feishu uses filename as doc title)
            tmp_dir = run_dir / "_upload"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / display_to_filename(display)
            tmp_path.write_bytes(path.read_bytes())

            file_token = client.upload_markdown(tmp_path)
            ticket = client.create_import_task(file_token, display_to_filename(display).replace(".md", ""))
            result = client.poll_import_task(ticket)
            client.delete_drive_file(file_token)

            doc_url = result.get("url") or f"https://feishu.cn/docx/{result['token']}"
            state["files"][rel] = {
                "sha256": sha,
                "doc_token": result["token"],
                "doc_url": doc_url,
                "display": display,
                "synced_at": datetime.now(TZ).isoformat(timespec="seconds"),
            }
            save_state(state)
            print(f"[ok] {rel}  ->  {doc_url}")
            run_log.results.append(SyncResult(kind, rel, display, True, result["token"], doc_url))
        except Exception as e:
            print(f"[fail] {rel}: {e}", file=sys.stderr)
            run_log.results.append(SyncResult(kind, rel, display, False, error=str(e)))

    # Build and sync landing page (always refreshed)
    if not filter_kinds or "landing" in filter_kinds or len(targets) > 0:
        landing_md = build_landing(state, collect_targets(None))  # landing always reflects all
        landing_path = run_dir / "_landing.md"
        landing_path.write_text(landing_md, encoding="utf-8")
        try:
            # delete previous landing docx if we had one? skip — Feishu import with same name
            # creates a new copy. Track the last landing token in state.
            prev_landing = state.get("landing") or {}
            file_token = client.upload_markdown(landing_path)
            ticket = client.create_import_task(file_token, LANDING_NAME)
            result = client.poll_import_task(ticket)
            client.delete_drive_file(file_token)
            doc_url = result.get("url") or f"https://feishu.cn/docx/{result['token']}"
            state["landing"] = {
                "doc_token": result["token"],
                "doc_url": doc_url,
                "synced_at": datetime.now(TZ).isoformat(timespec="seconds"),
                "previous_token": prev_landing.get("doc_token", ""),
            }
            save_state(state)
            print(f"[ok landing] {doc_url}")
            run_log.results.append(SyncResult("landing", "_landing.md", LANDING_NAME, True, result["token"], doc_url))
        except Exception as e:
            print(f"[fail landing] {e}", file=sys.stderr)
            run_log.results.append(SyncResult("landing", "_landing.md", LANDING_NAME, False, error=str(e)))

    counts = run_log.counts()
    report = run_dir / "report.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write(f"# Feishu sync — {stamp}\n\n")
        fh.write(f"- new: {counts['new']}\n- skipped: {counts['skipped']}\n- failed: {counts['failed']}\n- total: {counts['total']}\n\n")
        landing_url = (state.get("landing") or {}).get("doc_url", "")
        if landing_url:
            fh.write(f"## 导读页\n\n[{LANDING_NAME}]({landing_url})\n\n")
        fh.write("## 明细\n\n")
        for r in run_log.results:
            status = "✅" if r.ok else "❌"
            suffix = " (skipped)" if r.skipped else ""
            fh.write(f"- {status} `{r.rel_path}` -> [{r.display_name}]({r.doc_url or '-'}){suffix}\n")
            if r.error:
                fh.write(f"    - error: {r.error}\n")

    print(f"\n=== done ===")
    print(f"new={counts['new']} skipped={counts['skipped']} failed={counts['failed']}")
    print(f"报告:{report.relative_to(PROJECT_ROOT)}")
    landing = (state.get("landing") or {}).get("doc_url")
    if landing:
        print(f"\n导读页(把这个链接发给朋友):\n{landing}")
    return 0 if counts["failed"] == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync Karpathy content to Feishu docs.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--filter",
        nargs="+",
        choices=["blogs", "transcripts", "gists", "threads", "dashboard", "landing"],
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
