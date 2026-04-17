"""iCloud 同步锁，防止多设备并发写同一 wiki 页。

用法：
    python icloud-lock.py acquire <vault> <operation>  # 加锁
    python icloud-lock.py release <vault> [--force]    # 解锁
    python icloud-lock.py status <vault>               # 查询

Lock 文件：<vault>/.wiki.lock
  包含：pid、hostname、operation、started_at

安全性：
- acquire 用 `open('x')` 独占创建（OS 级原子性）
- release 强制校验 PID + hostname，不匹配需 --force
- iCloud 是最终一致性同步，跨设备的锁仍是 best-effort
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


CST = timezone(timedelta(hours=8))
LOCK_FILENAME = ".wiki.lock"
STALE_LOCK_SECONDS = 3600  # 1 小时视为过期


def lock_path(vault: Path) -> Path:
    return vault / LOCK_FILENAME


def read_lock(vault: Path) -> dict | None:
    lp = lock_path(vault)
    if not lp.exists():
        return None
    try:
        return json.loads(lp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_stale(lock: dict) -> bool:
    """时区容错：fromisoformat 可能返回 naive datetime。"""
    try:
        started = datetime.fromisoformat(lock["started_at"])
    except (KeyError, ValueError):
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=CST)
    age = (datetime.now(CST) - started).total_seconds()
    return age > STALE_LOCK_SECONDS


def _write_lock_atomic(lp: Path, lock_data: dict) -> bool:
    """使用 open('x') 独占创建，返回是否成功。"""
    try:
        with lp.open("x", encoding="utf-8") as f:
            json.dump(lock_data, f, ensure_ascii=False, indent=2)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        print(f"[error] cannot write lock file: {e}", file=sys.stderr)
        return False


def acquire(vault: Path, operation: str) -> int:
    lp = lock_path(vault)
    existing = read_lock(vault)

    if existing and not is_stale(existing):
        print(f"[error] lock held by {existing.get('hostname')}:{existing.get('pid')}",
              file=sys.stderr)
        print(f"  operation: {existing.get('operation')}", file=sys.stderr)
        print(f"  started_at: {existing.get('started_at')}", file=sys.stderr)
        return 1

    if existing and is_stale(existing):
        print(f"[warn] removing stale lock from {existing.get('hostname')}", file=sys.stderr)
        try:
            lp.unlink()
        except OSError as e:
            print(f"[error] could not remove stale lock: {e}", file=sys.stderr)
            return 1

    lock_data = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "operation": operation,
        "started_at": datetime.now(CST).isoformat(timespec="seconds"),
    }

    if not _write_lock_atomic(lp, lock_data):
        print("[error] lost race to another process, lock now held elsewhere", file=sys.stderr)
        return 1

    print(f"[ok] lock acquired for '{operation}'")
    return 0


def release(vault: Path, force: bool = False) -> int:
    lp = lock_path(vault)
    existing = read_lock(vault)

    if existing is None:
        print("[ok] no lock to release")
        return 0

    same_host = existing.get("hostname") == socket.gethostname()
    same_pid = existing.get("pid") == os.getpid()

    if not (same_host and same_pid) and not force:
        print(f"[error] lock owned by {existing.get('hostname')}:{existing.get('pid')}, "
              f"refusing release (use --force to override)", file=sys.stderr)
        return 1

    try:
        lp.unlink()
    except OSError as e:
        print(f"[error] could not remove lock file: {e}", file=sys.stderr)
        return 1

    print("[ok] lock released")
    return 0


def status(vault: Path) -> int:
    existing = read_lock(vault)
    if existing is None:
        print("[ok] no lock")
        return 0
    stale_mark = " (STALE)" if is_stale(existing) else ""
    print(f"[lock]{stale_mark}")
    print(json.dumps(existing, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python icloud-lock.py {acquire|release|status} <vault> [operation|--force]",
              file=sys.stderr)
        return 2

    cmd = sys.argv[1]
    vault = Path(sys.argv[2]).resolve()

    if not vault.exists():
        print(f"[error] vault not found: {vault}", file=sys.stderr)
        return 1

    if cmd == "acquire":
        if len(sys.argv) < 4:
            print("acquire requires <operation> arg", file=sys.stderr)
            return 2
        return acquire(vault, sys.argv[3])
    if cmd == "release":
        force = "--force" in sys.argv
        return release(vault, force=force)
    if cmd == "status":
        return status(vault)

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
