"""iCloud 同步锁，防止多设备并发写同一 wiki 页。

用法：
    python icloud-lock.py acquire <vault> <operation>  # 加锁
    python icloud-lock.py release <vault>               # 解锁
    python icloud-lock.py status <vault>                # 查询

Lock 文件：<vault>/.wiki.lock
  包含：pid、hostname、operation、started_at
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
    try:
        started = datetime.fromisoformat(lock["started_at"])
    except (KeyError, ValueError):
        return True
    age = (datetime.now(CST) - started).total_seconds()
    return age > STALE_LOCK_SECONDS


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

    lock_data = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "operation": operation,
        "started_at": datetime.now(CST).isoformat(timespec="seconds"),
    }
    lp.write_text(json.dumps(lock_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] lock acquired for '{operation}'")
    return 0


def release(vault: Path) -> int:
    lp = lock_path(vault)
    existing = read_lock(vault)

    if existing is None:
        print("[ok] no lock to release")
        return 0

    if existing.get("pid") != os.getpid() and existing.get("hostname") == socket.gethostname():
        print(f"[warn] lock pid {existing.get('pid')} != current {os.getpid()}")

    lp.unlink()
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
        print("Usage: python icloud-lock.py {acquire|release|status} <vault> [operation]",
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
        return release(vault)
    if cmd == "status":
        return status(vault)

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
