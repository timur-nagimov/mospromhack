#!/usr/bin/env python3
"""Залить код проекта (guard/, tpu/, tools/jhub.py) на Jupyter-стенд в ~/work/guard_repo/.

    python tpu/sync_to_jupyter.py [--dst guard_repo] [--extra path ...]

Загружаются только *.py/*.sh/*.json/*.yaml из git-дерева; файлы с тем же размером и mtime не старше
предыдущей заливки пропускаются (кеш ~/.cache/zazerkalye/sync_<dst>.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.jhub import JHub  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="guard_repo")
    ap.add_argument("--paths", nargs="*", default=["guard", "tpu", "tools/jhub.py", "configs"])
    ap.add_argument("--extra", nargs="*", default=[], help="доп. файлы (абсолютные пути) -> <dst>/extra/")
    a = ap.parse_args()
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", *a.paths],
                           cwd=REPO, capture_output=True, text=True).stdout.split()
    files = [f for f in files if f.endswith((".py", ".sh", ".json", ".yaml", ".yml", ".txt"))]
    cache_p = Path.home() / ".cache" / "zazerkalye" / f"sync_{a.dst}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    jh = JHub.from_env()
    jh.ensure()
    n = 0
    for f in files:
        p = REPO / f
        key = f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
        if cache.get(f) == key:
            continue
        jh.upload(p, f"{a.dst}/{f}")
        cache[f] = key
        n += 1
    for e in a.extra:
        jh.upload_parallel(e, f"{a.dst}/extra/{Path(e).name}", workers=4, part_mb=4)
        n += 1
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(cache))
    print(f"залито файлов: {n} (из {len(files)}) -> ~/work/{a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
