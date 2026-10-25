#!/usr/bin/env python3
"""Разность часов ноутбука и стенда по NTP на обеих сторонах (для guard.edge_probe --clock-offset-ms).

    python tools/stand_clock.py [--url http://127.0.0.1:8888] [--env .env]

На стенде NTP запрашивается в ядре Jupyter (guard.sysinfo.ntp_best из ~/work/mosprom, только stdlib),
на ноутбуке — локально. Печатает JSON: laptop_minus_stand_ms (часы ноутбука − часы стенда) и
uncertainty_ms (сумма погрешностей двух замеров). Edge topology v2 свой NTP-сдвиг не публикует.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
NTP_SERVERS = ("time.google.com", "pool.ntp.org", "time.cloudflare.com")
REMOTE = (
    "import json, os, sys\n"
    "sys.path.insert(0, os.path.expanduser('~/work/mosprom'))\n"
    "from guard.sysinfo import ntp_best\n"
    f"print(json.dumps(ntp_best({NTP_SERVERS!r}, samples=6)))\n"
)


def laptop_minus_stand(stand: dict[str, Any] | None, laptop: dict[str, Any] | None) -> dict[str, float]:
    """offset_ms — поправка к своим часам до NTP, поэтому ноутбук − стенд = offset_стенд − offset_ноутбук."""
    if not stand or "offset_ms" not in stand or not laptop or "offset_ms" not in laptop:
        raise ValueError("NTP недоступен на одной из сторон")
    return {
        "laptop_minus_stand_ms": round(stand["offset_ms"] - laptop["offset_ms"], 1),
        "uncertainty_ms": round(stand.get("uncertainty_ms", 0.0) + laptop.get("uncertainty_ms", 0.0), 1),
    }


def measure(client: Any, local_ntp: Callable[[], dict[str, Any] | None] | None = None) -> dict[str, Any]:
    if local_ntp is None:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from guard.sysinfo import ntp_best

        def local_ntp():
            return ntp_best(NTP_SERVERS, samples=6)

    result = client.execute(REMOTE, timeout=120)
    if not result.ok:
        raise ValueError("NTP на стенде не выполнился")
    stand = json.loads(result.stdout.strip().splitlines()[-1])
    laptop = local_ntp()
    return {**laptop_minus_stand(stand, laptop), "stand_ntp": stand, "laptop_ntp": laptop}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8888/")
    parser.add_argument("--env", default=str(REPO / ".env"))
    args = parser.parse_args(argv)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import jhub

    client = jhub.JHub.from_env(args.env, url=args.url, verbose=False)
    try:
        print(json.dumps(measure(client), ensure_ascii=False))
    except ValueError as error:
        print(f"stand_clock: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
