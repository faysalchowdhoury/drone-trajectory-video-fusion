"""``dronefusion run`` | ``dronefusion serve``."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_RAW = "../cloud-trajectory-etl-geh-pipeline/data/raw/i80_1600"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dronefusion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="render NGSIM -> detect -> track -> georeference -> evaluate -> fuse")
    r.add_argument("--raw", default=DEFAULT_RAW, help="NGSIM CSV pages (download with `trajetl download`)")
    r.add_argument("--seconds", type=float, default=300.0)
    r.add_argument("--out", default="results/i80_drone")
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import RunConfig, run
        from .report import write

        res = run(RunConfig(raw_dir=Path(a.raw), seconds=a.seconds, seed=a.seed))
        print(write(res, Path(a.out)).read_text(encoding="utf-8"))
    else:
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
