"""Hyperparameter sweep on the official validation split.

The release's own split holds out 20% of the participants who have no test
recording; those are ``cache/val.npz``. Choosing on them keeps the 40 test
recordings - the Codabench warm-up evaluation set - untouched, so the number
reported there afterwards is not a number anything was selected on.

Usage::

    python code/sweep.py --what reservorio
    python code/sweep.py --what ridge
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models import (ElapsedFeatRidge, ReservoirStream,  # noqa: E402
                    StatelessRidge, wbmae_recording_mean)
from run_cv import load_cache  # noqa: E402


def score(model, train, test):
    model.fit(train)
    pred = model.predict_feats(test["feats"], test["elapsed"],
                               test["record_id"])
    return wbmae_recording_mean(test["y"], pred, test["record_id"])


def ordered(test):
    order = np.lexsort((test["elapsed"], test["record_id"]))
    return {k: v[order] for k, v in test.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", default="reservorio",
                        choices=["reservorio", "ridge"])
    parser.add_argument("--out", default="experiments/sweep.json")
    args = parser.parse_args()

    train = load_cache("train")
    test = ordered(load_cache("val"))
    print(f"train {len(train['y'])} ventanas / "
          f"{len(np.unique(train['record_id']))} grabaciones; "
          f"val {len(test['y'])} / "
          f"{len(np.unique(test['record_id']))}", flush=True)

    rows = []
    if args.what == "ridge":
        for alpha in (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
            for name, factory in (
                    ("B_ridge_sin_estado",
                     lambda a=alpha: StatelessRidge(alpha=a)),
                    ("D_reloj_mas_ventana",
                     lambda a=alpha: ElapsedFeatRidge(alpha=a))):
                t0 = time.time()
                value = score(factory(), train, test)
                rows.append(dict(model=name, alpha=alpha, wbmae=value,
                                 seconds=round(time.time() - t0, 1)))
                print(f"{name} alpha={alpha:<7g} W-bMAE={value:7.2f}",
                      flush=True)
    else:
        grid = itertools.product((100, 300), (0.1, 0.3, 0.7),
                                 (10.0, 100.0, 1000.0))
        for size, leak, alpha in grid:
            t0 = time.time()
            model = ReservoirStream(size=size, leak=leak, alpha=alpha, seed=0)
            value = score(model, train, test)
            rows.append(dict(model="E_reservorio_holografico", size=size,
                             leak=leak, alpha=alpha, wbmae=value,
                             seconds=round(time.time() - t0, 1)))
            print(f"size={size:<4d} leak={leak:<4g} alpha={alpha:<7g} "
                  f"W-bMAE={value:7.2f}  ({rows[-1]['seconds']}s)", flush=True)

    rows.sort(key=lambda r: r["wbmae"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = (json.loads(out.read_text(encoding="utf-8"))
                if out.exists() else [])
    out.write_text(json.dumps(existing + rows, indent=1), encoding="utf-8")
    print("\nmejores:")
    for row in rows[:5]:
        print(" ", json.dumps(row))


if __name__ == "__main__":
    main()
