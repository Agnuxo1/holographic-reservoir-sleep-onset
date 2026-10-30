"""Does the EEG *history* help at all on this task, in any form?

The reservoir sweep points one way: the shorter its memory, the better it
scores, and at the shortest memory it merely ties the stateless model. That
could be a quirk of this particular recurrent form, so before concluding
anything the same question gets asked three more ways, all on the official
validation split:

* ``ema``       - explicit exponential moving averages of the features at
  several timescales, appended to the stateless model. Interpretable memory,
  no recurrence, no tuning beyond the timescale.
* ``leak``      - the reservoir at the fast end, where the sweep says it
  lives, with the regularisation it prefers.
* ``residual``  - the reservoir state used only to correct the stateless
  model's prediction, so the history can only add and cannot damage the
  base fit.

If none of them beats the stateless model, the conclusion is about the task,
not about one architecture.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models import (ElapsedFeatRidge, EmaRidge,  # noqa: E402
                    ReservoirResidual, ReservoirStream,
                    wbmae_recording_mean)
from run_cv import load_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/memory_probe.json")
    args = parser.parse_args()

    train = load_cache("train")
    val = load_cache("val")
    order = np.lexsort((val["elapsed"], val["record_id"]))
    val = {k: v[order] for k, v in val.items()}

    def ev(model):
        model.fit(train)
        pred = model.predict_feats(val["feats"], val["elapsed"],
                                    val["record_id"])
        return wbmae_recording_mean(val["y"], pred, val["record_id"])

    rows = []

    def run(label, model):
        value = ev(model)
        rows.append(dict(label=label, wbmae=value))
        print(f"{label:46s} W-bMAE={value:7.2f}", flush=True)

    run("D_sin_memoria_alpha3", ElapsedFeatRidge(alpha=3.0))
    run("D_sin_memoria_alpha10", ElapsedFeatRidge(alpha=10.0))
    for halflives in ((2,), (6,), (2, 6, 24), (12, 48)):
        run(f"EMA_mitades={halflives}", EmaRidge(alpha=10.0,
                                                 halflives=halflives))
    for leak in (0.7, 0.85, 0.95):
        for alpha in (3.0, 10.0, 30.0):
            run(f"reservorio_leak={leak}_alpha={alpha}",
                ReservoirStream(size=100, leak=leak, alpha=alpha, seed=0))
    for size, leak, alpha in ((200, 0.5, 100.0), (200, 0.8, 30.0)):
        run(f"residual_size={size}_leak={leak}_alpha={alpha}",
            ReservoirResidual(size=size, leak=leak, alpha=alpha))

    rows.sort(key=lambda r: r["wbmae"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nordenado:")
    for row in rows:
        print(f"  {row['label']:46s} {row['wbmae']:7.2f}")


if __name__ == "__main__":
    main()
