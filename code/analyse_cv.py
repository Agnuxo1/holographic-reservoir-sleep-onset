"""Summarise the cross-validation, and compare models the paired way.

Fold-to-fold variation on this task is several seconds of W-bMAE, far larger
than the differences between the models near the top. Comparing two models
by their mean over folds therefore says very little. What matters is the
difference *within* each fold, where the held-out recordings are the same
for both: that is what ``paired`` reports, as mean, standard deviation and
the number of folds in which each model won.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

PUBLISHED = (("Constante 35 s", 60.7), ("EEGNet", 60.6), ("CTNet", 60.1),
             ("REVE afinado", 72.9))


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def by_model(rows, regime):
    """Mean over seeds per (model, fold), then the fold values per model."""
    per = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["regime"] != regime:
            continue
        per[row["model"]][row["fold"]].append(row["wbmae"])
    return {model: {fold: float(np.mean(values))
                    for fold, values in folds.items()}
            for model, folds in per.items()}


def table(rows, regime):
    folds_by_model = by_model(rows, regime)
    print(f"\n== {regime} ==")
    print(f"{'modelo':28s} {'media':>7s} {'desv':>6s} {'pliegues':>9s}")
    order = sorted(folds_by_model,
                   key=lambda m: np.mean(list(folds_by_model[m].values())))
    for model in order:
        values = np.array(list(folds_by_model[model].values()))
        print(f"{model:28s} {values.mean():7.2f} {values.std():6.2f} "
              f"{len(values):9d}")
    return folds_by_model


def paired(folds_by_model, a, b):
    """Within-fold difference a - b (negative means a is better)."""
    if a not in folds_by_model or b not in folds_by_model:
        return None
    shared = sorted(set(folds_by_model[a]) & set(folds_by_model[b]))
    diffs = np.array([folds_by_model[a][f] - folds_by_model[b][f]
                      for f in shared])
    return dict(a=a, b=b, mean=float(diffs.mean()), std=float(diffs.std()),
                n_folds=len(diffs), a_wins=int((diffs < 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/cv_results.json")
    parser.add_argument("--out", default="experiments/summary.json")
    args = parser.parse_args()
    rows = load(args.results)
    regimes = sorted({r["regime"] for r in rows})

    print("referencias publicadas por los organizadores:")
    for name, value in PUBLISHED:
        print(f"  {name:28s} {value:7.2f}")

    summary = {}
    for regime in regimes:
        folds_by_model = table(rows, regime)
        comparisons = []
        for a, b in (("E_reservorio_holografico", "D_reloj_mas_ventana"),
                     ("E_reservorio_holografico", "F_reservorio_esn"),
                     ("I_memoria_explicita_10s", "D_reloj_mas_ventana"),
                     ("I_memoria_explicita_10s", "E_reservorio_holografico"),
                     ("J_reservorio_residual", "D_reloj_mas_ventana"),
                     ("H_reservorio_solo_estado", "E_reservorio_holografico"),
                     ("D_reloj_mas_ventana", "B_ridge_sin_estado"),
                     ("C_solo_reloj", "A_constante"),
                     ("B_ridge_sin_estado", "A_constante")):
            comparison = paired(folds_by_model, a, b)
            if comparison:
                comparisons.append(comparison)
        print(f"\ncomparaciones emparejadas por pliegue ({regime}):")
        for c in comparisons:
            verdict = ("gana " + c["a"].split("_")[0] if c["mean"] < 0
                       else "gana " + c["b"].split("_")[0])
            print(f"  {c['a']:26s} - {c['b']:26s} "
                  f"{c['mean']:+7.2f} +/- {c['std']:5.2f}  "
                  f"({c['a_wins']}/{c['n_folds']} pliegues, {verdict})")
        summary[regime] = dict(
            per_model={m: dict(mean=float(np.mean(list(f.values()))),
                               std=float(np.std(list(f.values()))),
                               n_folds=len(f))
                       for m, f in folds_by_model.items()},
            paired=comparisons)

    out = ROOT / args.out
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nescrito", out)


if __name__ == "__main__":
    main()
