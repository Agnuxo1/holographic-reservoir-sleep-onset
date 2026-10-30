"""Cross-validated comparison of the model ladder on the public Muse data.

Two validation regimes, because the sealed phase scores a macro-average over
participants seen and unseen in training:

* ``unseen`` - grouped k-fold by participant: no recording of a held-out
  participant is in the training set. This is the leave-subject-out regime.
* ``seen``   - held-out recordings of participants who do have other
  recordings in training.

Every model is fitted on the training part of the fold only, including the
elapsed-time table and the feature standardisation, so nothing about the
held-out recordings leaks. Predictions are produced recording by recording
in time order, which is what the platform does.

Usage::

    python code/run_cv.py --folds 5 --seeds 0 1 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models import (ConstantModel, ElapsedFeatRidge, ElapsedQuantile,  # noqa
                    EmaRidge, ReservoirResidual, ReservoirStream,
                    StatelessRidge, wbmae_recording_mean)

WINDOW_S = 5.0


def load_cache(split="train"):
    data = np.load(ROOT / "cache" / f"{split}.npz", allow_pickle=False)
    out = {k: data[k] for k in data.files}
    out["record_id"] = out["record_id"].astype(str)
    # Elapsed time inside the recording: the window's own start. A streaming
    # model recovers it by counting its calls; here it comes from the loader
    # so the offline fit sees exactly what the streamed model will see.
    out["elapsed"] = out["onset"].astype(float)
    # The loader gives the participant alongside the recording, so the
    # held-out folds are genuinely by participant, not by recording: the
    # same sleeper's other nights never sit in the training set of a fold
    # that scores them.
    out["subject"] = out["subject_id"].astype(str)
    return out


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()
            if isinstance(v, np.ndarray) and len(v) == len(data["y"])}


def folds_unseen(data, n_folds, seed=0):
    """Grouped k-fold over participants."""
    subjects = np.unique(data["subject"])
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    for f in range(n_folds):
        held = set(subjects[f::n_folds])
        mask = np.array([s in held for s in data["subject"]])
        yield mask


def folds_seen(data, n_folds, seed=0):
    """Held-out recordings of participants kept in training."""
    rng = np.random.default_rng(seed)
    records = np.unique(data["record_id"])
    subject_by_record = dict(zip(data["record_id"], data["subject"]))
    by_subject = {}
    for r in records:
        by_subject.setdefault(subject_by_record[r], []).append(r)
    multi = {s: rs for s, rs in by_subject.items() if len(rs) >= 2}
    pool = []
    for s, rs in multi.items():
        rs = sorted(rs)
        pool.append(rs[rng.integers(len(rs))])
    pool = np.array(pool)
    rng.shuffle(pool)
    for f in range(n_folds):
        held = set(pool[f::n_folds])
        mask = np.array([r in held for r in data["record_id"]])
        yield mask


def build_models(seed):
    """The ladder. ``needs_seed`` marks the ones repeated over seeds."""
    return [
        ("A_constante", lambda: ConstantModel(), False),
        ("B_ridge_sin_estado", lambda: StatelessRidge(alpha=30.0), False),
        ("C_solo_reloj", lambda: ElapsedQuantile(), False),
        ("D_reloj_mas_ventana", lambda: ElapsedFeatRidge(alpha=3.0), False),
        # Reservoir hyperparameters chosen on the official validation split
        # (code/sweep.py): size 100, leak 0.7, alpha 30 - its fast end, which
        # is where the sweep says it scores best.
        ("E_reservorio_holografico",
         lambda: ReservoirStream(size=100, leak=0.7, alpha=30.0, seed=seed,
                                 holographic=True), True),
        ("F_reservorio_esn",
         lambda: ReservoirStream(size=100, leak=0.7, alpha=30.0, seed=seed,
                                 holographic=False), True),
        ("G_reservorio_sin_reloj",
         lambda: ReservoirStream(size=100, leak=0.7, alpha=30.0, seed=seed,
                                 holographic=True, use_prior=False), True),
        ("H_reservorio_solo_estado",
         lambda: ReservoirStream(size=100, leak=0.7, alpha=30.0, seed=seed,
                                 holographic=True, use_feats=False), True),
        # Explicit short memory instead of a recurrent state: one moving
        # average with a half-life of two windows (10 s).
        ("I_memoria_explicita_10s",
         lambda: EmaRidge(alpha=10.0, halflives=(2,)), False),
        ("J_reservorio_residual",
         lambda: ReservoirResidual(size=200, leak=0.8, alpha=30.0,
                                   seed=seed), True),
    ]


def evaluate(model, test):
    """Predict the held-out windows recording by recording, in time order."""
    order = np.lexsort((test["elapsed"], test["record_id"]))
    feats = test["feats"][order]
    elapsed = test["elapsed"][order]
    records = test["record_id"][order]
    pred = model.predict_feats(feats, elapsed, records)
    return wbmae_recording_mean(test["y"][order], pred, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--regimes", nargs="+", default=["unseen", "seen"])
    parser.add_argument("--out", default="experiments/cv_results.json")
    parser.add_argument("--limit-records", type=int, default=0,
                        help="usar solo N grabaciones (prueba rapida)")
    args = parser.parse_args()

    data = load_cache("train")
    if args.limit_records:
        keep = set(np.unique(data["record_id"])[:args.limit_records])
        data = subset(data, np.array([r in keep for r in data["record_id"]]))
    print(f"{len(data['y'])} ventanas, "
          f"{len(np.unique(data['record_id']))} grabaciones, "
          f"{len(np.unique(data['subject']))} participantes", flush=True)

    results = []
    for regime in args.regimes:
        splitter = folds_unseen if regime == "unseen" else folds_seen
        for fold, mask in enumerate(splitter(data, args.folds)):
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            train, test = subset(data, ~mask), subset(data, mask)
            for name, _factory, needs_seed in build_models(0):
                seeds = args.seeds if needs_seed else [0]
                for seed in seeds:
                    factory = dict((m[0], m[1])
                                   for m in build_models(seed))[name]
                    model = factory()
                    t0 = time.time()
                    model.fit(train)
                    score = evaluate(model, test)
                    row = dict(regime=regime, fold=fold, model=name, seed=seed,
                               wbmae=score,
                               n_test_records=int(len(np.unique(
                                   test["record_id"]))),
                               seconds=round(time.time() - t0, 1))
                    results.append(row)
                    print(f"{regime} f{fold} {name} s{seed}: "
                          f"W-bMAE={score:.2f} ({row['seconds']}s)", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n=== resumen (media +/- desv. tipica entre pliegues y semillas) ===")
    for regime in args.regimes:
        print(f"-- {regime} --")
        for name, _f, _s in build_models(0):
            vals = [r["wbmae"] for r in results
                    if r["model"] == name and r["regime"] == regime]
            if vals:
                print(f"  {name:28s} {np.mean(vals):7.2f} +/- "
                      f"{np.std(vals):5.2f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
