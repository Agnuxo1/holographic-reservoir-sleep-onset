"""Checks on the model ladder that do not need the real recordings.

Synthetic recordings with the same structure as the task: a duration drawn
per recording, non-overlapping 5 s windows from the start up to onset, a
target capped at 600 s, and features that are mostly noise plus a weak
drowsiness ramp as onset approaches. Enough to catch the bugs that matter:

1. ``metric_weights`` is consistent with the ranking metric - the weighted
   mean absolute error over all windows must equal W-bMAE averaged over
   recordings.
2. The streaming contract matches the offline fit: calling ``predict`` one
   window at a time, with ``reset_state()`` per recording, must give the
   same numbers as ``predict_feats`` over the whole array. If these drift,
   the submission scores differently from the cross-validation.
3. The ladder is ordered the way the task's structure says it must be: a
   model that knows the clock beats one that does not.

Run: ``python code/test_ladder.py``
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (CAP_S, ConstantModel, ElapsedQuantile,  # noqa
                    EmaRidge, ReservoirStream, StatelessRidge,
                    elapsed_features, metric_weights,
                    wbmae_recording_mean)

WINDOW_S = 5.0
CAP_S = 600.0
N_FEATS = 12


def synth(n_records=60, seed=0):
    """Recordings whose features carry a weak signal about time-to-onset."""
    rng = np.random.default_rng(seed)
    feats, ys, elapsed, records = [], [], [], []
    for r in range(n_records):
        duration = float(np.clip(rng.lognormal(5.6, 0.7), 60.0, 1500.0))
        n = max(4, int(duration // WINDOW_S))
        for k in range(n):
            e = k * WINDOW_S
            y = min(CAP_S, duration - e)
            # Drowsiness proxy: grows as onset nears, buried in noise.
            closeness = np.exp(-y / 120.0)
            x = rng.normal(0.0, 1.0, N_FEATS)
            x[0] += 1.4 * closeness
            x[1] -= 1.1 * closeness
            feats.append(x)
            ys.append(y)
            elapsed.append(e)
            records.append(f"sub-{r:03d}_rec")
    return dict(feats=np.array(feats), y=np.array(ys, float),
                elapsed=np.array(elapsed, float),
                record_id=np.array(records))


def check_metric_weights(data):
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.0, CAP_S, len(data["y"]))
    weights = metric_weights(data["y"], data["record_id"])
    weighted = float(np.sum(weights * np.abs(pred - data["y"])))
    direct = wbmae_recording_mean(data["y"], pred, data["record_id"])
    assert abs(weighted - direct) < 1e-9, (weighted, direct)
    print(f"[ok] pesos de la metrica: {weighted:.6f} == {direct:.6f}")


def check_streaming_contract(train, test):
    """Offline prediction and one-window-at-a-time must agree."""
    import copy

    model = ReservoirStream(size=80, seed=0).fit(train)
    order = np.lexsort((test["elapsed"], test["record_id"]))
    feats = test["feats"][order]
    elapsed = test["elapsed"][order]
    records = test["record_id"][order]
    offline = model.predict_feats(feats, elapsed, records)

    # Mimic benchmark_utils.streaming: a fresh deep copy per recording whose
    # reset_state() runs first, then one window per predict call.
    streamed = np.empty(len(feats))
    current = None
    previous = None
    for i in range(len(feats)):
        if records[i] != previous:
            current = copy.deepcopy(model)
            current.reset_state()
            previous = records[i]
        # predict() recomputes features from raw windows; here the features
        # are the input, so go through the same readout path by hand.
        u = (feats[i] - current.mean_in) / current.scale_in
        current.state = ((1.0 - current.leak) * current.state
                         + current.leak * np.tanh(
                             current.W_in @ u + current.W @ current.state))
        Z = current._inputs(current.state[None], [current.k * WINDOW_S],
                            feats[i:i + 1])
        streamed[i] = current.core.predict(Z)[0]
        current.k += 1

    gap = float(np.max(np.abs(streamed - offline)))
    assert gap < 1e-9, f"streaming y offline difieren en {gap}"
    print(f"[ok] contrato de streaming: diferencia maxima {gap:.2e}")


def check_ema_streaming(train, test):
    """The submitted model: window by window must match the offline fit.

    EmaRidge is what actually ships, so its streamed path is the one that has
    to agree with the predictions the cross-validation scored.
    """
    import copy

    model = EmaRidge(alpha=10.0, halflives=(2,)).fit(train)
    order = np.lexsort((test["elapsed"], test["record_id"]))
    feats = test["feats"][order]
    elapsed = test["elapsed"][order]
    records = test["record_id"][order]
    offline = model.predict_feats(feats, elapsed, records)

    streamed = np.empty(len(feats))
    current = None
    previous = None
    for i in range(len(feats)):
        if records[i] != previous:
            current = copy.deepcopy(model)
            current.reset_state()
            previous = records[i]
        row = feats[i]
        if current.ema is None:
            current.ema = [row.copy() for _ in current.halflives]
        for j, halflife in enumerate(current.halflives):
            decay = 0.5 ** (1.0 / halflife)
            current.ema[j] = decay * current.ema[j] + (1.0 - decay) * row
        parts = [row[None], np.concatenate(current.ema)[None],
                 elapsed_features([current.k * WINDOW_S])]
        if current.prior is not None:
            parts.append(current.prior.lookup(
                [current.k * WINDOW_S]).reshape(-1, 1) / CAP_S)
        streamed[i] = current.core.predict(
            np.concatenate(parts, axis=1))[0]
        current.k += 1

    gap = float(np.max(np.abs(streamed - offline)))
    assert gap < 1e-9, f"EmaRidge: streaming y offline difieren en {gap}"
    print(f"[ok] contrato de streaming (modelo enviado): {gap:.2e}")


def check_ladder(train, test):
    order = np.lexsort((test["elapsed"], test["record_id"]))
    test = {k: v[order] for k, v in test.items()}
    scores = {}
    for name, model in (
            ("A_constante", ConstantModel()),
            ("B_ridge_sin_estado", StatelessRidge(alpha=10.0)),
            ("C_solo_reloj", ElapsedQuantile()),
            ("E_reservorio", ReservoirStream(size=120, seed=0))):
        model.fit(train)
        pred = model.predict_feats(test["feats"], test["elapsed"],
                                   test["record_id"])
        scores[name] = wbmae_recording_mean(test["y"], pred,
                                            test["record_id"])
        print(f"     {name:22s} W-bMAE={scores[name]:7.2f}")
    assert scores["C_solo_reloj"] < scores["A_constante"], scores
    assert scores["E_reservorio"] < scores["A_constante"], scores
    print("[ok] la escalera ordena como dice la estructura de la tarea")


def main():
    data = synth()
    subjects = np.unique(data["record_id"])
    held = set(subjects[::4])
    mask = np.array([r in held for r in data["record_id"]])
    train = {k: v[~mask] for k, v in data.items()}
    test = {k: v[mask] for k, v in data.items()}
    print(f"sintetico: {len(data['y'])} ventanas, {len(subjects)} grabaciones")
    check_metric_weights(data)
    check_streaming_contract(train, test)
    check_ema_streaming(train, test)
    check_ladder(train, test)
    print("\ntodas las comprobaciones pasan")


if __name__ == "__main__":
    main()
