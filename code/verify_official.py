"""Score a built submission with the official harness, not with my own code.

Loads ``submission.py`` and ``weights.npz`` exactly as the platform would,
then runs the kit's own ``predict_streams`` (deep copy of the model per
recording, ``reset_state()`` first, one window per call, time order) and the
kit's own ``binned_mae`` / ``group_scores`` over the public test split - the
40 recordings that are the Codabench warm-up evaluation set.

This is the check that matters before submitting: it exercises the real
contract, with the real loader, through the real metric. Anything my own
cross-validation gets right but this gets wrong is a packaging bug.

Usage::

    python code/verify_official.py --submission submissions/holografico
"""

import argparse
import functools
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kit" / "tracks" / "sleep_onset"))
sys.path.insert(0, str(ROOT / "code"))

import kit_compat  # noqa: E402
kit_compat.apply()

from benchmark_utils.metrics import binned_mae, group_scores  # noqa: E402
from benchmark_utils.nb_task import load_task  # noqa: E402
from benchmark_utils.streaming import predict_streams  # noqa: E402

TASK = "_sleep_onset_stream"
BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)


def load_submission_model(submission_dir, meta):
    """Import the submission module and build its model, as the platform does."""
    path = Path(submission_dir) / "submission.py"
    spec = importlib.util.spec_from_file_location("participant_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["participant_submission"] = module
    spec.loader.exec_module(module)
    solver = module.Solver.__new__(module.Solver)
    return solver.load_model({**meta, "submission_dir": Path(submission_dir)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submissions/holografico")
    parser.add_argument("--out", default="experiments/official_check.json")
    args = parser.parse_args()

    loaders, meta = load_task(
        "eeg", TASK, data_dir=ROOT / "data", device="cpu", batch_size=64,
        seed=0, num_workers=0, subset="full",
        target_transform=lambda y: y.reshape(()),
    )
    meta = {**meta, "n_outputs": 1, "device": "cpu"}
    model = load_submission_model(ROOT / args.submission, meta)

    t0 = time.time()
    y_true, y_pred, info = predict_streams(model, loaders["test"])
    seconds = time.time() - t0
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    wbmae = functools.partial(binned_mae, bin_edges=BIN_EDGES,
                              bin_weights=BIN_WEIGHTS)
    per_recording = group_scores(wbmae, y_true, y_pred, info["stream_id"])
    result = dict(
        submission=str(args.submission),
        wbmae_recording_mean=float(per_recording.mean()),
        wbmae_recording_std=float(per_recording.std()),
        bmae=float(binned_mae(y_true, y_pred, BIN_EDGES)),
        mae=float(np.abs(y_pred - y_true).mean()),
        n_recordings=int(len(per_recording)),
        n_windows=int(len(y_true)),
        inference_seconds=round(seconds, 1),
    )
    print(json.dumps(result, indent=1))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
