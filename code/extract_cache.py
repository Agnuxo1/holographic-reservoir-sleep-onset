"""Turn the prepared Muse windows into a feature cache on disk.

The experiments need the same windows many times (five models, several
seeds, grouped cross-validation), and re-reading the raw recordings each
time would dominate the run. One pass over the loaders writes per-window
features, targets and identifiers to a single npz, which every later script
loads in a second.

Raw signals are never written out: only the aggregate spectral features of
each window, so nothing identifiable and no redistributable signal leaves
the download directory.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kit" / "tracks" / "sleep_onset"))
sys.path.insert(0, str(ROOT / "code"))

import kit_compat  # noqa: E402
kit_compat.apply()

from features import window_features  # noqa: E402
from benchmark_utils.nb_task import load_task  # noqa: E402

TASK = "_sleep_onset_stream"
OUT = ROOT / "cache"


def dump(loaders, split):
    loader = loaders.get(split)
    if loader is None:
        print(f"  {split}: sin loader", flush=True)
        return
    feats, ys, records, streams, onsets, subjects = [], [], [], [], [], []
    start = time.time()
    for n, (X, y, info) in enumerate(loader):
        Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        yn = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        feats.append(window_features(Xn).astype(np.float32))
        ys.append(np.asarray(yn, dtype=np.float32).reshape(-1))
        for key, sink in (("record_id", records), ("stream_id", streams),
                          ("onset", onsets), ("subject_id", subjects)):
            value = info[key]
            value = (value.detach().cpu().numpy()
                     if hasattr(value, "detach") else np.asarray(value))
            sink.append(np.asarray(value).reshape(-1))
        if n % 100 == 0:
            print(f"  {split}: {n} lotes, {time.time() - start:.0f}s",
                  flush=True)

    out = dict(
        feats=np.concatenate(feats),
        y=np.concatenate(ys),
        record_id=np.concatenate(records).astype(str),
        stream_id=np.concatenate(streams).astype(str),
        onset=np.concatenate(onsets).astype(np.float64),
        subject_id=np.concatenate(subjects).astype(str),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{split}.npz", **out)
    print(f"  {split}: {len(out['y'])} ventanas, "
          f"{len(np.unique(out['record_id']))} grabaciones, "
          f"{time.time() - start:.0f}s", flush=True)


def main():
    print("cargando tarea…", flush=True)
    loaders, meta = load_task(
        "eeg", TASK, data_dir=ROOT / "data", device="cpu", batch_size=64,
        seed=0, num_workers=0, subset="full",
        target_transform=lambda y: y.reshape(()),
    )
    printable = {k: v for k, v in meta.items() if isinstance(v, (int, float, str, list))}
    print("meta:", printable, flush=True)
    print("splits:", list(loaders), flush=True)
    for split in ("train", "valid", "val", "test"):
        if split in loaders:
            dump(loaders, split)


if __name__ == "__main__":
    main()
