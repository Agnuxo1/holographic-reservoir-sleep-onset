"""One-time download + window extraction of the public Muse sleep-onset study."""
import sys, os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "kit" / "tracks" / "sleep_onset"
sys.path.insert(0, str(KIT))
DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)

from benchmark_utils.nb_task import download_study, load_task  # noqa: E402

TASK = "_sleep_onset_stream"

print("descargando estudio…", flush=True)
download_study("eeg", TASK, DATA)
print("descargado; extrayendo ventanas…", flush=True)
loaders, meta = load_task(
    "eeg", TASK, data_dir=DATA, device="cpu", batch_size=64,
    seed=0, num_workers=0, subset="full",
    target_transform=lambda y: y.reshape(()),
)
print("meta:", {k: v for k, v in meta.items() if not hasattr(v, "shape")}, flush=True)
print("splits:", list(loaders), flush=True)
