<!--
Project: Holographic Reservoir Sleep Onset / GPU Muse CNN Pilot
Date: 2026-11-04
Author: Francisco Angulo de Lafuente
ORCID: https://orcid.org/0009-0001-1634-7063
Provenance: original, reproducible experiment; raw outputs held on Server-IA.
Repository license: Apache-2.0. Retain original copyright and license.
-->
# Audited GPU Pilot: Four-Channel Muse CNN (3 November 2026)

**Research status:** completed local experiment; **not selected for submission**.
**Author:** Francisco Angulo de Lafuente · [ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063).
**Competition:** Neural Interfaces 2026, Track 03 — Sleep Onset.
**Reproducibility status:** local artifacts exist on Server-IA; official sealed-test performance is **unknown**.

## Research question

Can a compact, directly trained four-channel 1D CNN outperform a constant-latency baseline on participant-disjoint public Muse sleep-onset recordings? The preliminary result is **no** under this particular local pipeline; this is a valuable negative control, not evidence against other EEG architectures.

## Experimental setup and safeguards

- **Source:** public Muse sleep-onset training recordings, prepared from the locally downloaded BIDS material; no confidential sealed-evaluation labels were used.
- **Training split:** 406 recordings across 170 participants, 61,127 prepared windows.
- **Subject-disjoint validation:** 94 recordings across 33 participants, 13,979 windows.
- **Public warm-up test:** 40 recordings across 40 participants, 5,869 windows.
- **Inputs:** four EEG channels, 640-sample causal windows, 128 Hz preprocessing.
- **Training:** one NVIDIA GeForce RTX 3090, independently frozen 24-hour JEV deadline, five runs comprising weighted-loss seeds 42–44 and smooth-loss seeds 42–43.
- **Architecture:** a compact Conv1D / GroupNorm / GELU network with adaptive pooling. No inference-time tuning on the public test labels.
- **Trial selection:** lowest subject-disjoint validation weighted binned MAE; the public warm-up split was evaluated after selection.

The script's local preparation and preflight were rechecked on 4 November 2026. This is a research-only experiment; the file is not a competition submission or a claimed official leaderboard result.

## Observed numerical results

| Trial | Subject-disjoint validation W-bMAE (s) |
|:--|--:|
| Weighted loss, seed 42 | **59.7952** |
| Weighted loss, seed 43 | 61.9593 |
| Weighted loss, seed 44 | 61.8892 |
| Smooth loss, seed 42 | 98.8156 |
| Smooth loss, seed 43 | 118.6310 |

The selected checkpoint (weighted, seed 42) scored **60.7921 s** W-bMAE on the local public warm-up test, versus **60.7417 s** for the same experiment's constant-35 s reference. The CNN therefore did **not** establish an improvement under this evaluation. The observed difference is approximately +0.0504 s (lower is better), without an uncertainty claim.

**Crucial comparability note:** the repository's separately published 55.816 public-harness result uses its own verified streaming readout and official evaluation contract. Do **not** infer a head-to-head comparison with the above CNN pilot until sample definitions, metric implementation, preprocessing, and test cohorts are harmonized and independently rerun. No sealed score is claimed here.

## Evidence and provenance

The original, protected, local experiment files are stored on Server-IA:

- `/mnt/datos/competition-24h/eeg-emg-foundation/train_public_muse.py`
- `/mnt/datos/competition-24h/eeg-emg-foundation/prepared/manifest.json`
- `/mnt/datos/competition-24h/eeg-emg-foundation/runs/20261103/scientific_result.json`
- `/mnt/datos/competition-24h/eeg-emg-foundation/runs/20261103/epochs.jsonl`
- Five experiment checkpoint files in the same `runs/20261103/` directory.

Recorded official starting-kit revision: `423b5afa888dbd7eefe49ada8783af52f8b9b53d`. The private local outputs are **not** hosted by GitHub, and paths above are provenance references, not public download links. Original implementation may be MIT-licensed; original Muse datasets remain subject to their upstream **CC BY-NC-SA 4.0** terms, which are not overridden by the repository's Apache-2.0 licence.

## Next scientific decision

The JEV oracle was consulted on 4 November 2026. Its recorded decision is **stop and redesign**, rather than repeating this identical CNN training merely to occupy a GPU. Any next experiment must freeze a genuinely new, falsifiable hypothesis, document the contest's official metric and legal data constraints, request a fresh authorized GPU lease, and demonstrate an independently reproducible improvement over a matched baseline.

For priority purposes, the sealed final phase of Track 03 continues into November 2026; verify personal registration, submission quotas and deadline directly on the [official Codabench page](https://www.codabench.org/competitions/17983/) before preparing or sending any entry.

## Attribution

Copyright © 2026 Francisco Angulo de Lafuente. ORCID: [0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063). The original repository license and copyright notices must be retained. Please acknowledge this author and cite the repository when reusing its original research; this scholarly request does not impose a new condition on the applicable open-source license.
