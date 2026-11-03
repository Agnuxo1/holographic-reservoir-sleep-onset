# Server-IA: reproducibility audit, 3 November 2026

**Project:** Track 03 Sleep Onset (Neural Interfaces 2026).
**Author:** Francisco Angulo de Lafuente ? [ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063).
**Repository licence:** Apache-2.0; preserve all original third-party notices.
**Experiment status:** synthetic regression tests **PASS**; official public challenge score **not freshly reproduced** in this audit.

## Environment and exact dependencies

Server-IA (Linux), existing Python scientific environment with NumPy and scikit-learn.
The standalone [Holographic Reservoir](https://github.com/Agnuxo1/Holographic-Reservoir)
package must be installed to run the *reservoir ablation* in `code/test_ladder.py`.
This audit discovered that the dependency was not declared, causing
`ModuleNotFoundError: No module named 'holographic_reservoir'`.
The package `holographic-reservoir==1.0.0` was independently verified on PyPI,
installed into a separate vendor directory, and retested without modifying
the original algorithm or pre-trained weights. `requirements.txt` is updated.

## Actual test

The test was executed on the public GitHub revision
`c2cc3e16b255f4199ffba3d8a299c2a4394b239b`
with the pinned Holographic Reservoir 1.0.0 Python dependency
and a CPU-only PyTorch/NumPy scientific environment:

```console
sintetico: 3097 ventanas, 60 grabaciones
[ok] pesos de la metrica: 256.403494 == 256.403494
[ok] contrato de streaming: diferencia maxima 1.78e-14
[ok] contrato de streaming (modelo enviado): 2.84e-14
     A_constante            W-bMAE=  38.20
     B_ridge_sin_estado     W-bMAE=  37.90
     C_solo_reloj           W-bMAE=  33.93
     E_reservorio           W-bMAE=  32.02
[ok] la escalera ordena como dice la estructura de la tarea
todas las comprobaciones pasan
```

**Interpretation:** exact metric weights, the synthetic streaming contracts
and ladder ordering pass this deterministic test. These are **synthetic**
datasets (60 artificial recordings, 3,097 windows), so numbers in this
section are *not comparable* to the official W-bMAE test set.

The repository's earlier `experiments/official_check.json` documents
W-bMAE **55.8163 s** on 40 real public warm-up recordings
(5,869 windows), but this audit did not execute the complete
`code/verify_official.py` evaluation harness again. The official
runner also needs `neuralbench`/toolkit dependencies and the complete
public NEMAR data. This distinction is intentional.

## Independent GPU experiment audit (performed earlier on Server-IA)

Five previous CUDA CNN variants ran over NEMAR public Muse EEG, with
a subject-disjoint validation split and their own checkpoint receipts
at `/mnt/datos/competition-24h/eeg-emg-foundation/runs/20261103/`.
Their best subject-validation W-bMAE was **59.7952 s**, while the
selected model's separate local public warm-up W-bMAE was
**60.7921 s**, versus **60.7417 s** for the 35-second constant.
The CNN did not demonstrate an improvement over the control on that
public warm-up test. That experiment has a different scoring/evidence
chain from the 55.8163-s official kit check above; it is not claimed
to be the same evaluation.

**Next scientific gate:** reproduce the published 55.8163-s reference
against the official kit, preserve subject-disjoint partitions and
submission contracts, then test only a pre-registered improvement
hypothesis. GPU queue allocation must be authorized via JEV, bounded
to 24 h, and results must include seeds and source commit IDs.

No new Codabench submission was made by this audit, and no proprietary
or participant-level NEMAR data has been published.
