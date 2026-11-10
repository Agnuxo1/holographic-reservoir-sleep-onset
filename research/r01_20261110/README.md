# R01: chronological training repair and verified sealed entry

On 10 November 2026, Codabench completed submission **1046573** in Track 03,
Sealed Final Phase: **W-bMAE macro 49.82 seconds**, with a displayed position
of **38 out of 45** at observation. This is a current result, not a final
ranking or prize. Evaluation duration was 90.31 seconds. The exact submitted
17,483-byte ZIP has SHA-256
`e3dbd05f639f722b87bbee1a4b9c7fd06533386206cba308270f886d82f68cf1`.
See [official results](https://www.codabench.org/competitions/17983/#/results-tab)
and [the captured aggregate receipt](OFFICIAL_RESULT.json).

## What was repaired

The historical training-cache extractor iterated NeuralBench's regression
sampler. Its 61,303 rows contained only 30,086 distinct recording/time pairs,
with replacement draws in shuffled order. The original EMA fit therefore
did not see the chronological context used at inference. Sorting and
deduplicating could not recover the missing windows.

We recovered every training-dataset index exactly once: 61,303 unique windows,
412 recordings and170 participants, each stream starting at zero and advancing
in five-second steps. All old sampled rows match their recovered features
and targets exactly. Original files are preserved; see the
[metadata audit](CACHE_METADATA_AUDIT.json) and
[sequential reconstruction receipt](SEQUENTIAL_RECONSTRUCTION.json).

## Fixed prospective comparison

Before outcomes, the protocol fixed five unseen-participant folds and five
seen-participant recording folds, a shared split seed, three equal-weighted
neural seeds and explicit gates. All models are fitted using foldtrain only
and evaluated on identical complete held-out streams: 412 unseen and 91 seen
recordings. Every prior and scaler is fitted inside its training fold.

| Candidate | Unseen(s) | Seen(s) | Equal-regime macro(s) | Fixed decision |
|---|---:|---:|---:|---|
| H: historical sampled fit |51.033127|50.824883|50.929005|comparator|
| L: chronological IRLS |49.132254|48.700529|48.916391|selected; all gates pass|
| N: bounded nonlinear residual |51.062118|46.425222|48.743670|rejected: unseen regression and CI|

L improves the macro by 2.012614 seconds, with a paired participant-cluster
97.5% interval [1.172111, 2.831942] seconds. Eligibility required at least 0.5 seconds,
no point regression in either regime and a strictly positive lower interval
bound. Bonferroni adjusts the two comparisons at a **nominal** family level
of 5%; bootstrap coverage is approximate. All 10,000 resamples share each
participant's multiplicity across both regimes. Independent direct recording
duplication reproduced the result to 1e-12. N's lower point macro is insufficient:
it regresses on unseen people and its interval crosses zero.

These are **public-training CV** results following historical development on
that release. They are not independent new-cohort confirmation. The official
49.82 result is reported separately. The old 55.816 public-test result and
original GIFs refer to the old weights; previous sampled-cache CV claims do
not establish chronological parity. No test/validation loader or sealed
feedback was used to select this repair. See the exact
[protocol](executed_study/PROTOCOL.json), [gate result](CV_GATE_RESULT.json)
and [submitted method](METHOD.md).

## Reproduce the selected model

Code is Apache-2.0. Obtain public Muse data under its original
[CC-BY-NC-SA-4.0 terms](https://github.com/neural-interfaces26/2026-competition/blob/79b44530adda3ffa687dded0046d3424caed9266/codabench/pages/scientific_faq_sleep.md).
No raw EEG, private feature cache, original identifiers, labels, per-record
scores, hidden cohort or model weights are redistributed in this public
source packet. The submitted model is retained in a private backup.

Use the official kit pinned at 79b44530adda3ffa687dded0046d3424caed9266;
the kit itself is referenced, not redistributed. NeuralBench, neuralset,
neuralfetch and neuraltrain 0.4.0 are needed for extraction. NumPy is enough
for fitting from the reconstructed private feature cache:

```bash
python reproduce_linear.py --private-cache /private/train_features_private.npz \
  --out /private/reproduced-model --require-reference
```

Alternatively, with the public study already downloaded and the official
runtime installed, reconstruct the chronological training cache and fit:

```bash
NEURALBENCH_EXTRACT_CPUS=1 python reproduce_linear.py \
  --data-dir /private/public-muse-data --official-kit /path/to/2026-competition \
  --out /private/reconstructed-model
```

The cache branch of this portable helper was executed on the server:
it reproduced the exact submitted weights, SHA-256
`a0afd3db3e227e7cea91f5fca08b680e23405afc92f4c1c7376394eddd1862c3`,
in 2.663 CPU seconds using NumPy 2.5.2. The original complete sequential
extraction was executed in 172.928 CPU seconds using NeuralBench 0.4.0 on
Windows. The portable raw-data CLI wiring has not been run again; it reuses
that extraction procedure. Other BLAS/NumPy versions may differ in final bits.
The reproduction audit and any final award remain the organisers' decision.

L is a CPU-trained linear model. The prospective comparison also tested N
on one server RTX 3090: 30 fixed fits in a 72.008-second study including CPU H/L
fits; N was negative. No GPU computation is attributed to L. Actual loaded
Solver/offline parity, independent closed-form EMA checks, and complete
ingestion/scoring of the exact final ZIP passed. Synthetic checks establish
technical integration, not predictive utility or sealed-image equivalence.

The executed-study manifest captures **private input hashes**, not a claim
that those omitted inputs are in this public archive. All published payload
hashes are separately listed in FILE_MANIFEST.json. Public code/aggregates
and private model backup are distinct from the official competition entry.

<!-- author-background:start -->
## Author background

**Francisco Angulo de Lafuente**, independent AI researcher · [ORCID](https://orcid.org/0009-0001-1634-7063) · [arXiv](https://arxiv.org/a/angulodelafuente_f_1.html) · [Google Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ) · [ResearchGate](https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3) · [Figshare](https://figshare.com/authors/Francisco_Angulo_de_Lafuente/22601234) · [Wikipedia](https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente)

Earlier research at ECOFA focused on biotechnology for clean energy and on second-generation biofuels obtained by microbiological processing of organic waste. Two Spanish patents document that work:

- **ES 2273594 B1**, *Process for obtaining fuels from organic waste*. Filed 5 September 2005, grant published 16 February 2008. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/27/35/ES-2273594_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2273594B1/en)
- **ES 2341194 B1**, *Method for obtaining paraffin by a biological process with microorganisms from organic matter, and its refining for use as fuel*. Filed 11 September 2007, grant published 26 April 2011. Also published as EP2196519 and WO2009034217. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/34/11/ES-2341194_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2341194B1/en)

These are historical publication and grant records under the inventor name Francisco Angulo Lafuente, and both have since lapsed. They are cited as evidence of a research trajectory and make no statement about current enforceability.
<!-- author-background:end -->
