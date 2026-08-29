# Archived model candidates

Files here are **archived candidates only — none of them is a shipped
model**. They are kept in-repo so a future decision doesn't depend on
the Modal volume (`hamr-gpu`), which is not permanent storage.

## `url-model-int8.bin`

The int8 per-channel QAT variant of the payload-version-3 architecture
(the same 21.6M-param, dim-512/8-layer/16-head/mlp-1536 model that
shipped as `url-model.bin`, distilled from the 71.7M teacher), trained
during the v3 campaign but **not shipped** because it is 2x the
download of the int4-group-64 export that shipped instead.

- **Run**: `v3-qat8` (`model/harness/gpu/results.jsonl`), initialized
  from the `v3-final` checkpoint and fine-tuned 400M tokens of
  QAT (`qat: "int8"`, symmetric per-output-channel scales,
  straight-through estimator — see `build_model()` in
  `../modal_train.py`), lr 2e-4, warmup 200, same KD config as
  `v3-final` (teacher `g-teacher768`, `kd_alpha` 0.5, `kd_temp` 1.0).
  24.1 train-minutes on Modal (A10G/A100 per the `hamr-train` app).
- **Checkpoint**: `ckpt-v3-qat8.pt` on the `hamr-gpu` volume; export
  already existed on the volume as `url-model-v3-int8.bin` (produced
  by `export_model()` with `quant: "int8"`) and was downloaded
  unchanged — no retraining or re-export was needed.
- **Numbers** (shared eval, 3,995 held-out URLs — same set as every
  other row in `results.jsonl`):
  - **1.705 bits/char** (`bits_per_char: 1.7048` in `results.jsonl`)
  - **−61.2% vs classic** on the 500-URL benchmark
    (`model/benchmark.mjs`)
  - **22.2MB** (22,211,106 bytes; int8 weights, one f16 scale per
    output row — "per-channel", no column grouping)
  - 21,553,664 parameters (verified by loading the file with
    `neural.js`'s `URLModel` — see below)
- **QAT quantization ladder** this run belongs to, all at the same
  21.6M architecture (`CLAUDE.md`, `model/README.md`): fp16 1.721
  (`v3-final`), **int8 1.705, 22.2MB (this file)**, int4 group-64
  1.868, 12.3MB (shipped, `v3-qat4g` → `url-model.bin`), int4
  per-channel 1.937, 11.7MB (`v3-qat4`). int8 is the highest-quality
  point on the ladder but loses on size; int4-group-64 is the
  size/quality compromise that actually shipped.

### Verified load

Loaded with `neural.js`'s `URLModel` from the raw file buffer:

```
dim: 512, layers: 8, heads: 16, mlpDim: 1536, maxLen: 96, vocab: 1024
linkVersion: 3, fastKernels: true, tensorCount: 51
params: 21553664
```

### Why this is archived, not shipped

Candidate for a possible **payload version 4**, pending two
measurements happening in parallel elsewhere: a WASM inference engine
and weight-compression work that could shrink int8's download penalty
below 2x. If those land favorably, this file (or a fresh export of
the same `ckpt-v3-qat8.pt` checkpoint) is the natural v4 candidate.
Until then it stays out of the shipped path entirely.

### It must NEVER ship under `linkVersion 3`

The file's embedded header carries `format: "hamr-url-model-v3"` and
`linkVersion: 3` — those are the values `export_model()` stamped based
on the training config, **not** a statement that this file is
compatible with the current v3 deployment. Payload version 3's issued
links are pinned to the int4-group-64 weights that actually shipped
as `model/url-model.bin` (frozen the moment those links were issued,
per `CLAUDE.md`'s invariant #2/#3). This file's weights are different
(int8, not int4-group-64), so loading it as-is under `linkVersion: 3`
would silently produce different arithmetic-coder probabilities than
what v3 links were encoded with — corrupting decode for every v3 link
that happens to hit this file instead of the real `url-model.bin`.

If this candidate is ever promoted to ship, it must go through the
documented upgrade path in `model/README.md` ("Versioning and the
upgrade path"): re-export from `ckpt-v3-qat8.pt` with `link_version`
bumped to **4**, archive the current `url-model.bin` as
`url-model-v3.bin`, ship the re-exported file as the new
`url-model.bin`, and add new pinned vectors in `test/neural.test.mjs`
for version 4 while keeping every existing version's vectors green.
It can never simply be copied in as `url-model.bin` in place.
