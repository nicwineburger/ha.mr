# Neural URL model

`url-model.bin` is the tokenized character transformer used for neural
link compression (see `neural.js` in the repository root). It predicts
each token of a URL; an arithmetic coder converts those predictions
into near-optimal bits.

## Format (hamr-url-model-v2)

Little-endian binary: a `uint32` header length, a JSON header (model
dimensions, token manifest, tensor manifest), then float16 tensor data
in manifest order. Loaded and evaluated by `neural.js` in plain
JavaScript — no runtime dependencies, no WebGPU/WASM, and only IEEE
correctly-rounded operations so that encoding and decoding are
bit-identical on every browser and platform.

Tokenization is greedy longest-match over the header's `tokens` list
(EOS = 0, ids follow list order from 1). The vocabulary was learned by
BPE over separator-delimited URL chunks, then applied greedily — an
exact string operation, identical in the Python trainer and the JS
engine. v1-format files (no `tokens`; character-level) still load.

## Architecture

- 1024-token vocabulary (~2.1 characters/token on URLs)
- 5 transformer layers, 192 hidden dim, 6 heads, 576 MLP dim
- Learned absolute position embeddings, 96-token context (~200 chars)
- RMSNorm, ReLU MLP, output head tied to the embedding table
- ~2.06M parameters ≈ 4.1MB as float16

The architecture is restricted to operations that are deterministic in
JavaScript (`+ - * / sqrt`): no GELU/SiLU (needs `exp`/`tanh`), no
rotary embeddings (needs `sin`/`cos`). Softmax uses a deterministic
`exp` built from basic operations.

This configuration won a controlled screening campaign over
tokenizers, vocab sizes, and model scales — see
[`harness/README.md`](harness/README.md) for the methodology and full
results table.

## Training data

~1.68M URLs (102M characters): a uniform sample across the
[Common Crawl](https://commoncrawl.org/) `CC-MAIN-2025-26` URL index
(via `cluster.idx` block sampling, capped at 30 URLs per host,
153K distinct hosts) plus
[ada-url/url-dataset](https://github.com/ada-url/url-dataset)
popular-site URLs upweighted 3x. Train/val/holdout membership is
hashed from the URL string, so splits are stable and leak-free across
corpus rebuilds.

## Reproducing

```sh
cd harness
./fetch-data.sh          # corpus + vocab + baselines (~500MB download)
pip install torch numpy  # CPU build is fine
python3 train-final.py   # ~80 min on 4 CPU cores; writes ../url-model.bin
```

Training is seeded but not guaranteed bit-reproducible across
hardware/library versions. That's fine: the model file itself is the
source of truth, and determinism is only required of *inference*, which
`neural.js` guarantees for any given model file.

**Compatibility note:** replacing `url-model.bin` with a retrained
model breaks all previously issued neural (version 1) links — the
decoder must use the exact model that encoded them. Classic (version 0)
links are unaffected. If you retrain, either keep the old model
deployed for decoding or accept the break on your own deployment.
