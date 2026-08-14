# Neural URL model

`url-model.bin` is the tokenized character transformer used for neural
link compression (see `neural.js` in the repository root). It predicts
each token of a URL; an arithmetic coder converts those predictions
into near-optimal bits.

Current shipped model (payload version 2): trained on GPU for 3B
tokens over a 19.8M-URL corpus sampled from the Common Crawl index —
see [`harness/gpu/RESULTS.md`](harness/gpu/RESULTS.md). On 1,000
held-out URLs, hybrid payloads average **51.8% smaller than the
classic scheme alone**. `url-model-v1.bin` is the archived first
model, kept deployed so version-1 links stay decodable.

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

19.8M URLs (1.2B characters): a uniform sample across the
[Common Crawl](https://commoncrawl.org/) `CC-MAIN-2025-26` URL index
(every 25th `cluster.idx` block, capped at 30 URLs per host, 1.9M
distinct hosts) plus
[ada-url/url-dataset](https://github.com/ada-url/url-dataset)
popular-site URLs upweighted 3x. Train/val/holdout membership is
hashed from the URL string, so splits are stable and leak-free across
corpus rebuilds and corpus growth.

Distillation from a 71.7M-parameter teacher was evaluated and tied
direct training at this model size (the student's capacity is the
binding constraint) - the shipped model is the directly-trained one.
Details in [`harness/gpu/RESULTS.md`](harness/gpu/RESULTS.md).

## Reproducing

GPU path (what produced the shipped model - Modal, ~$25):

```sh
cd harness/gpu
modal run modal_corpus.py --stage all
modal run modal_train.py --action train --config '{...}'  # see gpu/README.md
```

CPU path (smaller corpus, weaker model, no accounts needed):

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

## Versioning and the upgrade path

Neural payloads can only be decoded by the exact model that encoded
them, so every model carries a `linkVersion` in its header, and every
payload records its model's version in the payload's unary version
marker. Upgrading to a retrained model **never breaks old links** if
you follow the path:

1. Train the new model with `link_version` bumped to the next number
   (see `harness/train-final.py`).
2. Archive the current model: `url-model.bin` → `url-model-v<old>.bin`.
3. Ship the new model as `url-model.bin` (the stable "latest" URL).

Encoding always uses the latest model. Decoding reads the payload's
version: if it matches the latest model, that (already fetched) model
is used; otherwise `url-model-v<N>.bin` is lazy-loaded — visitors only
ever download the archived file when opening an old link, so retained
versions cost new users nothing. Update the pinned vectors in
`test/neural.test.mjs` for the new model, keeping the old model's
vectors green forever.

**Never replace `url-model.bin` in place without bumping
`linkVersion` and archiving the old file** — that is the one move that
breaks issued links, and the pinned-vector tests exist to catch it.
Classic (version 0) links never depend on any model.
