# Neural URL model

`url-model.bin` is the tokenized character transformer used for neural
link compression (see `neural.js` in the repository root). It predicts
each token of a URL; an arithmetic coder converts those predictions
into near-optimal bits.

Current shipped model (payload version 3): a 21.6M-parameter student
distilled from a 71.7M teacher on GPU (3B tokens over a 19.8M-URL
Common Crawl corpus), quantized to int4 with per-group scales —
12.3MB download. On held-out URLs, hybrid payloads average about
**58% smaller than the classic scheme alone** (exact figures in the
version-3 results on the PR/branch). Beyond-context URLs are coded
in chunks (see below), so long links no longer fall back to classic.
`url-model-v1.bin` and `url-model-v2.bin` are the archived earlier
models, kept deployed so old links stay decodable.

## Format (hamr-url-model-v2 / -v3)

Little-endian binary: a `uint32` header length, a JSON header (model
dimensions, token manifest, tensor manifest), then tensor data in
manifest order. v2 tensors are float16. v3 tensors may additionally
be quantized: `dtype: "int8"` or `"int4"` with one float16 scale per
output row, or per `group` input columns when the manifest entry
sets a group size (int4 packs two values per byte, low nibble first,
stored as value + 8). The loader dequantizes to floats at load —
integer times correctly-rounded f16 scale is exact — so inference
math is identical for every format. Loaded and evaluated by
`neural.js` in plain JavaScript — no runtime dependencies, no
WebGPU/WASM, and only IEEE correctly-rounded operations so that
encoding and decoding are bit-identical on every browser and
platform. v3 files evaluate with a 4-way-unrolled matmul kernel
(fixed summation order, still deterministic); v1/v2 files keep the
original kernel forever, because their issued payloads pin its exact
arithmetic.

Payload version 3 also introduces **chunked coding**: a URL longer
than the model context is tokenized once, split into
capacity-sized chunks, and coded as one arithmetic stream in which
each in-band EOS restarts the model context. Framing is
self-describing (a full-capacity chunk's EOS means another chunk
follows; a shorter chunk ends the URL), so arbitrarily long URLs
stay on the neural path at a few bits per extra chunk.

Tokenization is greedy longest-match over the header's `tokens` list
(EOS = 0, ids follow list order from 1). The vocabulary was learned by
BPE over separator-delimited URL chunks, then applied greedily — an
exact string operation, identical in the Python trainer and the JS
engine. v1-format files (no `tokens`; character-level) still load.

At encode time, `neural.js` additionally beam-searches alternative
segmentations of the URL: the decoder never re-tokenizes — it
arithmetic-decodes a symbol stream and concatenates the tokens'
strings — so any segmentation that concatenates back to the URL
decodes identically, and the encoder keeps whichever candidate
payload is genuinely smallest (greedy is always among the candidates,
so payloads never grow). This is purely encoder behavior: payload
format, decoders, and already-issued links are unaffected. The
archived version-1 model keeps the plain greedy encoder, so its
pinned payload bits stay put.

## Architecture

- 1024-token vocabulary (~2.1 characters/token on URLs)
- 8 transformer layers, 512 hidden dim, 16 heads, 1536 MLP dim
- Learned absolute position embeddings, 96-token context (~200 chars
  per chunk; longer URLs are chunk-coded)
- RMSNorm, ReLU MLP, output head tied to the embedding table
- ~21.6M parameters; int4 weights with per-64-column f16 scales
  (embeddings, positions, and norms stay f16) ≈ 12.3MB

The architecture is restricted to operations that are deterministic in
JavaScript (`+ - * / sqrt`): no GELU/SiLU (needs `exp`/`tanh`), no
rotary embeddings (needs `sin`/`cos`). Softmax uses a deterministic
`exp` built from basic operations.

The 2M-parameter architecture of versions 1-2 won the original
screening campaign ([`harness/README.md`](harness/README.md)); the
version-3 model was distilled from a 71.7M-parameter teacher (KD is
decisive at this capacity, unlike at 2M where it tied) and
QAT-quantized - int8 beats int4 by ~1.6 est. symbols but doubles the
download, so int4-group64 shipped as the size/quality compromise.

## Training data

19.8M URLs (1.2B characters): a uniform sample across the
[Common Crawl](https://commoncrawl.org/) `CC-MAIN-2025-26` URL index
(every 25th `cluster.idx` block, capped at 30 URLs per host, 1.9M
distinct hosts) plus
[ada-url/url-dataset](https://github.com/ada-url/url-dataset)
popular-site URLs upweighted 3x. Train/val/holdout membership is
hashed from the URL string, so splits are stable and leak-free across
corpus rebuilds and corpus growth.

Distillation from the 71.7M-parameter teacher (archived under
`harness/gpu/teacher/`) tied direct training at 2M parameters but is
decisive at 21.6M (2.144 vs 2.235 bits/char at the 500M-token
screening budget) - the shipped version-3 model is the distilled one.
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
