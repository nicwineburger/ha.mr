# Neural URL model

`url-model.bin` is the character-level transformer used for neural link
compression (see `neural.js` in the repository root). It predicts each
character of a URL; an arithmetic coder converts those predictions into
near-optimal bits.

## Format

Little-endian binary: a `uint32` header length, a JSON header (model
dimensions and tensor manifest), then float16 tensor data in manifest
order. Loaded and evaluated by `neural.js` in plain JavaScript — no
runtime dependencies, no WebGPU/WASM, and only IEEE correctly-rounded
operations so that encoding and decoding are bit-identical on every
browser and platform.

## Architecture

- Character vocabulary: EOS + printable ASCII `0x21..0x7E` (95 symbols)
- 4 transformer layers, 128 hidden dim, 4 heads, 384 MLP dim
- Learned absolute position embeddings, 128-token context
- RMSNorm, ReLU MLP, output head tied to the embedding table
- ~685K parameters ≈ 1.4MB as float16

The architecture is deliberately restricted to operations that are
deterministic in JavaScript (`+ - * / sqrt`): no GELU/SiLU (needs
`exp`/`tanh`), no rotary embeddings (needs `sin`/`cos`). Softmax is
computed with a deterministic `exp` built from basic operations.

## Training data

- ~340K URLs sampled from the [Common Crawl](https://commoncrawl.org/)
  URL index (crawl `CC-MAIN-2025-26`), spread uniformly across the
  domain space via `cluster.idx` block sampling, capped at 30 URLs per
  host
- ~100K popular-site URLs from
  [ada-url/url-dataset](https://github.com/ada-url/url-dataset),
  upweighted 3x (popular links are what people actually shorten)

URLs are scheme-stripped (the scheme is a separate payload bit) and
packed into 128-character windows aligned to URL starts, separated by
EOS. 2000 URLs are held out before training for validation and
benchmarking.

## Reproducing

```sh
./fetch-data.sh          # downloads + curates the corpus (~250MB)
pip install torch numpy  # CPU build is fine
python3 train.py         # ~30 min on 4 CPU cores; writes url-model.bin
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
