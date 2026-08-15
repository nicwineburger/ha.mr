# CLAUDE.md

ha.mr compresses links and QR codes entirely client-side — no backend,
no database. The compressed payload lives inside the short link itself
(`https://ha.mr#<payload>`, or `HTTP://HA.MR/<payload>` for QR codes,
decoded by `404.html`). Deployed on GitHub Pages from the repo root.

## Architecture

- **Zero-build static site.** No bundler, no framework. `index.html`
  and `404.html` MUST stay byte-identical (CI enforces this; QR links
  depend on it). Serve from the domain root; the site is
  domain-portable (links/branding derive from `location`).
- **Hybrid compression** (`hybrid.js`): every link is encoded with the
  classic scheme (`compress.js`: Huffman dictionaries + subalphabets)
  and the neural scheme (`neural.js`: tiny transformer driving the
  arithmetic coder in `arithmetic-coder.js`); the smaller payload
  wins. A unary version field in the payload records the choice:
  version 0 = classic, version N>=1 = the model whose `linkVersion`
  is N.
- **Models** live in `model/`: `url-model.bin` is the latest
  (currently linkVersion 2, 2.06M params, 4.1MB f16);
  `url-model-v1.bin` is archived and lazy-loaded only when a
  version-1 link is opened. CLI: `standalone.js` (`hamr`).

## Critical invariants — breaking these breaks issued links

1. **Determinism.** The arithmetic coder requires bit-identical
   probabilities on every browser/CPU. `neural.js` therefore uses ONLY
   IEEE correctly-rounded operations (`+ - * / sqrt`) plus its own
   `detExp`; tokenization is exact string matching. NEVER introduce
   `Math.exp/pow/sin/tanh`, WebGPU/WASM inference, or float
   accumulation-order changes. Model architecture is constrained to
   what this supports: ReLU MLP, RMSNorm, learned absolute positions
   (no GELU/SiLU, no RoPE).
2. **Model files are compatibility surfaces.** Never overwrite
   `model/url-model.bin` in place. Upgrades: train with the next
   `link_version`, archive the current file as `url-model-v<N>.bin`,
   ship the new one as `url-model.bin`, add new pinned vectors, keep
   every old version's vectors green forever. Procedure in
   `model/README.md`.
3. **Pinned vectors** in `test/neural.test.mjs` freeze
   payload bits per model version. If one fails, you changed encoding
   behavior — fix the change, don't update the vector (only a NEW
   version legitimately adds vectors).
4. **Data splits are content-hashed** (`sha1(url) % 100` buckets in
   the harness corpus builders). Never change the bucket function or
   fractions — holdout comparability across all past campaigns
   depends on it.
5. **Classic payload format is frozen.** `compress.js` bit layout and
   its Huffman tables must stay bit-compatible (version-0 links).
   Known quirk kept on purpose: the tilde HACK in `compress.js`.

## Commands

- `npm test` — full suite (37 tests): round-trips, normalizations,
  pinned vectors per version, version routing, CLI. CI also diffs
  `index.html`/`404.html`.
- `node model/benchmark.mjs <urls-file> [limit]` — real-coder
  benchmark vs classic, verifies every round-trip.
- CPU experiments: `model/harness/` (see its README; config-driven,
  seeded, `fetch-data.sh` builds the corpus).
- GPU experiments: `model/harness/gpu/` on Modal (needs
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` env vars — set in the "Modal"
  Claude environment, never committed). Corpus + checkpoints live on
  the Modal volume `hamr-gpu`.

## What's been established (don't re-derive)

Full results: `model/harness/README.md` (CPU campaign) and
`model/harness/gpu/RESULTS.md` (GPU campaign). Headlines:

- v2 model: **−51.8% vs classic** on 1,000 held-out URLs (v1: −39.0%
  on the same set). Data at fixed size was the win: 16× corpus, 40×
  tokens on the same 2.06M architecture.
- **~2M params is the size/speed knee** (holds at GPU scale); vocab
  2048 ties 1024; greedy-1024 tokenization buys context coverage
  (~200 chars), not bits/char.
- **Distillation tied direct training** at this model size (71.7M
  teacher at 1.5693 bits/char; student capacity is the binding
  constraint at a 3B-token budget). Don't revisit without changing
  the student's budget.
- Estimated entropy floor of the URL distribution: ~1.2–1.4
  bits/char; practical ceiling for this envelope ≈ −55%.
- **Transform-unwrapping is a dead end with this model**
  (`model/harness/transforms/REPORT.md`): only 2.13% of URLs carry
  decodable substructure, ~96% of detections lose after tree
  overhead, and the model already codes percent/base64 spans as seen
  in training (unwrapped JSON costs MORE). Corpus gain +0.23% vs
  +0.27% for plain chunked coding of long URLs — chunking is the
  follow-up worth shipping (in a future payload version); transforms
  only merit revisiting if a model is trained on unwrapped text.

## Open follow-ups (each ships as a new payload version, breaking nothing)

1. **Quantization**: int8/int4 student weights = 2–4× more params per
   MB of download (dequantize to f64 at load, so determinism is
   unaffected; inference *time* still scales with params). Needs QAT
   for int4, a loader change in `neural.js`, and harness validation.
2. **Drift evaluation**: rerun the benchmark against a future crawl's
   holdout; retrain (as a new version) when savings sag.
3. **Multi-engine determinism CI**: run pinned vectors under
   V8/SpiderMonkey/JSC (jsvu or Playwright matrix) to turn the
   determinism argument into a tested guarantee.

## Conventions

- Two-space indent, space before function parens (`function foo (x)`),
  JSDoc on exported functions, comments explain constraints not
  mechanics. Tests use `node:test` + `assert/strict`.
- Vendored files (`qrcode.js`) are replaced whole, never edited.
- The fonts are self-hosted (`fonts/`, OFL license included).
