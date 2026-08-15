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
  (currently linkVersion 3, 21.6M params, int4 weights with per-group
  f16 scales, 12.3MB); `url-model-v1.bin` and `url-model-v2.bin` are
  archived and lazy-loaded only when an old link is opened. CLI:
  `standalone.js` (`hamr`). Version 3 payloads also chunk-code URLs
  longer than the context window (in-band EOS restarts the context),
  so the neural path covers arbitrarily long URLs.

## Critical invariants — breaking these breaks issued links

1. **Determinism.** The arithmetic coder requires bit-identical
   probabilities on every browser/CPU. `neural.js` therefore uses ONLY
   IEEE correctly-rounded operations (`+ - * / sqrt`) plus its own
   `detExp`/`detLog2`; tokenization is exact string matching. NEVER
   introduce `Math.exp/pow/sin/tanh`, WebGPU/WASM inference, or float
   accumulation-order changes for an EXISTING model version — each
   version's kernel summation order is frozen the moment links are
   issued (v1/v2: sequential `matmul`; v3+: 4-way-unrolled `matmul4`;
   a future version may pick a new fixed order, gated on its format,
   but can never change an old one). Model architecture is
   constrained to what this supports: ReLU MLP, RMSNorm, learned
   absolute positions (no GELU/SiLU, no RoPE). Quantized (v3) weights
   are dequantized to floats at load — int times f16 scale is exact —
   so quantization never touches inference determinism.
2. **Model files are compatibility surfaces.** Never overwrite
   `model/url-model.bin` in place. Upgrades: train with the next
   `link_version`, archive the current file as `url-model-v<N>.bin`,
   ship the new one as `url-model.bin`, add new pinned vectors, keep
   every old version's vectors green forever. Procedure in
   `model/README.md`.
3. **Pinned vectors** in `test/neural.test.mjs` come in two kinds.
   *Decode vectors* (payload → URL) freeze compatibility with issued
   links; they are NEVER touched — if one fails, you changed decode
   behavior, fix the change. *Encode vectors* (URL → payload) pin the
   current encoder's exact output; they may be regenerated ONLY by a
   deliberate encode-side improvement (e.g. the tokenization search),
   in which case the superseded encode outputs are demoted to
   decode-only vectors, kept green forever. A NEW model version adds
   vectors; it never edits older versions'.
4. **Data splits are content-hashed** (`sha1(url) % 100` buckets in
   the harness corpus builders). Never change the bucket function or
   fractions — holdout comparability across all past campaigns
   depends on it.
5. **Classic payload format is frozen.** `compress.js` bit layout and
   its Huffman tables must stay bit-compatible (version-0 links).
   Known quirk kept on purpose: the tilde HACK in `compress.js`.

## Commands

- `npm test` — full suite (70 tests): round-trips, normalizations,
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
- **~2M params is the size/speed knee for f16** (holds at GPU scale);
  vocab 2048 ties 1024; greedy-1024 tokenization buys context
  coverage (~200 chars), not bits/char. Quantization moved the knee:
  the v3 model is 21.6M params at 12.3MB (int4, group-64 scales).
- **Distillation tied direct training at 2M params** (student
  capacity was the binding constraint) but is **decisive at 21.6M**
  (2.144 vs 2.235 bits/char at the 500M screening budget) — the v3
  model is distilled from the archived 71.7M teacher
  (1.5693 bits/char, `model/harness/gpu/teacher/`).
- **QAT quantization ladder at 21.6M** (shared eval bits/char):
  fp16 1.721, int8 1.705 (22.2MB), int4 group-64 1.868 (12.3MB),
  int4 per-channel 1.937 (11.7MB). int8 reaches −61% vs classic but
  at 2× the download envelope; int4-g64 shipped.
- Estimated entropy floor of the URL distribution: ~1.2–1.4
  bits/char; the 71.7M teacher (1.57) approaches it.
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
