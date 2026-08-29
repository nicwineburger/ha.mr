# WASM engine prototype: measurements

Environment: 4-core Intel Xeon @ 2.80GHz (this container), Node
v22.22.2, headless Chromium 141.0.7390.37 (Playwright 1.56.1),
Ubuntu clang 18.1.3. JS and WASM engines run the identical pipeline
(same arithmetic coder, same tokenization search, same model files);
only the transformer inference differs. Benchmarks ran on an
otherwise idle machine; the bit-identity sweeps ran first.

## 1. Bit-identity (acceptance)

The WASM engine produced **byte-identical payloads and decodes** in
every check performed, in Node and in headless Chromium:

- `npm test` (wasm/wasm.test.mjs): pinned-vector links for v1, v2,
  and v3 (encode with search on and off, decode), the beyond-context
  chunked link, held-out URLs, refusal parity — green.
- Full sweep, Node (`wasm/verify.mjs`, holdout-4k corpus): all
  pinned-vector links of all three versions plus 100 held-out URLs
  against v3 and v2, encode search on/off plus decode:
  **878 checks, 0 mismatches**.
- Full sweep, headless Chromium (`wasm/verify-browser.mjs`), same
  links and 100 held-out URLs: **0 mismatches**.

This is the expected result, not luck: WASM f64 arithmetic is
IEEE-754 correctly rounded exactly like JS, and `engine.c` performs
the same operations in the same order as `neural.js` (the v3
`matmul4` kernel maps its interleaved accumulators s0..s3 onto two
`f64x2` SIMD lanes with the identical `(s0+s1)+(s2+s3)` combine; the
v1/v2 sequential kernel stays scalar; no FMA, no relaxed SIMD, no
libm). A rebuild of `engine.wasm` from source is byte-identical
(sha256 d01494e9...), and CI re-runs the bit-identity tests on
different hardware on every push.

## 2. Speed

30 held-out URLs per row (tokenizable, in-context so the search path
is active on all of them), 3 warmup URLs excluded, per-URL times.
p90 tracks URL length (the 30-URL set spans ~20-130 chars).

**v3 model (shipped, 21.6M params) — ms/URL:**

| operation | JS Node | WASM Node | speedup | JS Chromium | WASM Chromium | speedup |
|---|---|---|---|---|---|---|
| encode, search (median) | 12,657 | 2,690 | **4.7x** | 7,947 | 2,636 | **3.0x** |
| encode, search (p90) | 33,500 | 7,156 | 4.7x | 20,514 | 7,049 | 2.9x |
| encode, fast path (median) | 1,263 | 267 | **4.7x** | 783 | 258 | **3.0x** |
| encode, fast path (p90) | 3,290 | 698 | 4.7x | 2,096 | 684 | 3.1x |
| decode (median) | 1,267 | 268 | **4.7x** | 784 | 258 | **3.0x** |
| decode (p90) | 3,224 | 709 | 4.5x | 2,064 | 680 | 3.0x |

**v2 model (archived, 2.06M params) — ms/URL (median):**

| operation | JS Node | WASM Node | speedup | JS Chromium | WASM Chromium | speedup |
|---|---|---|---|---|---|---|
| encode, search | 1,238 | 519 | 2.4x | 878 | 527 | 1.7x |
| encode, fast path | 123 | 53 | 2.3x | 85 | 53 | 1.6x |
| decode | 123 | 52 | 2.4x | 85 | 53 | 1.6x |

**One-time overhead (v3):** WASM instantiation + weight copy-in
86 ms (Node) / 136 ms (Chromium) — on top of the model parse
(229 / 188 ms) that the JS engine also pays. v2: 9-10 ms.

Notes:

- WASM time is nearly identical across Node and Chromium (same
  binary, same V8 backend); the JS baseline is faster in Chromium
  than in Node (different V8 tiering defaults), which is why the
  speedup factor differs. Take the Chromium column as the
  user-facing one: **~3x** across all three operations on v3.
- The win concentrates exactly where it matters: the fast-path
  encode (per-keystroke rendering) drops from ~0.8-1.3 s to
  ~0.26 s median, and decode (link opening) the same — v3 starts
  feeling interactive. The searched encode drops from ~8-13 s to
  ~2.6 s.
- v1/v2 gain only 1.6-2.4x because their frozen sequential kernel
  cannot be SIMD-vectorized (summation order is a compatibility
  surface); they get the scalar-WASM win only. An acceptable
  production shape is v3-on-WASM with archived versions staying on
  the JS engine entirely.
- Prototype memory cost: the weights exist twice (JS Float32Arrays
  + the f32 copy in WASM linear memory, ~86 MB each for v3), plus a
  bump-allocated session arena (tens of MB peak during search).
  Production should drop the JS-side tensor copies after copy-in.

## 3. Engine size

| component | raw | gzip -9 |
|---|---|---|
| engine.wasm | 7,777 B | 2,795 B |
| engine.js (glue + pipeline port) | 15,267 B | 5,385 B |
| neural.js (whole shipped JS engine, incl. loader it reuses) | 31,178 B | 10,813 B |

The WASM engine adds ~8 KB (~3 KB over the wire) and cannot replace
neural.js (the loader, tokenizer, and payload framing are reused, and
the JS engine remains the fallback). Engine size is noise against the
12.3 MB model download; reported for completeness.

## 4. Download reduction: weight compression

Measured with `wasm/weights/measure.mjs`: gzip -9 and brotli -q11
baselines (what a CDN does with zero client code) vs "wpack", a
prototype entropy-coded container (static rANS: order-1 over int4
nibbles, order-0 over int8 bytes and phase-split f16 bytes; gzipped
JSON header), decoded at load in JS and verified **byte-identical**
(sha256) to the original file. Decode-at-load is not
determinism-sensitive - only the reconstructed bytes matter, and they
are checked.

**Shipped int4 model (`model/url-model.bin`, 12,315,586 B):**

| method | size | saving | note |
|---|---|---|---|
| raw | 11.75 MB | - | |
| gzip -9 | 10.24 MB | -12.8% | 0.4 s compress |
| brotli -q11 | 10.03 MB | **-14.6%** | 25 s compress (offline) |
| wpack | 10.05 MB | -14.5% | decode-at-load 0.9 s (Node, JS decoder) |

**int8 candidate (`model/harness/gpu/candidates/url-model-int8.bin`,
22,211,106 B, the -61.2%-vs-classic QAT export):**

| method | size | saving | note |
|---|---|---|---|
| raw | 21.18 MB | - | |
| gzip -9 | 19.19 MB | -9.4% | |
| brotli -q11 | 19.01 MB | **-10.2%** | |
| wpack | 19.02 MB | -10.2% | decode-at-load 0.9 s |

Why the ceiling is this low: QAT weights are already
near-incompressible. Measured empirical entropies -

- int4 nibbles: 3.447 bits of 4 order-0, and 3.446 order-1 -
  neighboring weights are statistically independent, so context
  modeling buys nothing;
- int8 bytes: 7.201 bits of 8 order-0;
- f16 (embeddings + scales): the skew lives in the sign/exponent
  byte (5.15-5.62 bits of 8); mantissa low bytes are ~8.0 bits
  (pure noise). Order-1 on the high bytes (4.53 bits) is worth only
  ~70 KB more on the int4 file.

A custom container therefore only ties brotli (and brotli over the
container gains nothing - the entropy is gone). The honest
conclusion: **transparent compression is worth ~15% on the shipped
model (best served by CDN/`Content-Encoding: br`, zero client code)
and ~10% on int8, and no clever client-side codec will change that
materially.**

**int8-as-payload-v4 assessment:** int8 lands at **19.0 MB
compressed vs 10.0 MB** for the shipped int4 - still 1.9x the
download envelope for an 8.7% bits/char improvement (1.705 vs 1.868;
-61.2% vs classic on the benchmark). Compression does not
change the tradeoff that kept int8 from shipping. Not recommended
as the default model. If its savings are ever wanted, the shape
would be int4-first with int8 as a lazily-fetched upgrade for repeat
visitors (payload v4 links decode only after the big fetch) - a
product decision, not a compression problem.

## 5. Ship path (what productionization requires)

1. **Engine selection / fallback.** Feature-detect
   `WebAssembly.validate` on a tiny SIMD128 probe; use WASM when
   available, `neural.js` otherwise. Because outputs are
   bit-identical, the choice is invisible - links encoded by either
   engine decode on both, so there is no compatibility policy to
   design, only a performance one. The JS engine stays forever as
   reference implementation and fallback. SIMD128 has been in all
   evergreen browsers since 2021; the fallback covers the rest.
2. **Verification matrix.** Extend open follow-up #3: run the pinned
   vectors under V8, SpiderMonkey, and JSC with BOTH engines (jsvu
   or a Playwright chromium/firefox/webkit matrix). WASM actually
   narrows the surface - one binary, one semantics - but the
   guarantee should be tested, not argued. Gate any `engine.wasm`
   rebuild on the full bit-identity suite (already wired: CI runs
   wasm/wasm.test.mjs on every push).
3. **Where the .wasm lives.** As a sibling static file (like the
   model files), NOT inlined into index.html - index.html/404.html
   must stay byte-identical and small. ~3 KB gzipped is one
   round-trip alongside the 10 MB model fetch. Load sequencing:
   fetch engine.wasm + model in parallel, compile the module
   (instant at 8 KB), parse the model, copy tensors in (~90 ms),
   and swap engines whenever ready - the JS engine can serve the
   first keystrokes and the swap changes nothing observable.
   Production should parse tensors directly into WASM memory and
   drop the JS-side copies (~86 MB saved vs this prototype).
4. **Scope.** Ship v3-on-WASM only; archived v1/v2 decodes are rare,
   their JS path is already fast enough (~120 ms), and keeping them
   JS-only avoids carrying the scalar kernel's test surface.

**Recommendation: productionize.** ~3x on decode and fast-path
encode in the browser (0.26 s vs 0.78 s median; more on slower
devices, where it matters most) for +3 KB of download and zero
compatibility risk, enforced by bit-identity tests rather than
promised. The weight-compression answer is to turn on brotli at the
CDN (-14.6% today, no client work) and to leave int8 archived.
