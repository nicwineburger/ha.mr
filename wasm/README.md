# WASM inference engine (measurement prototype)

Nothing in this directory ships. It exists to answer, with hard
numbers, whether a WebAssembly inference engine is worth
productionizing: how much faster the client gets, and whether the
download shrinks. Results and the ship recommendation are in
[REPORT.md](REPORT.md). The shipped engine (`neural.js`), the payload
formats, and the model files are untouched.

## Why WASM can be bit-identical (the design core)

The arithmetic coder requires bit-identical probabilities everywhere.
JavaScript and WebAssembly both implement IEEE-754 binary64 with
correctly-rounded `+ - * / sqrt` (and `floor`), so a WASM engine that
performs the **same operations in the same order** as `neural.js`
produces bit-identical results — determinism by faithful
transcription, not by luck. Concretely (`engine.c`):

- `detExp`, softmax, RMSNorm, and the kernels are transcribed
  op-for-op; no libm, no `-ffast-math`, no reassociation, no FMA.
- The v3 `matmul4` kernel keeps its four interleaved partial sums
  `s0..s3` combined as `(s0+s1)+(s2+s3)`; the WASM version maps the
  pairs (s0,s1) and (s2,s3) onto two `f64x2` SIMD accumulators — the
  identical summation order, four multiply-adds per iteration. Plain
  SIMD128 is per-lane IEEE-exact; relaxed-simd instructions are never
  used (they are non-deterministic by design).
- v1/v2 models use the sequential single-accumulator `matmul`, whose
  order cannot be vectorized; it stays scalar in WASM.
- `detLog2` (encoder search costs only) stays in JS, copied verbatim
  in `engine.js` — same source, same ops, same bits.

The acceptance criterion is byte-identical payloads and decodes
against the JS engine for every pinned-vector link of every model
version plus held-out URLs, in Node and headless Chromium
(`wasm.test.mjs` runs a subset under `npm test`; `verify.mjs` /
`verify-browser.mjs` run the full sweep).

## Files

- `engine.c` / `engine.wasm` — the inference engine (one `feed` =
  one transformer step) and its checked-in build artifact. Treat the
  binary like vendored code: rebuild whole with `build.sh`, never
  patch. The JS side owns all layout: weights are copied in as f32
  (exact, same as `neural.js` storage; math is f64), sessions get
  bump-allocated key/value slabs plus per-session offset tables so
  `fork` shares cached positions by pointer, exactly like the JS
  engine's write-once arrays.
- `engine.js` — glue plus a faithful port of the `neural.js`
  encode/decode pipeline (chunking, tokenization search, payload
  framing) with the WASM engine underneath. Reuses the shipped
  `URLModel` loader and arithmetic coder.
- `wasm.test.mjs` — bit-identity tests wired into `npm test`.
- `verify.mjs`, `verify-browser.mjs` — full acceptance sweeps.
- `bench.mjs`, `bench-browser.mjs` — ms/URL measurements (median,
  p90) for decode, encode fast path, encode with search.
- `browser/` — static harness page + Playwright driver (Playwright
  resolved from the environment; not a dependency).
- `weights/` — transparent weight-compression experiment: gzip and
  brotli baselines vs an entropy-coded container (static rANS,
  order-1 nibbles) with a byte-identity round-trip check.

## Build recipe (engine.wasm)

```
./wasm/build.sh
```

which runs exactly:

```
clang --target=wasm32 -O2 -msimd128 -nostdlib -ffreestanding \
  -Wall -Wextra \
  -Wl,--no-entry -Wl,--export=__heap_base -Wl,-z,stack-size=131072 \
  -o engine.wasm engine.c
```

Toolchain used for the checked-in binary: Ubuntu clang 18.1.3
(`clang --target=wasm32`, builtin `wasm_simd128.h`) with its bundled
`wasm-ld`. The build is reproducible: same compiler, same flags, same
source produce the same binary. Never add `-ffast-math`,
`-mrelaxed-simd`, or any flag that licenses reassociation — kernel
summation order is the compatibility surface.

Determinism is additionally *tested*, not assumed: any rebuild must
keep `npm test` green (bit-identity against the JS engine on pinned
links of all three model versions) before the binary is committed.

## Running the measurements

```
# bit-identity sweeps (urls file: one URL per line)
node wasm/verify.mjs <urls-file> 100
node wasm/verify-browser.mjs <urls-file> 100

# speed (Node and headless Chromium), v3 and v2
node wasm/bench.mjs <urls-file> 30 3
node wasm/bench-browser.mjs <urls-file> 30 3

# weight compression (shipped int4 model, archived int8 candidate)
node wasm/weights/measure.mjs model/url-model.bin
node wasm/weights/measure.mjs model/harness/gpu/candidates/url-model-int8.bin
```
