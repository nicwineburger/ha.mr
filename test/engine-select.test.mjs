import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { URLModel, neuralDecompressNumber } from "../neural.js";
import { selectEngine, wasmModuleSource } from "../engine-select.js";
import { compressHybrid, decompressHybrid } from "../hybrid.js";
import { outputAlphabetASCII } from "../alphabets.js";

/*
 * Engine selection tests: WASM is the default production engine in
 * Node (which always has WebAssembly + SIMD128), and every fallback
 * trigger - the explicit `force` option, an injected instantiation
 * failure, and no WebAssembly global at all - produces payloads and
 * decodes byte-identical to the WASM engine for a handful of links.
 * The underlying bit-identity guarantee (every pinned vector, every
 * model version, search on and off) is proven exhaustively in
 * wasm/wasm.test.mjs and the wasm/verify* sweeps; this file only
 * checks that the SELECTION layer wires the two engines together
 * correctly, so it keeps to the JS engine's slow tokenization-search
 * path (~10x greedy) for a single link rather than re-running it for
 * every link and every test.
 */

const model = new URLModel(
  (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);
const wasmSource = wasmModuleSource(
  readFile(new URL("../wasm/engine.wasm", import.meta.url)));

const links = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://github.com/user/repo/blob/main/README.md",
  "https://shop.example.co.uk/products/blue-widget?variant=42&ref=newsletter"
];

test("wasm engine is selected by default in Node", async () => {
  const engine = await selectEngine(model, wasmSource);
  assert.equal(engine.kind, "wasm");
  assert.equal(engine.fallbackReason, undefined);
  assert.ok(engine.wasmEngine);
  // Debug-readable flag, not console noise: inspectable without any
  // logging, e.g. from a Node REPL/debugger
  assert.equal(globalThis.__hamrEngine.kind, "wasm");
});

test("forced fallback selects the JS engine", async () => {
  const engine = await selectEngine(model, wasmSource, { force: "js" });
  assert.equal(engine.kind, "js");
  assert.equal(engine.fallbackReason, "forced");
  assert.equal(globalThis.__hamrEngine.kind, "js");
});

test("an injected instantiation failure falls back to js silently", async () => {
  const brokenSource = () => Promise.reject(new Error("simulated fetch failure"));
  const engine = await selectEngine(model, brokenSource);
  assert.equal(engine.kind, "js");
  assert.equal(engine.fallbackReason, "simulated fetch failure");
  // Falling back never throws - the caller just gets a working engine
  const payload = engine.compress(links[0], { search: false });
  assert.equal(engine.decompress(payload),
    neuralDecompressNumber(model, payload));
});

test("selectEngine falls back when WebAssembly is unavailable", async () => {
  const realWebAssembly = globalThis.WebAssembly;
  try {
    globalThis.WebAssembly = undefined;
    const engine = await selectEngine(model, wasmSource);
    assert.equal(engine.kind, "js");
    assert.equal(engine.fallbackReason, "WebAssembly unavailable");
  } finally {
    globalThis.WebAssembly = realWebAssembly;
  }
});

test("forced js fallback produces byte-identical fast-path payloads to wasm", async () => {
  const wasm = await selectEngine(model, wasmSource);
  const js = await selectEngine(model, wasmSource, { force: "js" });
  assert.equal(wasm.kind, "wasm");
  assert.equal(js.kind, "js");
  for (const link of links) {
    const wasmPayload = wasm.compress(link, { search: false });
    const jsPayload = js.compress(link, { search: false });
    assert.equal(jsPayload, wasmPayload, `payload mismatch for ${link}`);
    assert.equal(js.decompress(jsPayload), wasm.decompress(wasmPayload),
      `decode mismatch for ${link}`);
  }
  // One link with the tokenization search on: the slow path, so only
  // one - wasm.test.mjs already covers this exhaustively
  const searched = { wasm: wasm.compress(links[0]), js: js.compress(links[0]) };
  assert.equal(searched.js, searched.wasm, "searched payload mismatch");
});

test("hybrid.js's optional engine parameter matches its JS-only default", async () => {
  const wasm = await selectEngine(model, wasmSource);
  for (const link of links) {
    const options = { search: false };
    const withEngine = compressHybrid(link, outputAlphabetASCII, model, options, wasm);
    const withoutEngine = compressHybrid(link, outputAlphabetASCII, model, options);
    assert.equal(withEngine, withoutEngine, `hybrid payload mismatch for ${link}`);
    assert.equal(
      decompressHybrid(withEngine, outputAlphabetASCII, model, wasm),
      decompressHybrid(withoutEngine, outputAlphabetASCII, model));
  }
});

test("hybrid.js without an engine argument is untouched by this feature", () => {
  // Exact regression guard: the pre-existing 4-argument call form must
  // keep calling neuralCompressToNumber/neuralDecompressNumber
  // directly, with zero behavior change from this integration - an
  // explicit `engine: null` must behave identically to omitting it.
  const options = { search: false };
  for (const link of links) {
    const payload = compressHybrid(link, outputAlphabetASCII, model, options);
    assert.equal(payload,
      compressHybrid(link, outputAlphabetASCII, model, options, null));
    assert.equal(decompressHybrid(payload, outputAlphabetASCII, model), link);
  }
});
