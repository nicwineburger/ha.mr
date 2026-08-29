/**
 * @file Chooses the inference engine used for neural compression: the
 * WebAssembly engine (`wasm/engine.js` + `wasm/engine.wasm`) whenever
 * the runtime can load it, falling back automatically and silently to
 * the pure-JS engine (`neural.js`) otherwise - no WebAssembly at all,
 * no SIMD128 (engine.wasm fails to compile), a fetch error, or any
 * other instantiation failure. Both engines produce bit-identical
 * payloads and decodes (proven by wasm/wasm.test.mjs and the full
 * sweeps in wasm/verify.mjs / wasm/verify-browser.mjs), so this module
 * only ever makes a *performance* choice - it never changes what a
 * link decodes to.
 *
 * Feature detection is done by actually compiling `engine.wasm`
 * rather than a synthetic capability probe: a runtime missing the
 * SIMD128 proposal fails WebAssembly.compile() on the real module's
 * v128 opcodes with a CompileError, which is caught below exactly
 * like any other instantiation failure. This avoids maintaining a
 * second, hand-rolled feature test that could disagree with the real
 * module.
 *
 * Selection happens once per LOADED MODEL, not once per process:
 * `WasmEngine` binds one set of dequantized weights into one WASM
 * instance, so call `selectEngine` again whenever a different model
 * file is loaded (the latest model at startup, or an archived v1/v2
 * model lazy-loaded to decode an old link - engine.c implements both
 * kernel families, so archived versions get the WASM path too; see
 * wasm/README.md for which kernel each version uses).
 */

import { neuralCompressToNumber, neuralDecompressNumber } from "./neural.js";
import {
  WasmEngine,
  wasmCompressToNumber,
  wasmDecompressNumber
} from "./wasm/engine.js";

/**
 * Wraps a WebAssembly binary source (an ArrayBuffer/typed array, a
 * `fetch()` Response, or a promise of either) into a memoized async
 * getter for the compiled `WebAssembly.Module`. Call this ONCE per
 * process/page load with a source that has already started loading
 * (e.g. a `fetch()` call made immediately at startup, in parallel
 * with the model fetch) and pass the returned function to every
 * `selectEngine` call - the module compiles once and every
 * subsequently loaded model (including lazy-loaded archived
 * versions) reuses it without re-fetching or re-compiling.
 * @param {BufferSource|Response|Promise<BufferSource|Response>} source
 *  The engine.wasm bytes, however they're being fetched
 * @returns {() => Promise<WebAssembly.Module>} Memoized module getter
 */
export function wasmModuleSource (source) {
  let compiled = null;
  return () => {
    if (!compiled) {
      compiled = (async () => {
        let bytes = await source;
        if (bytes && typeof bytes.arrayBuffer === "function") {
          bytes = await bytes.arrayBuffer();
        }
        return WebAssembly.compile(bytes);
      })();
    }
    return compiled;
  };
}

/**
 * Records which engine got selected somewhere a developer can find it
 * without adding console noise: `globalThis.__hamrEngine` (readable
 * from a browser devtools console or a Node REPL/debugger).
 * @param {{kind: string, fallbackReason?: string}} info
 */
function recordSelection (info) {
  try {
    globalThis.__hamrEngine = { kind: info.kind, fallbackReason: info.fallbackReason };
  } catch {
    // Environments without a writable globalThis (unlikely) just skip
    // the debug flag; selection itself is unaffected.
  }
}

/**
 * Binds one loaded model to the best available inference engine.
 * @param {URLModel} model Parsed model (see neural.js URLModel)
 * @param {() => Promise<WebAssembly.Module>} wasmSource From
 *  wasmModuleSource(); omit (or pass null) to always use the JS engine
 * @param {{force?: "js"}} [options] `force: "js"` skips the WASM
 *  attempt entirely - used by tests and the CLI's HAMR_FORCE_ENGINE
 *  escape hatch to exercise the fallback path deterministically,
 *  without needing to actually break WebAssembly in the environment
 * @returns {Promise<{
 *   model: URLModel,
 *   kind: "wasm"|"js",
 *   compress: (input: string, options?: object) => BigInt?,
 *   decompress: (number: BigInt) => string,
 *   wasmEngine?: WasmEngine,
 *   fallbackReason?: string
 * }>} The selected engine, already bound to `model`. `compress` and
 *  `decompress` match neuralCompressToNumber/neuralDecompressNumber's
 *  signatures minus the model argument (already bound), so they drop
 *  straight into hybrid.js's optional `engine` parameter.
 */
export async function selectEngine (model, wasmSource, { force = null } = {}) {
  const jsEngine = (fallbackReason) => {
    const result = {
      model,
      kind: "js",
      compress: (input, options) => neuralCompressToNumber(model, input, options),
      decompress: (number) => neuralDecompressNumber(model, number)
    };
    if (fallbackReason) result.fallbackReason = fallbackReason;
    recordSelection(result);
    return result;
  };

  if (force === "js") return jsEngine("forced");
  if (typeof WebAssembly === "undefined") {
    return jsEngine("WebAssembly unavailable");
  }
  if (!wasmSource) return jsEngine("no wasm source provided");

  try {
    const module = await wasmSource();
    const wasm = await WasmEngine.create(model, module);
    const result = {
      model,
      kind: "wasm",
      wasmEngine: wasm,
      compress: (input, options) => wasmCompressToNumber(wasm, input, options),
      decompress: (number) => wasmDecompressNumber(wasm, number)
    };
    recordSelection(result);
    return result;
  } catch (e) {
    // No WebAssembly, no SIMD128 (engine.wasm fails to compile), a
    // fetch error propagated through wasmSource(), or any other
    // instantiation failure: fall back silently. This is the ONLY
    // catch site for WASM failures - once selection succeeds, a later
    // per-call error is a real bug, not a fallback trigger.
    return jsEngine(e && e.message ? e.message : String(e));
  }
}
