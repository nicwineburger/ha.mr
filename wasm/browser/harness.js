/**
 * @file In-page harness for the browser acceptance and benchmark
 * runs. Loaded by harness.html from the repo root over HTTP (see
 * drive.mjs); exposes window.hamr with functions the Playwright
 * drivers call via page.evaluate. Payload numbers cross the protocol
 * boundary as strings (BigInt doesn't serialize).
 */

import {
  URLModel,
  neuralCompressToNumber,
  neuralDecompressNumber
} from "../../neural.js";
import {
  WasmEngine,
  wasmCompressToNumber,
  wasmDecompressNumber
} from "../engine.js";

const engines = {};

window.hamr = {
  /**
   * Fetches a model file and the wasm binary, builds both engines.
   * @param {string} name Engine key, e.g. "v3"
   * @param {string} modelPath Model file path relative to site root
   * @returns {Promise<{parseMs: number, instantiateMs: number}>}
   */
  async setup (name, modelPath) {
    const [modelBytes, wasmBytes] = await Promise.all([
      fetch(modelPath).then(r => r.arrayBuffer()),
      fetch("/wasm/engine.wasm").then(r => r.arrayBuffer())
    ]);
    let t0 = performance.now();
    const model = new URLModel(modelBytes);
    const parseMs = performance.now() - t0;
    t0 = performance.now();
    const engine = await WasmEngine.create(model, wasmBytes);
    const instantiateMs = performance.now() - t0;
    engines[name] = { model, engine };
    return { parseMs, instantiateMs };
  },

  /**
   * Bit-identity check over links: both engines, search on and off,
   * plus decode agreement. Returns human-readable failure strings.
   * @param {string} name Engine key from setup()
   * @param {string[]} links URLs to verify
   * @returns {string[]} Failures (empty = all identical)
   */
  verify (name, links) {
    const { model, engine } = engines[name];
    const failures = [];
    for (const link of links) {
      for (const search of [false, true]) {
        const js = neuralCompressToNumber(model, link, { search });
        const wasm = wasmCompressToNumber(engine, link, { search });
        if (js !== wasm) {
          failures.push(`ENCODE [${name} search=${search}] ${link}: `
            + `${js} != ${wasm}`);
        }
        if (js !== null) {
          const jsDecoded = neuralDecompressNumber(model, js);
          const wasmDecoded = wasmDecompressNumber(engine, js);
          if (jsDecoded !== wasmDecoded) {
            failures.push(`DECODE [${name}] ${link}: `
              + `${jsDecoded} != ${wasmDecoded}`);
          }
        }
      }
    }
    return failures;
  },

  /**
   * Times decode / encode-fast / encode-search per URL on both
   * engines. The first `warmup` links are run untimed.
   * @param {string} name Engine key from setup()
   * @param {string[]} links URLs to time
   * @param {number} warmup Leading links to exclude from samples
   * @returns {Object<string, number[]>} ms samples per operation
   */
  bench (name, links, warmup) {
    const { model, engine } = engines[name];
    const ops = {
      "js encode (search)": (link, n) => neuralCompressToNumber(model, link),
      "js encode (fast)": (link, n) =>
        neuralCompressToNumber(model, link, { search: false }),
      "js decode": (link, n) => neuralDecompressNumber(model, n),
      "wasm encode (search)": (link, n) => wasmCompressToNumber(engine, link),
      "wasm encode (fast)": (link, n) =>
        wasmCompressToNumber(engine, link, { search: false }),
      "wasm decode": (link, n) => wasmDecompressNumber(engine, n)
    };
    const samples = Object.fromEntries(Object.keys(ops).map(k => [k, []]));
    for (const [i, link] of links.entries()) {
      const n = neuralCompressToNumber(model, link, { search: false });
      for (const [op, run] of Object.entries(ops)) {
        const start = performance.now();
        run(link, n);
        if (i >= warmup) samples[op].push(performance.now() - start);
      }
    }
    return samples;
  }
};

window.hamrReady = true;
