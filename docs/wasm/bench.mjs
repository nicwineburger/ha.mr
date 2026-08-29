/**
 * @file Node speed benchmark: JS engine vs WASM engine, per-URL
 * timings for decode, encode fast path (search: false), and encode
 * with search. Prints median and p90 ms/URL per operation, plus the
 * WASM engine's one-time instantiation/copy-in overhead.
 *
 * Usage: node wasm/bench.mjs <urls-file> [count] [version]
 *   urls-file  one URL per line
 *   count      URLs to time (default 30; 3 more are used as warmup)
 *   version    model to bench: 3 (default) or 2
 */

import { readFile } from "node:fs/promises";
import {
  URLModel,
  neuralCompressToNumber,
  neuralDecompressNumber
} from "../neural.js";
import {
  WasmEngine,
  wasmCompressToNumber,
  wasmDecompressNumber
} from "./engine.js";

const urlsFile = process.argv[2];
const count = Number(process.argv[3] || 30);
const version = Number(process.argv[4] || 3);
if (!urlsFile) {
  console.error("Usage: node wasm/bench.mjs <urls-file> [count] [version]");
  process.exit(2);
}

const modelFile = version === 3 ? "url-model.bin" : `url-model-v${version}.bin`;
const raw = (await readFile(new URL(`../model/${modelFile}`,
  import.meta.url))).buffer;

let t0 = performance.now();
const model = new URLModel(raw);
const parseMs = performance.now() - t0;

const wasmBinary = await readFile(new URL("./engine.wasm", import.meta.url));
t0 = performance.now();
const engine = await WasmEngine.create(model, wasmBinary);
const instantiateMs = performance.now() - t0;

/**
 * Picks URLs the model can code within one context window, so the
 * search path is active for every timed URL and the three operations
 * are measured on identical inputs.
 */
function pick (urls, n) {
  const picked = [];
  for (const link of urls) {
    if (picked.length === n) break;
    if (!URL.canParse(link)) continue;
    const url = new URL(link);
    if (url.protocol !== "http:" && url.protocol !== "https:") continue;
    const text = url.href.slice(url.protocol === "https:" ? 8 : 7);
    const tokens = model.tokenize(text);
    if (tokens !== null && tokens.length + 2 <= model.maxLen) picked.push(link);
  }
  return picked;
}

const urls = (await readFile(urlsFile, "utf8")).split("\n")
  .map(u => u.trim()).filter(Boolean);
const picked = pick(urls, count + 3);
const warmup = picked.slice(0, 3);
const timed = picked.slice(3);

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

for (const link of warmup) {
  const n = neuralCompressToNumber(model, link, { search: false });
  for (const op of Object.values(ops)) op(link, n);
}

const samples = Object.fromEntries(Object.keys(ops).map(k => [k, []]));
for (const link of timed) {
  const n = neuralCompressToNumber(model, link, { search: false });
  for (const [name, op] of Object.entries(ops)) {
    const start = performance.now();
    op(link, n);
    samples[name].push(performance.now() - start);
  }
}

const quantile = (xs, q) => {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1,
    Math.floor(q * sorted.length))];
};

console.log(`model v${version} (${model.dim}d x ${model.layers}L), `
  + `${timed.length} URLs, Node ${process.version}`);
console.log(`model parse: ${parseMs.toFixed(0)} ms; `
  + `wasm instantiate + weight copy-in: ${instantiateMs.toFixed(0)} ms`);
console.log("operation                | median ms | p90 ms");
for (const [name, xs] of Object.entries(samples)) {
  console.log(`${name.padEnd(24)} | ${quantile(xs, 0.5).toFixed(1).padStart(9)}`
    + ` | ${quantile(xs, 0.9).toFixed(1).padStart(6)}`);
}
for (const op of ["encode (search)", "encode (fast)", "decode"]) {
  const js = quantile(samples[`js ${op}`], 0.5);
  const wasm = quantile(samples[`wasm ${op}`], 0.5);
  console.log(`speedup ${op}: ${(js / wasm).toFixed(2)}x (median)`);
}
