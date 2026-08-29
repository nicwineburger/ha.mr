/**
 * @file Weight-download measurement: how far can transparent
 * compression shrink a model file? Reports gzip -9 and brotli -q 11
 * baselines (what a web server / CDN could do with zero client code),
 * the wpack entropy-coded container (codec.mjs), decode-at-load time,
 * and a byte-identity check of the round-trip. Also prints the
 * order-0/order-1 empirical entropies of the quantized weights - the
 * bound any order-1 scheme can reach.
 *
 * Usage: node wasm/weights/measure.mjs [model-file]
 */

import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { gzipSync, gunzipSync, brotliCompressSync, constants } from "node:zlib";
import { pack, unpack, buildStreams } from "./codec.mjs";

const file = new Uint8Array(await readFile(
  process.argv[2] || new URL("../../model/url-model.bin", import.meta.url)));
const mb = (n) => (n / 1024 / 1024).toFixed(2) + " MB";
const pct = (n) => ((1 - n / file.length) * 100).toFixed(1) + "%";
console.log(`original: ${mb(file.length)}`);

let t0 = performance.now();
const gz = gzipSync(file, { level: 9 });
console.log(`gzip -9:      ${mb(gz.length)}  (-${pct(gz.length)})  `
  + `[${(performance.now() - t0).toFixed(0)} ms compress]`);

t0 = performance.now();
const br = brotliCompressSync(file, {
  params: { [constants.BROTLI_PARAM_QUALITY]: 11 } });
console.log(`brotli -q11:  ${mb(br.length)}  (-${pct(br.length)})  `
  + `[${(performance.now() - t0).toFixed(0)} ms compress]`);

// Empirical entropies of the quantized weight stream (compression
// bound for order-0/order-1 models over the same alphabet)
const streams = buildStreams(file);
function entropy (symbols, contexts, alphabet) {
  const counts = [];
  for (let c = 0; c < contexts; c ++) counts.push(new Uint32Array(alphabet));
  let prev = 0;
  for (let i = 0; i < symbols.length; i ++) {
    counts[contexts === 1 ? 0 : (i === 0 ? 0 : prev)][symbols[i]] ++;
    prev = symbols[i];
  }
  let bits = 0;
  for (const table of counts) {
    let total = 0;
    for (const c of table) total += c;
    for (const c of table) {
      if (c) bits += c * Math.log2(total / c);
    }
  }
  return bits / symbols.length;
}
if (streams.nibbles.length) {
  console.log(`int4 nibbles: ${streams.nibbles.length.toLocaleString()} symbols, `
    + `order-0 entropy ${entropy(streams.nibbles, 1, 16).toFixed(3)} bits, `
    + `order-1 ${entropy(streams.nibbles, 16, 16).toFixed(3)} bits (of 4)`);
}
console.log(`f16 hi bytes: order-0 entropy `
  + `${entropy(streams.hi, 1, 256).toFixed(3)} bits, lo bytes `
  + `${entropy(streams.lo, 1, 256).toFixed(3)} bits (of 8)`);
if (streams.int8.length) {
  console.log(`int8 bytes: order-0 entropy `
    + `${entropy(streams.int8, 1, 256).toFixed(3)} bits (of 8)`);
}

t0 = performance.now();
const { container, breakdown } = pack(file, (b) => gzipSync(b, { level: 9 }));
console.log(`wpack:        ${mb(container.length)}  (-${pct(container.length)})  `
  + `[${(performance.now() - t0).toFixed(0)} ms pack]`);
console.log("  breakdown:", JSON.stringify(breakdown));

t0 = performance.now();
const restored = unpack(container, (b) => gunzipSync(b));
const decodeMs = performance.now() - t0;
const sha = (b) => createHash("sha256").update(b).digest("hex");
const identical = sha(restored) === sha(file);
console.log(`unpack: ${decodeMs.toFixed(0)} ms, byte-identical: ${identical}`);
if (!identical) process.exit(1);

// What the container gains over letting the CDN brotli the raw file
console.log(`wpack+nothing vs raw brotli: ${mb(container.length)} vs ${mb(br.length)}`);
console.log(`brotli of wpack container:   ${mb(brotliCompressSync(container, {
  params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length)}`);
