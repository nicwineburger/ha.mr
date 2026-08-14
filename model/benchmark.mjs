/**
 * Compression benchmark: classic vs hybrid (classic + neural) payload
 * sizes over a file of URLs, one per line.
 *
 * Usage: node model/benchmark.mjs <urls-file> [limit]
 *
 * The URL file should be held-out data the model never saw during
 * training - see model/README.md for how the shipped model's holdout
 * set was produced.
 */
import { readFile } from "node:fs/promises";
import { compress } from "../compress.js";
import { compressHybrid, decompressHybrid } from "../hybrid.js";
import { URLModel } from "../neural.js";
import { outputAlphabetASCII } from "../alphabets.js";

const file = process.argv[2];
const limit = parseInt(process.argv[3] || "500", 10);
if (!file) {
  console.error("Usage: node model/benchmark.mjs <urls-file> [limit]");
  process.exit(1);
}

const model = new URLModel(
  (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);

const urls = (await readFile(file, "utf8"))
  .split("\n").map(u => u.trim()).filter(Boolean).slice(0, limit);

let classicTotal = 0;
let hybridTotal = 0;
let inputTotal = 0;
let neuralWins = 0;
let failures = 0;

const t0 = performance.now();
for (const raw of urls) {
  const link = /^https?:\/\//.test(raw) ? raw : "https://" + raw;
  let classic, hybrid;
  try {
    classic = compress(link, outputAlphabetASCII);
    hybrid = compressHybrid(link, outputAlphabetASCII, model);
    const back = decompressHybrid(hybrid, outputAlphabetASCII, model);
    if (new URL(back).href !== new URL(link).href) {
      // Classic-scheme payloads normalize escapes of unreserved
      // characters (documented behavior, pinned in the test suite)
      const decode = (u) => { try { return decodeURIComponent(u); } catch { return u; } };
      if (decode(new URL(back).href) !== decode(new URL(link).href)) {
        throw `round-trip mismatch: ${back}`;
      }
    }
  } catch (e) {
    failures ++;
    console.error(`FAIL ${link}: ${e}`);
    continue;
  }
  inputTotal += link.replace(/^https?:\/\//, "").length;
  classicTotal += classic.length;
  hybridTotal += hybrid.length;
  if (hybrid.length < classic.length) neuralWins ++;
}
const seconds = (performance.now() - t0) / 1000;

const n = urls.length - failures;
console.log(`URLs: ${n} (${failures} failures)`);
console.log(`Average input length (scheme stripped): ${(inputTotal / n).toFixed(1)}`);
console.log(`Average classic payload: ${(classicTotal / n).toFixed(1)} symbols`);
console.log(`Average hybrid payload:  ${(hybridTotal / n).toFixed(1)} symbols`);
console.log(`Neural chosen for ${neuralWins}/${n} URLs (${(100 * neuralWins / n).toFixed(1)}%)`);
console.log(`Hybrid payloads are ${(100 * (1 - hybridTotal / classicTotal)).toFixed(1)}% smaller than classic overall`);
console.log(`Throughput: ${(n / seconds).toFixed(1)} URLs/s (encode + decode + verify)`);
