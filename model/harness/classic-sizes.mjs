// Computes classic-scheme payload sizes for a URL list -> JSON summary
// Usage: node classic-sizes.mjs <urls-file> <out.json>
import { readFile, writeFile } from "node:fs/promises";
import { compress } from "../../compress.js";
import { outputAlphabetASCII } from "../../alphabets.js";

const [file, out] = process.argv.slice(2);
const urls = (await readFile(file, "utf8")).split("\n").map(u => u.trim()).filter(Boolean);
const sizes = {};
let total = 0, n = 0, failures = 0;
for (const u of urls) {
  try {
    const len = compress(u, outputAlphabetASCII).length;
    sizes[u] = len;
    total += len; n ++;
  } catch { failures ++; }
}
await writeFile(out, JSON.stringify({ avg: total / n, n, failures, sizes }));
console.log(`classic avg: ${(total / n).toFixed(2)} symbols over ${n} URLs (${failures} failures)`);
