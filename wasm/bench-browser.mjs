/**
 * @file Browser speed benchmark: the same measurement as bench.mjs,
 * run inside headless Chromium via Playwright.
 *
 * Usage: node wasm/bench-browser.mjs <urls-file> [count] [version]
 */

import { readFile } from "node:fs/promises";
import { URLModel } from "../neural.js";
import { withHarnessPage } from "./browser/drive.mjs";

const urlsFile = process.argv[2];
const count = Number(process.argv[3] || 30);
const version = Number(process.argv[4] || 3);
if (!urlsFile) {
  console.error("Usage: node wasm/bench-browser.mjs <urls-file> [count] [version]");
  process.exit(2);
}

const modelFile = version === 3 ? "url-model.bin" : `url-model-v${version}.bin`;
const model = new URLModel(
  (await readFile(new URL(`../model/${modelFile}`, import.meta.url))).buffer);

const urls = (await readFile(urlsFile, "utf8")).split("\n")
  .map(u => u.trim()).filter(Boolean);
const picked = [];
for (const link of urls) {
  if (picked.length === count + 3) break;
  if (!URL.canParse(link)) continue;
  const url = new URL(link);
  if (url.protocol !== "http:" && url.protocol !== "https:") continue;
  const text = url.href.slice(url.protocol === "https:" ? 8 : 7);
  const tokens = model.tokenize(text);
  if (tokens !== null && tokens.length + 2 <= model.maxLen) picked.push(link);
}

await withHarnessPage(async (page) => {
  const overhead = await page.evaluate(
    ([p]) => window.hamr.setup("bench", p), [`/model/${modelFile}`]);
  const samples = await page.evaluate(
    ([l]) => window.hamr.bench("bench", l, 3), [picked]);

  const quantile = (xs, q) => {
    const sorted = [...xs].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  };
  console.log(`model v${version}, ${picked.length - 3} URLs, headless Chromium`);
  console.log(`model parse: ${overhead.parseMs.toFixed(0)} ms; `
    + `wasm instantiate + weight copy-in: ${overhead.instantiateMs.toFixed(0)} ms`);
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
});
