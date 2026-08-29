/**
 * Cross-environment inference benchmark: encode + decode timings for
 * a model file in Node AND headless Chromium, verifying that both
 * environments produce byte-identical payloads (the determinism the
 * whole scheme rests on).
 *
 * Usage: node bench-browser.mjs <model.bin> <urls-file> [limit]
 */
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve } from "node:path";
import { chromium } from "playwright";
import { compressHybrid, decompressHybrid } from "../../../hybrid.js";
import { URLModel } from "../../../neural.js";
import { outputAlphabetASCII } from "../../../alphabets.js";

const [modelFile, urlsFile] = process.argv.slice(2);
const limit = parseInt(process.argv[4] || "50", 10);
if (!modelFile || !urlsFile) {
  console.error("Usage: node bench-browser.mjs <model.bin> <urls-file> [limit]");
  process.exit(1);
}
const repoRoot = new URL("../../..", import.meta.url).pathname;

const urls = (await readFile(urlsFile, "utf8"))
  .split("\n").map(u => u.trim()).filter(Boolean).slice(0, limit)
  .map(u => (/^https?:\/\//.test(u) ? u : "https://" + u));

// The benchmark body runs identically in both environments
const benchSource = `
  async function bench (model, urls, { compressHybrid, decompressHybrid, alphabet }) {
    const payloads = [];
    // warmup
    for (const u of urls.slice(0, 5)) {
      decompressHybrid(compressHybrid(u, alphabet, model), alphabet, model);
    }
    let encMs = 0, decMs = 0;
    for (const u of urls) {
      const t0 = performance.now();
      const p = compressHybrid(u, alphabet, model);
      const t1 = performance.now();
      const back = decompressHybrid(p, alphabet, model);
      const t2 = performance.now();
      if (new URL(back).href !== new URL(u).href &&
          decodeURIComponent(new URL(back).href) !== decodeURIComponent(new URL(u).href)) {
        throw new Error("round-trip mismatch: " + u);
      }
      encMs += t1 - t0;
      decMs += t2 - t1;
      payloads.push(p);
    }
    return { payloads, encMs: encMs / urls.length, decMs: decMs / urls.length };
  }
`;

// --- Node run ---
const modelBuf = await readFile(resolve(modelFile));
const nodeModel = new URLModel(
  modelBuf.buffer.slice(modelBuf.byteOffset, modelBuf.byteOffset + modelBuf.byteLength));
const bench = new Function(`${benchSource}; return bench;`)();
const nodeRes = await bench(nodeModel, urls,
  { compressHybrid, decompressHybrid, alphabet: outputAlphabetASCII });
console.log(`node     : encode ${nodeRes.encMs.toFixed(1)} ms/URL, ` +
  `decode ${nodeRes.decMs.toFixed(1)} ms/URL (${urls.length} URLs)`);

// --- Chromium run (repo served over local HTTP so ES modules load) ---
const server = createServer(async (req, res) => {
  try {
    const path = resolve(repoRoot, "." + new URL(req.url, "http://x").pathname);
    if (!path.startsWith(repoRoot)) throw new Error("outside root");
    const body = await readFile(path);
    const mime = { ".js": "text/javascript", ".mjs": "text/javascript",
      ".bin": "application/octet-stream", ".html": "text/html" }[extname(path)]
      || "application/octet-stream";
    res.writeHead(200, { "content-type": mime });
    res.end(body);
  } catch {
    res.writeHead(404); res.end();
  }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined
});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/index.html`).catch(() => {});
const relModel = resolve(modelFile).slice(repoRoot.length);
const chromeRes = await page.evaluate(async ({ benchSource, urls, relModel }) => {
  const { compressHybrid, decompressHybrid } = await import("/hybrid.js");
  const { URLModel } = await import("/neural.js");
  const { outputAlphabetASCII } = await import("/alphabets.js");
  const buf = await (await fetch("/" + relModel)).arrayBuffer();
  const model = new URLModel(buf);
  const bench = new Function(`${benchSource}; return bench;`)();
  return await bench(model, urls,
    { compressHybrid, decompressHybrid, alphabet: outputAlphabetASCII });
}, { benchSource, urls, relModel });
await browser.close();
server.close();

console.log(`chromium : encode ${chromeRes.encMs.toFixed(1)} ms/URL, ` +
  `decode ${chromeRes.decMs.toFixed(1)} ms/URL`);

let mismatches = 0;
for (let i = 0; i < urls.length; i ++) {
  if (nodeRes.payloads[i] !== chromeRes.payloads[i]) {
    mismatches ++;
    console.error(`PAYLOAD MISMATCH for ${urls[i]}`);
  }
}
console.log(mismatches === 0
  ? `payload equality: ${urls.length}/${urls.length} identical across environments`
  : `payload equality: ${mismatches} MISMATCHES - DO NOT SHIP`);
process.exit(mismatches === 0 ? 0 : 1);
