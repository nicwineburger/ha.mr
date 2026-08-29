/**
 * @file Full bit-identity acceptance sweep (Node): every pinned-vector
 * link for every supported model version, plus N held-out URLs, must
 * produce byte-identical payloads (search on AND off) and identical
 * decodes on the JS and WASM engines. Exits non-zero on any mismatch.
 *
 * Usage: node wasm/verify.mjs <urls-file> [count]
 *   urls-file  one URL per line (held-out corpus)
 *   count      how many corpus URLs to check (default 100)
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
const count = Number(process.argv[3] || 100);
if (!urlsFile) {
  console.error("Usage: node wasm/verify.mjs <urls-file> [count]");
  process.exit(2);
}

const wasmBinary = await readFile(new URL("./engine.wasm", import.meta.url));
const load = async (file) => {
  const model = new URLModel(
    (await readFile(new URL(`../model/${file}`, import.meta.url))).buffer);
  return { model, engine: await WasmEngine.create(model, wasmBinary) };
};
const engines = {
  v1: await load("url-model-v1.bin"),
  v2: await load("url-model-v2.bin"),
  v3: await load("url-model.bin")
};

const pinnedLinks = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://github.com/user/repo/blob/main/README.md",
  "https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator",
  "https://illashbyilly.com.au/collections/eye-lashes",
  "https://www.dyson.com.ro/asistenta-dyson/contactati-ne",
  "https://data.example-archive.org/collections/2024/expedition-photos/"
    + "region-north-atlantic/vessel-research-7/deck-camera-03/"
    + "capture-2024-06-19T14-22-51Z-frame-000482-exposure-auto-wb-daylight.jpg"
    + "?checksum=9f3a1c77d2e648b0&signature=vRt2LpQ8xYw4Nc6bJmH0aZsEuDkFgO51"
    + "&expires=1718900000&session=b81f2ce4a90d47e3"
];

let checks = 0;
let failures = 0;

function check (name, { model, engine }, link) {
  for (const search of [false, true]) {
    const js = neuralCompressToNumber(model, link, { search });
    const wasm = wasmCompressToNumber(engine, link, { search });
    checks ++;
    if (js !== wasm) {
      failures ++;
      console.error(`ENCODE MISMATCH [${name} search=${search}] ${link}`
        + `\n  js:   ${js}\n  wasm: ${wasm}`);
    }
    if (js !== null) {
      const jsDecoded = neuralDecompressNumber(model, js);
      const wasmDecoded = wasmDecompressNumber(engine, js);
      checks ++;
      if (jsDecoded !== wasmDecoded) {
        failures ++;
        console.error(`DECODE MISMATCH [${name}] ${link}`
          + `\n  js:   ${jsDecoded}\n  wasm: ${wasmDecoded}`);
      }
    }
  }
}

console.log("Pinned-vector links, all supported versions...");
for (const [name, pair] of Object.entries(engines)) {
  for (const link of pinnedLinks) {
    // v1/v2 refuse the beyond-context link; verify both refuse
    if (pair.model.tokenize(new URL(link).href.slice(8)).length + 2 >
        pair.model.maxLen && pair.model.linkVersion < 3) {
      const js = neuralCompressToNumber(pair.model, link);
      const wasm = wasmCompressToNumber(pair.engine, link);
      checks ++;
      if (js !== null || wasm !== null) {
        failures ++;
        console.error(`REFUSAL MISMATCH [${name}] ${link}: ${js} ${wasm}`);
      }
      continue;
    }
    check(name, pair, link);
  }
  console.log(`  ${name} done (${checks} checks, ${failures} failures)`);
}

console.log(`Held-out URLs (${count}) against v3 and v2...`);
const urls = (await readFile(urlsFile, "utf8")).split("\n")
  .map(u => u.trim()).filter(Boolean);
// Deterministic pick: every k-th URL that the v3 model can tokenize
const picked = [];
for (let i = 0; i < urls.length && picked.length < count; i ++) {
  const url = new URL(urls[i]);
  const text = url.href.slice(url.protocol === "https:" ? 8 : 7);
  if (engines.v3.model.tokenize(text) !== null) picked.push(urls[i]);
}

for (const [i, link] of picked.entries()) {
  check("v3", engines.v3, link);
  check("v2", engines.v2, link);
  if ((i + 1) % 10 === 0) {
    console.log(`  ${i + 1}/${picked.length} (${checks} checks, `
      + `${failures} failures)`);
  }
}

console.log(`${checks} checks, ${failures} failures`);
process.exit(failures ? 1 : 0);
