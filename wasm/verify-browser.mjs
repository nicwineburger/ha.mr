/**
 * @file Browser bit-identity acceptance sweep: the same checks as
 * verify.mjs, run inside headless Chromium via Playwright. Exits
 * non-zero on any mismatch.
 *
 * Usage: node wasm/verify-browser.mjs <urls-file> [count]
 */

import { readFile } from "node:fs/promises";
import { URLModel } from "../neural.js";
import { withHarnessPage } from "./browser/drive.mjs";

const urlsFile = process.argv[2];
const count = Number(process.argv[3] || 100);
if (!urlsFile) {
  console.error("Usage: node wasm/verify-browser.mjs <urls-file> [count]");
  process.exit(2);
}

const pinnedLinks = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://github.com/user/repo/blob/main/README.md",
  "https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator",
  "https://illashbyilly.com.au/collections/eye-lashes",
  "https://www.dyson.com.ro/asistenta-dyson/contactati-ne"
];
const longPinnedLink =
  "https://data.example-archive.org/collections/2024/expedition-photos/"
  + "region-north-atlantic/vessel-research-7/deck-camera-03/"
  + "capture-2024-06-19T14-22-51Z-frame-000482-exposure-auto-wb-daylight.jpg"
  + "?checksum=9f3a1c77d2e648b0&signature=vRt2LpQ8xYw4Nc6bJmH0aZsEuDkFgO51"
  + "&expires=1718900000&session=b81f2ce4a90d47e3";

// The same deterministic corpus pick as verify.mjs
const model = new URLModel(
  (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);
const urls = (await readFile(urlsFile, "utf8")).split("\n")
  .map(u => u.trim()).filter(Boolean);
const picked = [];
for (let i = 0; i < urls.length && picked.length < count; i ++) {
  if (!URL.canParse(urls[i])) continue;
  const url = new URL(urls[i]);
  const text = url.href.slice(url.protocol === "https:" ? 8 : 7);
  if (model.tokenize(text) !== null) picked.push(urls[i]);
}

let failures = [];
await withHarnessPage(async (page) => {
  for (const [name, path] of [["v1", "/model/url-model-v1.bin"],
      ["v2", "/model/url-model-v2.bin"], ["v3", "/model/url-model.bin"]]) {
    await page.evaluate(
      ([n, p]) => window.hamr.setup(n, p), [name, path]);
  }
  console.log("Pinned-vector links...");
  for (const name of ["v1", "v2", "v3"]) {
    const links = name === "v3" ? [...pinnedLinks, longPinnedLink] : pinnedLinks;
    failures = failures.concat(await page.evaluate(
      ([n, l]) => window.hamr.verify(n, l), [name, links]));
    console.log(`  ${name} done, ${failures.length} failures`);
  }
  console.log(`Held-out URLs (${picked.length}) against v3 and v2...`);
  for (let i = 0; i < picked.length; i += 10) {
    const batch = picked.slice(i, i + 10);
    for (const name of ["v3", "v2"]) {
      failures = failures.concat(await page.evaluate(
        ([n, l]) => window.hamr.verify(n, l), [name, batch]));
    }
    console.log(`  ${Math.min(i + 10, picked.length)}/${picked.length}, `
      + `${failures.length} failures`);
  }
});

for (const failure of failures) console.error(failure);
console.log(`${failures.length} failures`);
process.exit(failures.length ? 1 : 0);
