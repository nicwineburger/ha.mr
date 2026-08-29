import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { URLModel, neuralCompressToNumber } from "../neural.js";
import { compressHybrid } from "../hybrid.js";
import { outputAlphabetASCII } from "../alphabets.js";

/*
 * Production browser acceptance: drives the REAL site (index.html +
 * main.js), not the wasm/browser measurement harness, in headless
 * Chromium via Playwright, and checks that:
 *   1. it actually selects the WASM engine (via the debug-readable
 *      globalThis.__hamrEngine flag engine-select.js sets), and
 *   2. the searched-stage payload it renders for a handful of links
 *      matches what Node's compressHybrid (JS engine) computes for
 *      the exact same links - proving the production page's WASM
 *      path is bit-identical through the real UI, not just through
 *      the wasm/ test harness.
 *
 * Playwright + a headless Chromium binary are dev-time-only tools not
 * guaranteed present everywhere `npm test` runs (see wasm/README.md);
 * this file skips itself with a clear reason instead of failing when
 * they're unavailable, exactly like verify-browser.mjs/bench-browser.mjs
 * already assume for manual runs.
 */

const CHROMIUM_PATH = "/opt/pw-browsers/chromium";
const PLAYWRIGHT_PATH = process.env.PLAYWRIGHT
  || "/opt/node22/lib/node_modules/playwright/index.mjs";

// node:test's `skip` option treats even a falsy null as "skip" for
// reporting purposes (while still running the body) - only `undefined`
// (or omitting the key) means "run normally", so that's the not-skipped
// sentinel here, not null.
let unavailableReason = undefined;
let withHarnessPage = null;
if (!existsSync(CHROMIUM_PATH)) {
  unavailableReason = `no Chromium binary at ${CHROMIUM_PATH}`;
} else {
  try {
    ({ withHarnessPage } = await import("./browser/drive.mjs"));
    await import(PLAYWRIGHT_PATH);
  } catch (e) {
    unavailableReason = `Playwright unavailable: ${e && e.message ? e.message : e}`;
  }
}

test("production page selects wasm and matches Node payloads",
  { skip: unavailableReason }, async () => {
    const links = [
      "https://www.example.com/some/path?a=1&b=2",
      "https://en.wikipedia.org/wiki/Hammer",
      "https://github.com/user/repo/blob/main/README.md"
    ];

    const model = new URLModel(
      (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);
    const expected = links.map(link =>
      compressHybrid(link, outputAlphabetASCII, model));
    // Independent cross-check: the classic-only path never wins for
    // these links, so the neural payload really is under test
    for (const [i, link] of links.entries()) {
      assert.ok(neuralCompressToNumber(model, link) !== null,
        `expected ${link} to be neural-codable`);
      assert.equal(expected[i],
        compressHybrid(link, outputAlphabetASCII, model, { search: true }));
    }

    await withHarnessPage(async (page) => {
      // Model parse (12.3MB) + wasm instantiate happen once on load
      await page.waitForFunction(
        "typeof globalThis.__hamrEngine !== 'undefined'", null, { timeout: 30000 });
      const kind = await page.evaluate(() => globalThis.__hamrEngine.kind);
      assert.equal(kind, "wasm",
        "headless Chromium is expected to support WebAssembly SIMD128");

      for (const [i, link] of links.entries()) {
        await page.fill("#input-link", "");
        await page.fill("#input-link", link);
        // Wait past the 800ms searched-stage timer, plus generous
        // margin for the searched encode itself (REPORT.md measures
        // ~2.6s median WASM search-encode in Chromium for the shipped
        // model; short pinned links are faster, but the margin is
        // kept wide so this test isn't timing-flaky)
        await page.waitForTimeout(6000);
        const text = await page.textContent("#output-link");
        const hashIndex = text.indexOf("#");
        assert.ok(hashIndex !== -1, `no payload rendered for ${link}: "${text}"`);
        const payload = text.slice(hashIndex + 1);
        assert.equal(payload, expected[i],
          `browser payload for ${link} should match Node's JS-engine payload`);
      }
    }, "/");
  });
