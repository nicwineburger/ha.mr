import { test } from "node:test";
import assert from "node:assert/strict";
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

/*
 * Bit-identity acceptance tests: the WASM engine must produce
 * byte-identical payloads (search on AND off) and identical decodes
 * for every model version it supports. Any mismatch is a bug in the
 * WASM transcription - never in neural.js, whose behavior is frozen
 * by issued links. This file keeps npm test's added runtime modest by
 * searching only a subset of v3 links; the full sweep (all pinned
 * vectors plus 100 held-out URLs, Node and Chromium) is
 * wasm/verify.mjs / wasm/verify-browser.mjs - results in
 * wasm/REPORT.md.
 */

const wasmBinary = await readFile(new URL("./engine.wasm", import.meta.url));
const load = async (file) => {
  const model = new URLModel(
    (await readFile(new URL(`../model/${file}`, import.meta.url))).buffer);
  return { model, engine: await WasmEngine.create(model, wasmBinary) };
};
const v1 = await load("url-model-v1.bin");
const v2 = await load("url-model-v2.bin");
const v3 = await load("url-model.bin");

// The links behind test/neural.test.mjs's pinned vectors
const pinnedLinks = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://github.com/user/repo/blob/main/README.md",
  "https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator"
];

const longPinnedLink =
  "https://data.example-archive.org/collections/2024/expedition-photos/"
  + "region-north-atlantic/vessel-research-7/deck-camera-03/"
  + "capture-2024-06-19T14-22-51Z-frame-000482-exposure-auto-wb-daylight.jpg"
  + "?checksum=9f3a1c77d2e648b0&signature=vRt2LpQ8xYw4Nc6bJmH0aZsEuDkFgO51"
  + "&expires=1718900000&session=b81f2ce4a90d47e3";

// Held-out-style URLs not among the pinned links
const extraLinks = [
  "https://www.petsfollower.com/can-dogs-eat-potted-meat/",
  "http://www.roadlinkxpress.com/certifications/",
  "https://cellapplications.com/rabbit-cell-media",
  "https://shop.example.co.uk/products/blue-widget?variant=42&ref=newsletter",
  "http://forum.example.org/viewtopic.php?f=12&t=98765&start=25",
  "https://docs.example.io/en/latest/api/reference.html#section-3.2"
];

/** Asserts both engines agree on a link, in both encode modes. */
function assertIdentical ({ model, engine }, link, { search = true } = {}) {
  const js = neuralCompressToNumber(model, link, { search });
  const wasm = wasmCompressToNumber(engine, link, { search });
  assert.equal(wasm, js, `payload mismatch (search: ${search}) for ${link}`);
  if (js === null) return;
  assert.equal(wasmDecompressNumber(engine, js),
    neuralDecompressNumber(model, js), `decode mismatch for ${link}`);
}

test("wasm engine is bit-identical on v1 pinned links", () => {
  for (const link of pinnedLinks) {
    assertIdentical(v1, link, { search: false });
    assertIdentical(v1, link, { search: true });
  }
});

test("wasm engine is bit-identical on v2 pinned links", () => {
  for (const link of pinnedLinks) {
    assertIdentical(v2, link, { search: false });
    assertIdentical(v2, link, { search: true });
  }
});

test("wasm engine is bit-identical on v3 pinned links", () => {
  for (const link of pinnedLinks) assertIdentical(v3, link, { search: false });
  // The searched encoder exercises session forking; two links keep
  // the JS-side cost of this test reasonable (verify.mjs runs all)
  for (const link of pinnedLinks.slice(0, 2)) assertIdentical(v3, link);
});

test("wasm engine is bit-identical on chunked coding", () => {
  // Beyond-context URL: multiple context restarts in one stream
  assertIdentical(v3, longPinnedLink);
});

test("wasm engine is bit-identical on held-out URLs", () => {
  for (const link of extraLinks) assertIdentical(v3, link, { search: false });
  assertIdentical(v3, extraLinks[0]);
  for (const link of extraLinks.slice(0, 3)) assertIdentical(v2, link);
});

test("wasm engine refuses what the js engine refuses", () => {
  assert.equal(wasmCompressToNumber(v3.engine, "ftp://example.com/x"), null);
  // Out-of-vocabulary bytes survive URL normalization only in paths
  assert.equal(wasmCompressToNumber(v3.engine, "https://example.com/§x"),
    neuralCompressToNumber(v3.model, "https://example.com/§x"));
  // Beyond-context URLs: v2 refuses, v3 chunks
  assert.equal(wasmCompressToNumber(v2.engine, longPinnedLink), null);
});

test("wasm sessions are reusable after reset", () => {
  // Two encodes back to back on one engine (arena reset in between)
  // must agree with themselves and with the JS engine
  const first = wasmCompressToNumber(v3.engine, pinnedLinks[1],
    { search: false });
  const second = wasmCompressToNumber(v3.engine, pinnedLinks[1],
    { search: false });
  assert.equal(first, second);
  assert.equal(first,
    neuralCompressToNumber(v3.model, pinnedLinks[1], { search: false }));
});
