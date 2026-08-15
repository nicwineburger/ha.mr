import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanLink } from "../clean.js";

test("strips utm_* parameters", () => {
  const { cleaned, removed } = cleanLink(
    "https://example.com/page?utm_source=news&utm_medium=email&utm_campaign=x");
  assert.equal(cleaned, "https://example.com/page");
  assert.deepEqual(removed, ["utm_source", "utm_medium", "utm_campaign"]);
});

test("preserves the order of surviving parameters", () => {
  const { cleaned, removed } = cleanLink(
    "https://example.com/p?b=2&utm_source=x&a=1&fbclid=abc&c=3");
  assert.equal(cleaned, "https://example.com/p?b=2&a=1&c=3");
  assert.deepEqual(removed, ["utm_source", "fbclid"]);
});

test("strips whole prefix families", () => {
  const { cleaned, removed } = cleanLink(
    "https://example.com/?_hsenc=e&_hsmi=m&mtm_kwd=k&pk_campaign=c&oly_enc_id=o&vero_id=v&matomo_campaign=m&keep=1");
  assert.equal(cleaned, "https://example.com/?keep=1");
  assert.deepEqual(removed, [
    "_hsenc", "_hsmi", "mtm_kwd", "pk_campaign",
    "oly_enc_id", "vero_id", "matomo_campaign"
  ]);
});

test("strips exact-match click identifiers", () => {
  const { cleaned, removed } = cleanLink(
    "https://example.com/x?gclid=1&msclkid=2&mc_eid=3&mkt_tok=4&srsltid=5");
  assert.equal(cleaned, "https://example.com/x");
  assert.deepEqual(removed, ["gclid", "msclkid", "mc_eid", "mkt_tok", "srsltid"]);
});

test("returns links without a query unchanged", () => {
  const link = "https://example.com/some/path#fragment";
  assert.deepEqual(cleanLink(link), { cleaned: link, removed: [] });
});

test("drops the ? when every parameter is removed", () => {
  const { cleaned } = cleanLink("https://example.com/page?utm_source=x&gclid=y");
  assert.equal(cleaned, "https://example.com/page");
});

test("leaves the fragment untouched", () => {
  const { cleaned } = cleanLink(
    "https://example.com/p?utm_source=x&a=1#section?utm_medium=not-a-query");
  assert.equal(cleaned, "https://example.com/p?a=1#section?utm_medium=not-a-query");
});

test("a ? inside the fragment does not start a query", () => {
  // Hash-routed apps put "?" after "#"; that is part of the fragment
  const link = "https://example.com/app#/route?utm_source=state";
  assert.deepEqual(cleanLink(link), { cleaned: link, removed: [] });
});

test("never strips ambiguous or functional parameters", () => {
  // "ref", "q", "id", "page", "spm", "source" can select content on
  // some sites, so they always survive
  const link = "https://example.com/s?ref=hn&q=terms&id=42&page=2&spm=a.b&source=rss";
  assert.deepEqual(cleanLink(link), { cleaned: link, removed: [] });
});

test("strips parameters that have no value", () => {
  const { cleaned, removed } = cleanLink("https://example.com/?utm_source&a=1");
  assert.equal(cleaned, "https://example.com/?a=1");
  assert.deepEqual(removed, ["utm_source"]);
});

test("strips repeated parameters and reports the name once", () => {
  const { cleaned, removed } = cleanLink(
    "https://example.com/?utm_source=a&x=1&utm_source=b");
  assert.equal(cleaned, "https://example.com/?x=1");
  assert.deepEqual(removed, ["utm_source"]);
});

test("preserves surviving parameters byte-exact", () => {
  // Empty segments, missing values, and unusual spellings all pass
  // through untouched
  const { cleaned } = cleanLink(
    "https://example.com/?&a&b=&utm_source=x&C=%2F");
  assert.equal(cleaned, "https://example.com/?&a&b=&C=%2F");
});
