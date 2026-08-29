import { test } from "node:test";
import assert from "node:assert/strict";
import { compressHybrid, decompressHybrid } from "../docs/hybrid.js";
import { outputAlphabetASCII } from "../docs/alphabets.js";

// These tests run classic-only (model = null): they cover hybrid.js's
// own input handling, which is independent of the neural scheme.

function roundtrip (link) {
  const payload = compressHybrid(link, outputAlphabetASCII, null);
  return decompressHybrid(payload, outputAlphabetASCII, null);
}

test("links the encoding can't represent are rejected, not mangled", () => {
  const rejected = [
    // Non-http(s) protocols used to be silently rewritten to http://
    "ftp://example.com/file",
    "mailto:someone@example.com",
    "javascript:alert(1)",
    // Credentials used to be silently dropped - and doubling as a
    // phishing pattern ("https://google.com@evil.example") makes
    // accepting them actively harmful
    "https://user:pass@example.com/",
    "https://google.com@evil.example/",
    // A bare protocol is not a link (it used to encode as http://http/)
    "http://",
    ""
  ];
  for (const link of rejected) {
    assert.throws(() => compressHybrid(link, outputAlphabetASCII, null),
      undefined, `should reject ${JSON.stringify(link)}`);
  }
});

test("links without a protocol are assumed to be http", () => {
  assert.equal(roundtrip("example.com"), "http://example.com");
  assert.equal(roundtrip("example.com/a/b?q=1"), "http://example.com/a/b?q=1");
});

test("intentional classic normalizations pass round-trip verification", () => {
  // compressHybrid verifies that the classic payload decodes back to
  // the input link; these differ as strings but only by the scheme's
  // pinned normalizations, so they must not be rejected
  const cases = [
    ["https://example.com/%7Euser", "https://example.com/~user"],
    ["https://example.com/a%2fb", "https://example.com/a%2Fb"],
    ["https://example.com/?foo", "https://example.com?foo="],
    ["https://example.com/?", "https://example.com"],
    ["https://example.com/50%_off", "https://example.com/50%25_off"],
    ["https://example.com/a b", "https://example.com/a%20b"],
    ["https://EXAMPLE.COM/Path", "https://example.com/Path"]
  ];
  for (const [input, expected] of cases) {
    assert.equal(roundtrip(input), expected, `hybrid round-trip of ${input}`);
  }
});

test("ordinary links still round-trip through the hybrid path", () => {
  const links = [
    "https://example.com",
    "https://www.example.com/some/path?a=1&b=2#frag",
    "https://en.wikipedia.org/wiki/Hammer_(disambiguation)",
    "https://example.com/app#/x/y?z=1",
    "https://example.com/x?redirect=https%3A%2F%2Fother.example%2Fpath"
  ];
  for (const link of links) {
    assert.equal(roundtrip(link), link, `round-trip of ${link}`);
  }
});
