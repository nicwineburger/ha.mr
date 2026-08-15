import { test } from "node:test";
import assert from "node:assert/strict";
import {
  b64Decode,
  b64Encode,
  buildContent,
  escapeContent,
  parseText,
  pctDecode,
  pctEncode,
  reconstruct,
  serializeTree,
  stripBinary,
  treeSpanCounts,
  verifyInvertible
} from "./transforms.mjs";

/** Asserts that a text's transform tree replays back byte-exactly. */
function assertInvertible (text) {
  const { ok, hasTransforms } = verifyInvertible(text);
  assert.equal(ok, true, `inversion failed for: ${text}`);
  return hasTransforms;
}

test("percent decode/encode is byte-exact for mixed-case escapes", () => {
  const raw = "a%2fb%2Fc%3ad%3AE%e2%82%ac";
  const { decoded, escapes } = pctDecode(raw);
  assert.equal(decoded,
    "a/b/c:d:E" + String.fromCharCode(0xe2, 0x82, 0xac));
  assert.equal(pctEncode(decoded, escapes), raw);
});

test("percent decode leaves invalid escapes literal", () => {
  const raw = "100%25%zz%2";
  const { decoded, escapes } = pctDecode(raw);
  assert.equal(decoded, "100%%zz%2");
  assert.equal(pctEncode(decoded, escapes), raw);
});

test("nested URL in query value unwraps and inverts", () => {
  const text = "example.com/r?url=https%3A%2F%2Fnews.site%2Fa%2Fb" +
    "%3Fx%3D1%26y%3D2&other=1";
  assert.equal(assertInvertible(text), true);
  const parts = parseText(text);
  const { text: content } = buildContent(parts);
  assert.match(content, /url=https:\/\/news\.site\/a\/b\?x=1&y=2/);
});

test("double percent-encoding unwraps recursively", () => {
  const text = "b.org/l?u=https%253A%252F%252Fexample.com%252Fa%253Fb%253Dc";
  assert.equal(assertInvertible(text), true);
  const { text: content } = buildContent(parseText(text));
  assert.match(content, /https:\/\/example\.com\/a\?b=c/);
});

test("uppercase vs lowercase hex escapes both invert", () => {
  assertInvertible("x.com/?a=q%3Dw%26e%3Dr");
  assertInvertible("x.com/?a=q%3dw%26e%3dr");
  // Mixed canonical-defying pattern forces the mask variant
  assertInvertible("x.com/?a=q%3Dw%26e%3dr%2Fz");
});

test("base64 padding variants re-encode exactly", () => {
  for (const [text, bytes] of [
    ["aGVsbG8gd29ybGQh", "hello world!"], // No padding needed
    ["aGVsbG8gd29ybGQ=", "hello world"], // One pad char
    ["aGVsbG8gd29ybA==", "hello worl"], // Two pad chars
    ["aGVsbG8gd29ybA", "hello worl"] // Unpadded variant
  ]) {
    const dec = b64Decode(text);
    assert.ok(dec, `should decode: ${text}`);
    assert.equal(dec.bytes.toString("latin1"), bytes);
    assert.equal(b64Encode(dec.bytes, dec.urlSafe, dec.pad), text);
  }
});

test("non-canonical base64 (stray trailing bits) is rejected", () => {
  assert.equal(b64Decode("QR=="), null); // Decodes like QQ== leniently
  assert.equal(b64Decode("QUJDREVGR0g5MDF="), null); // Nonzero tail bits
  assert.equal(b64Decode("ab=cd"), null); // Inner padding
});

test("base64url alphabet round-trips through a URL", () => {
  const payload = Buffer.from('{"k":"a+b/c?d"}').toString("base64url");
  assert.match(payload, /[-_]/);
  const text = `site.io/x?d=${payload}&v=1`;
  assert.equal(assertInvertible(text), true);
  const counts = treeSpanCounts(parseText(text));
  assert.equal(counts.b64, 1);
});

test("base64 JSON query value unwraps despite '=' gluing", () => {
  const text = "x.com/p?data=eyJmb28iOiJiYXIiLCJ1cmwiOiJodHRwczovL2V4" +
    "YW1wbGUuY29tL3BhdGgifQ&v=2";
  assert.equal(assertInvertible(text), true);
  const { text: content } = buildContent(parseText(text));
  assert.match(content, /\{"foo":"bar","url":"https:\/\/example\.com\/path"\}/);
});

test("binary base64 goes to the binary channel, not the model", () => {
  const blob = Buffer.from(
    [137, 4, 250, 33, 7, 99, 200, 154, 18, 240, 61, 5, 171, 77, 30, 222]
  ).toString("base64");
  const stats = {};
  const text = `x.com/t?s=${blob}`;
  const parts = parseText(text, { stats });
  assert.equal(treeSpanCounts(parts).b64bin, 1);
  assert.equal(treeSpanCounts(parts).b64, undefined);
  assert.equal(stats.b64bin, 1);
  const { text: content, bytes } = buildContent(parts);
  assert.equal(bytes.length, 16); // Raw bytes, 8 bits each
  assert.ok(!content.includes(blob));
  assertInvertible(text);
  // The stripped variant reverts the span to a literal
  const stripped = stripBinary(parts);
  assert.equal(treeSpanCounts(stripped).b64bin, undefined);
  assert.equal(reconstruct(stripped), text);
});

test("JWT: JSON segments unwrap, signature goes binary", () => {
  const sig = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
  const text = "api.io/t/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." +
    "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0Ijox" +
    "NTE2MjM5MDIyfQ." + sig;
  assert.equal(assertInvertible(text), true);
  const parts = parseText(text);
  assert.equal(treeSpanCounts(parts).jwt, 3); // Header, payload, sig
  const { text: content, bytes } = buildContent(parts);
  assert.match(content, /\{"alg":"HS256","typ":"JWT"\}/);
  assert.match(content, /John%20Doe/); // Space re-escaped injectively
  assert.ok(!content.includes(sig)); // Signature: 32 raw bytes
  assert.equal(bytes.length, 32);
});

test("hex spans capture bytes and case, and strip back to literals", () => {
  for (const text of [
    "cdn.io/a/5f2b8c9d1e3a4b5c6d7e8f9a0b1c2d3e",
    "cdn.io/a/5F2B8C9D1E3A4B5C6D7E8F9A0B1C2D3E"
  ]) {
    assert.equal(assertInvertible(text), true);
    const parts = parseText(text);
    assert.equal(treeSpanCounts(parts).hex, 1);
    const stripped = stripBinary(parts);
    assert.equal(treeSpanCounts(stripped).hex, undefined);
    assert.equal(reconstruct(stripped), text);
  }
  // Mixed case is skipped rather than risking inexact inversion
  const mixed = parseText("cdn.io/a/5f2B8c9D1e3A4b5C6d7E8f9A0b1C2d3E");
  assert.equal(treeSpanCounts(mixed).hex, undefined);
});

test("content escaping is injective over all byte values", () => {
  const bytes = Array.from({ length: 256 }, (_, i) => i);
  const text = String.fromCharCode(...bytes);
  const escaped = escapeContent(text);
  assert.match(escaped, /^[\x21-\x7e]*$/); // Model-representable
  // Unescape: %XX (uppercase hex) back to bytes, literals pass through
  const unescaped = escaped.replace(/%[0-9A-F]{2}/g,
    m => String.fromCharCode(parseInt(m.slice(1), 16)));
  assert.equal(unescaped, text);
});

test("serialized tree is a bit array and mask variant costs more", () => {
  const canonical = parseText("x.com/?u=https%3A%2F%2Fa.io%2Fb%2Fc");
  const masked = parseText("x.com/?u=https%3a%2F%2Fa.io%2fb%2Fc");
  for (const tree of [canonical, masked]) {
    for (const bit of serializeTree(tree)) {
      assert.ok(bit === 0 || bit === 1);
    }
  }
  assert.ok(serializeTree(masked).length > serializeTree(canonical).length);
});

test("nasty corpus stays byte-exact", () => {
  const nasty = [
    "example.com/", // No transforms at all
    "a.io/?q=%2", // Truncated escape
    "a.io/?q=100%25+2%3d3&r=%zz",
    "t.co/r?u=http%3A%2F%2Fa.b%2Fc%20d%2Fe%3Ff%3Dg%26h%3Di",
    "x.com/eyJhIjoxfQ.eyJiIjoyfQ.c2ln?after=1",
    "s.io/v?a=aGVsbG8gd29ybGQ%3D&b=aGVsbG8gd29ybGQ=",
    "h.io/0123456789abcdef0123456789ABCDEF/deadbeef01234567",
    "m.co/?j=%7B%22url%22%3A%22https%3A%2F%2Fx.io%2F%3Fq%3D1%22%7D",
    "d.io/?x=" + Buffer.from("nested=" +
      encodeURIComponent("https://deep.example/?a=b")).toString("base64url")
  ];
  for (const text of nasty) assertInvertible(text);
});
