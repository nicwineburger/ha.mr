import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { arithmeticEncode, arithmeticDecode } from "../arithmetic-coder.js";
import {
  URLModel,
  neuralCompressToNumber,
  neuralDecompressNumber,
  payloadVersion
} from "../neural.js";
import {
  compressHybrid,
  decompressHybrid,
  payloadSchemeVersion
} from "../hybrid.js";
import { compress, decompress, numberToString } from "../compress.js";
import { outputAlphabetASCII, outputAlphabetQR } from "../alphabets.js";

/**
 * A deterministic mock model: probabilities depend on the last context
 * symbol, exercising the coder without neural inference.
 */
function mockProbabilities (context) {
  const last = context.length ? context[context.length - 1] : 0;
  const probs = new Float64Array(8);
  for (let i = 0; i < 8; i ++) {
    probs[i] = 1 + ((i * 7 + last * 3) % 11);
  }
  let sum = 0;
  for (const p of probs) sum += p;
  for (let i = 0; i < 8; i ++) probs[i] /= sum;
  return probs;
}

test("arithmetic coder round-trips symbol sequences", () => {
  // Deterministic pseudo-random sequences over symbols 1..7,
  // terminated by 0
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % 7 + 1;
  for (let len = 0; len < 40; len ++) {
    const symbols = [];
    for (let i = 0; i < len; i ++) symbols.push(rnd());
    symbols.push(0);
    const bits = arithmeticEncode(symbols, mockProbabilities);
    const decoded = arithmeticDecode(bits, mockProbabilities, s => s === 0, 1000);
    assert.deepEqual(decoded, symbols, `sequence of length ${len}`);
  }
});

test("arithmetic coder approaches the model's entropy", () => {
  // A highly skewed distribution should compress far below 3 bits
  // (uniform cost) per symbol
  const skewed = () => Float64Array.from([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.0025, 0.0025]);
  const symbols = new Array(200).fill(0).concat([1]);
  const bits = arithmeticEncode(symbols, skewed);
  // 200 symbols at p=0.9 is ~30 bits of entropy; allow generous slack
  assert.ok(bits.length < 60, `expected < 60 bits, got ${bits.length}`);
});

const model = new URLModel(
  (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);

test("model file parses with expected dimensions", () => {
  // The shipped v2 model: 1024-token vocabulary, 96-token context
  assert.equal(model.vocab, 1024);
  assert.equal(model.maxLen, 96);
  assert.ok(model.tokens.length === 1023);
  assert.ok(model.dim >= 32);
  assert.ok(model.layers >= 1);
});

test("tokenizer round-trips URL text", () => {
  const texts = [
    "www.example.com/some/path?a=1&b=2",
    "en.wikipedia.org/wiki/Hammer_(tool)",
    "a-b.xyz:8080/x_y/z.html#frag%20ment"
  ];
  for (const text of texts) {
    const ids = model.tokenize(text);
    assert.notEqual(ids, null);
    assert.equal(model.detokenize(ids), text);
  }
  // Out-of-vocabulary characters are rejected, not mangled
  assert.equal(model.tokenize("café.example/x"), null);
});

test("model inference is self-consistent across sessions", () => {
  // Two sessions fed the same tokens must produce identical logits -
  // the foundation the arithmetic coder relies on
  const a = model.session();
  const b = model.session();
  const tokens = [0, 5, 10, 20, 30, 1];
  for (const t of tokens) {
    const la = a.feed(t);
    const lb = b.feed(t);
    assert.deepEqual(Array.from(la), Array.from(lb));
  }
});

const neuralCases = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://www.google.com/search?q=test",
  "http://localhost:8080/test",
  "https://github.com/user/repo/blob/main/README.md",
  "https://example.com",
  "https://a-b.xyz/x_y/z.html#section",
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
];

test("neural coder round-trips URLs", () => {
  for (const link of neuralCases) {
    const number = neuralCompressToNumber(model, link);
    assert.notEqual(number, null, `expected ${link} to fit the model`);
    assert.equal(payloadVersion(number), 1);
    const back = neuralDecompressNumber(model, number);
    assert.equal(new URL(back).href, new URL(link).href, `round-trip of ${link}`);
  }
});

test("neural coder refuses what it can't represent", () => {
  // Too long for the model's ~200-char effective context window
  const long = "https://example.com/" +
    Array.from({ length: 120 }, (_, i) => `x${i % 10}z`).join("/");
  assert.equal(neuralCompressToNumber(model, long), null);
  // Non-http(s) schemes aren't representable in the payload format
  assert.equal(neuralCompressToNumber(model, "ftp://example.com/x"), null);
});

test("hybrid survives classic-scheme encode failures", () => {
  // Underscores are invalid DNS but occur in real hostnames; the
  // classic domain dictionary can't encode them and throws. The
  // hybrid should fall through to the neural coder.
  const link = "http://www_test.example.com/a";
  assert.throws(() => compress(link, outputAlphabetASCII));
  const payload = compressHybrid(link, outputAlphabetASCII, model);
  assert.equal(decompressHybrid(payload, outputAlphabetASCII, model), link);
  // Without a model, the classic error propagates
  assert.throws(() => compressHybrid(link, outputAlphabetASCII, null));
});

test("hybrid picks a scheme and round-trips either way", () => {
  for (const link of neuralCases) {
    for (const alphabet of [outputAlphabetASCII, outputAlphabetQR]) {
      const payload = compressHybrid(link, alphabet, model);
      const back = decompressHybrid(payload, alphabet, model);
      assert.equal(new URL(back).href, new URL(link).href, `round-trip of ${link}`);
      // The hybrid payload is never larger than classic-only
      assert.ok(payload.length <= compress(link, alphabet).length,
        `hybrid should not exceed classic for ${link}`);
    }
  }
});

test("hybrid without a model equals classic compression", () => {
  for (const link of neuralCases) {
    assert.equal(compressHybrid(link, outputAlphabetASCII, null),
      compress(link, outputAlphabetASCII));
  }
});

test("classic payloads decode through the hybrid path", () => {
  // Links encoded before the neural coder existed (or by clients
  // without the model) must keep decoding identically
  for (const link of neuralCases) {
    const payload = compress(link, outputAlphabetASCII);
    assert.equal(payloadSchemeVersion(payload, outputAlphabetASCII), 0);
    assert.equal(decompressHybrid(payload, outputAlphabetASCII, model),
      decompress(payload, outputAlphabetASCII));
  }
});

test("pinned payloads stay stable", () => {
  /*
   * Payloads generated with the shipped model. If this test fails,
   * encoding behavior changed - which BREAKS EVERY ISSUED NEURAL LINK.
   * That must never happen by accident: the model file, the arithmetic
   * coder, and the inference math (including its deterministic exp)
   * all have to stay bit-compatible. See model/README.md.
   */
  const vectors = [
    ["https://www.example.com/some/path?a=1&b=2", "XnE2w.Z((Ar+H4UO]:$", "6EI81FE:WFI/8IPJ3Q:O-:"],
    ["https://en.wikipedia.org/wiki/Hammer", "XY)ICJh#", "CO0XB:$*:"],
    ["https://github.com/user/repo/blob/main/README.md", "/dq~n$0R-]VJ.pX*dDKtK2", "WVEOBO0P/FC-BC+-8593UUQ5:/"],
    ["https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator", "kd#IyiYiGz#-@Qe1WP71qSYF5!", "O$3852YF/AF24MOFRLW.W82MH35FM/"]
  ];
  for (const [link, ascii, qr] of vectors) {
    assert.equal(compressHybrid(link, outputAlphabetASCII, model), ascii);
    assert.equal(compressHybrid(link, outputAlphabetQR, model), qr);
    assert.equal(new URL(decompressHybrid(ascii, outputAlphabetASCII, model)).href,
      new URL(link).href);
    assert.equal(new URL(decompressHybrid(qr, outputAlphabetQR, model)).href,
      new URL(link).href);
  }
});

/** Rebuilds a model buffer with a different linkVersion in its header. */
function withLinkVersion (buffer, version) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  header.linkVersion = version;
  const hb = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + hb.length + buffer.byteLength - 4 - headerLength);
  new DataView(out.buffer).setUint32(0, hb.length, true);
  out.set(hb, 4);
  out.set(new Uint8Array(buffer, 4 + headerLength), 4 + hb.length);
  return out.buffer;
}

test("payload versions route to the matching model", async () => {
  // A future retrained model ships with the next linkVersion; its
  // payloads carry that version and refuse to decode with any other
  // model. This is the upgrade path that keeps old links working.
  const raw = (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer;
  const modelV3 = new URLModel(withLinkVersion(raw, 3));
  assert.equal(modelV3.linkVersion, 3);

  const link = "https://www.example.com/versioned/path";
  const payload = compressHybrid(link, outputAlphabetASCII, modelV3);
  assert.equal(payloadSchemeVersion(payload, outputAlphabetASCII), 3);
  assert.equal(decompressHybrid(payload, outputAlphabetASCII, modelV3), link);

  // The version-1 model must refuse a version-3 payload, and vice
  // versa - silent cross-decoding would produce garbage URLs
  assert.throws(() => decompressHybrid(payload, outputAlphabetASCII, model),
    /version/);
  const payloadV1 = compressHybrid(link, outputAlphabetASCII, model);
  assert.equal(payloadSchemeVersion(payloadV1, outputAlphabetASCII), 1);
  assert.throws(() => decompressHybrid(payloadV1, outputAlphabetASCII, modelV3),
    /version/);
});

test("neural payloads require the model to decode", () => {
  const number = neuralCompressToNumber(model, "https://www.example.com/x");
  assert.notEqual(number, null);
  // Force a neural payload through the hybrid decoder without a model
  const payload = numberToString(number, outputAlphabetASCII);
  assert.throws(() => decompressHybrid(payload, outputAlphabetASCII, null));
  assert.equal(new URL(decompressHybrid(payload, outputAlphabetASCII, model)).href,
    "https://www.example.com/x");
});
