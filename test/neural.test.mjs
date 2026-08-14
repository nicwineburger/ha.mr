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
  assert.equal(model.vocab, 95);
  assert.ok(model.dim >= 32);
  assert.ok(model.layers >= 1);
  assert.equal(model.maxLen, 128);
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
  // Too long for the model's context window
  const long = "https://example.com/" + "a".repeat(150);
  assert.equal(neuralCompressToNumber(model, long), null);
  // Non-http(s) schemes aren't representable in the payload format
  assert.equal(neuralCompressToNumber(model, "ftp://example.com/x"), null);
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
    ["https://www.example.com/some/path?a=1&b=2", "!/Z+$L6r5_zF9d(e-X!", "R$O0TCZ9Z2C$UR60D5BFE+"],
    ["https://en.wikipedia.org/wiki/Hammer", "@vbdmvo_W!", "GPU-YOKK0L7"],
    ["https://github.com/user/repo/blob/main/README.md", "Ph&Ttw?$#v9i!A0si/RFHg", "NCTK5P2ASD3+L5K$+/:L0YK:DG"],
    ["https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator", "+T@k9Jt[E?+U-eo9QKbu5hTh.k$", "*PB76DOVDO1T07G+PCMSBRUF:MY-4MW"]
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

test("neural payloads require the model to decode", () => {
  const number = neuralCompressToNumber(model, "https://www.example.com/x");
  assert.notEqual(number, null);
  // Force a neural payload through the hybrid decoder without a model
  const payload = numberToString(number, outputAlphabetASCII);
  assert.throws(() => decompressHybrid(payload, outputAlphabetASCII, null));
  assert.equal(new URL(decompressHybrid(payload, outputAlphabetASCII, model)).href,
    "https://www.example.com/x");
});
