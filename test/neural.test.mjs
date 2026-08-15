import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { arithmeticEncode, arithmeticDecode } from "../arithmetic-coder.js";
import {
  URLModel,
  neuralCompressToNumber,
  neuralDecompressNumber,
  modelProbabilities,
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

// The latest model (encodes payload version 2) and the archived v1
// model (kept deployed so version-1 links stay decodable)
const model = new URLModel(
  (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer);
const modelV1 = new URLModel(
  (await readFile(new URL("../model/url-model-v1.bin", import.meta.url))).buffer);

test("model files parse with expected dimensions and versions", () => {
  assert.equal(model.linkVersion, 2);
  assert.equal(modelV1.linkVersion, 1);
  for (const m of [model, modelV1]) {
    assert.equal(m.vocab, 1024);
    assert.equal(m.maxLen, 96);
    assert.ok(m.tokens.length === 1023);
    assert.ok(m.dim >= 32);
    assert.ok(m.layers >= 1);
  }
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
    assert.equal(payloadVersion(number), model.linkVersion);
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

/**
 * Mirrors the pre-search encoder: greedy tokenization fed straight
 * into the arithmetic coder, then the payload framing. The searched
 * encoder must never do worse than this baseline.
 */
function greedyNeuralNumber (model, link) {
  const url = new URL(link);
  const isHTTPS = url.protocol === "https:";
  const symbols = model.tokenize(url.href.slice(isHTTPS ? 8 : 7));
  symbols.push(0); // EOS
  const bits = arithmeticEncode(symbols, modelProbabilities(model));
  let number = 1n;
  for (const bit of bits) number = (number << 1n) | BigInt(bit);
  number = (number << 1n) | (isHTTPS ? 1n : 0n);
  const version = BigInt(model.linkVersion);
  return (number << (version + 1n)) | ((1n << version) - 1n);
}

test("searched payloads never exceed greedy payloads", () => {
  for (const link of neuralCases) {
    const searched = neuralCompressToNumber(model, link);
    assert.ok(searched <= greedyNeuralNumber(model, link),
      `search must not lose to greedy for ${link}`);
  }
});

test("search opt-out reproduces the greedy encoder exactly", () => {
  // The browser's typing path encodes with search disabled for
  // latency; that must be bit-identical to the pre-search encoder
  for (const link of neuralCases) {
    assert.equal(neuralCompressToNumber(model, link, { search: false }),
      greedyNeuralNumber(model, link), link);
  }
});

test("encoding is deterministic", () => {
  // The tokenization search must break ties consistently: the same
  // link always yields the same payload
  for (const link of neuralCases) {
    assert.equal(compressHybrid(link, outputAlphabetASCII, model),
      compressHybrid(link, outputAlphabetASCII, model));
    assert.equal(compressHybrid(link, outputAlphabetQR, model),
      compressHybrid(link, outputAlphabetQR, model));
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

/*
 * Pinned payloads per model version, in two kinds:
 *
 * - DECODE vectors (payload -> URL) freeze compatibility with issued
 *   links. If one fails, decode behavior changed - which BREAKS EVERY
 *   ISSUED NEURAL LINK of that version. That must never happen: the
 *   model files, the arithmetic coder, and the inference math
 *   (including its deterministic exp) all have to stay bit-compatible.
 *   These vectors are kept green forever, retired model versions
 *   included (against their archived model file).
 * - ENCODE vectors (URL -> payload) pin the current encoder's exact
 *   output bits. They may be regenerated only by a deliberate
 *   encode-side improvement (like the tokenization search), and the
 *   superseded encode outputs are then demoted to decode-only vectors
 *   above all else - links carrying them are already in the wild.
 *
 * See model/README.md and the invariants in CLAUDE.md.
 */
const pinnedLinks = [
  "https://www.example.com/some/path?a=1&b=2",
  "https://en.wikipedia.org/wiki/Hammer",
  "https://github.com/user/repo/blob/main/README.md",
  "https://blog.example-widgets.net/2024/06/announcing-the-new-widget-configurator"
];

function checkVectors (vectors, vectorModel) {
  for (const [link, ascii, qr] of vectors) {
    assert.equal(compressHybrid(link, outputAlphabetASCII, vectorModel), ascii);
    assert.equal(compressHybrid(link, outputAlphabetQR, vectorModel), qr);
    assert.equal(new URL(decompressHybrid(ascii, outputAlphabetASCII, vectorModel)).href,
      new URL(link).href);
    assert.equal(new URL(decompressHybrid(qr, outputAlphabetQR, vectorModel)).href,
      new URL(link).href);
  }
}

test("pinned v1 payloads stay stable against the archived model", () => {
  checkVectors([
    [pinnedLinks[0], "XnE2w.Z((Ar+H4UO]:$", "6EI81FE:WFI/8IPJ3Q:O-:"],
    [pinnedLinks[1], "XY)ICJh#", "CO0XB:$*:"],
    [pinnedLinks[2], "/dq~n$0R-]VJ.pX*dDKtK2", "WVEOBO0P/FC-BC+-8593UUQ5:/"],
    [pinnedLinks[3], "kd#IyiYiGz#-@Qe1WP71qSYF5!", "O$3852YF/AF24MOFRLW.W82MH35FM/"]
  ], modelV1);
});

test("pinned v2 payloads stay stable", () => {
  // The tokenization search ties greedy on these links, and ties
  // resolve to greedy, so these vectors survived the search's
  // introduction unchanged
  checkVectors([
    [pinnedLinks[0], "$Q-gtbg+.'LjUAFA-", "KT79QRUW+*90A949+8A/"],
    [pinnedLinks[1], "?r9)?8@", "PK3+O8RT"],
    [pinnedLinks[2], "?O+UoHb62N6/FK_rEoI~", "+N6PPKWG5+0OJES24PM6/$Q*"],
    [pinnedLinks[3], "Jo#z:yEW+jB$mrY[bxd2!", "1*AT7DJ/XR.8QGVXJHW$WST1"]
  ], model);
});

test("searched v2 payloads improve on superseded greedy payloads", () => {
  // Links where the tokenization search beats greedy longest-match.
  // Rows: link, the payloads the pre-search encoder issued for it
  // (decode-only vectors - such links are in the wild forever), then
  // the searching encoder's pinned outputs.
  const cases = [
    ["https://illashbyilly.com.au/collections/eye-lashes",
      "$Qk+YKh1Iww?JP+-&", "758DO5+*RW1O2U-5.E6*",
      "ey$xIm[4~tdwDt[", "-UY.J--NKQ7H6ZDUA0"],
    ["https://www.dyson.com.ro/asistenta-dyson/contactati-ne",
      "5p-LrY0O]$Z/B,;c63x&", "8XV76LO**HNM/E*B1CC4PYC",
      "?.z1Xyf&vrvphXZ)/9W", "9ZZ$A4XKAQQDLA8*KAL9SI*"]
  ];
  for (const [link, greedyAscii, greedyQr, ascii, qr] of cases) {
    // Superseded greedy payloads must decode forever
    assert.equal(decompressHybrid(greedyAscii, outputAlphabetASCII, model), link);
    assert.equal(decompressHybrid(greedyQr, outputAlphabetQR, model), link);
    // Current encoder output, never larger than what greedy issued
    assert.equal(compressHybrid(link, outputAlphabetASCII, model), ascii);
    assert.equal(compressHybrid(link, outputAlphabetQR, model), qr);
    assert.ok(ascii.length <= greedyAscii.length, `ascii regression for ${link}`);
    assert.ok(qr.length <= greedyQr.length, `qr regression for ${link}`);
    assert.equal(decompressHybrid(ascii, outputAlphabetASCII, model), link);
    assert.equal(decompressHybrid(qr, outputAlphabetQR, model), link);
  }
});

test("v1 and v2 payloads refuse each other's models", () => {
  const link = "https://www.example.com/cross-version";
  const p1 = compressHybrid(link, outputAlphabetASCII, modelV1);
  const p2 = compressHybrid(link, outputAlphabetASCII, model);
  assert.equal(payloadSchemeVersion(p1, outputAlphabetASCII), 1);
  assert.equal(payloadSchemeVersion(p2, outputAlphabetASCII), 2);
  assert.equal(decompressHybrid(p1, outputAlphabetASCII, modelV1), link);
  assert.equal(decompressHybrid(p2, outputAlphabetASCII, model), link);
  assert.throws(() => decompressHybrid(p1, outputAlphabetASCII, model), /version/);
  assert.throws(() => decompressHybrid(p2, outputAlphabetASCII, modelV1), /version/);
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

  // The latest model must refuse a version-3 payload, and vice
  // versa - silent cross-decoding would produce garbage URLs
  assert.throws(() => decompressHybrid(payload, outputAlphabetASCII, model),
    /version/);
  const payloadLatest = compressHybrid(link, outputAlphabetASCII, model);
  assert.equal(payloadSchemeVersion(payloadLatest, outputAlphabetASCII),
    model.linkVersion);
  assert.throws(() => decompressHybrid(payloadLatest, outputAlphabetASCII, modelV3),
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
