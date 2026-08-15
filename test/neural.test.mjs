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

/*
 * hamr-url-model-v3: quantized tensors (int8/int4 with per-row f16
 * scales) dequantized at load, and the version-gated matmul4 kernel.
 * These build a tiny synthetic v3 file in memory - real-model pinned
 * vectors live above with the other versions.
 */
function buildTinyV3 () {
  // f16 bit patterns for exactly-representable constants
  const F16 = { "1": 0x3c00, "0.5": 0x3800, "0.25": 0x3400, "-0.5": 0xb800, "2": 0x4000 };
  const dim = 4, mlp = 8, ctx = 6, vocabN = 5;
  const qkvInts = Int8Array.from(
    { length: 3 * dim * dim }, (_, i) => ((i * 37) % 255) - 127);
  const projNibbles = Uint8Array.from(
    { length: dim * dim }, (_, i) => (i * 7) % 16); // stored value+8
  const upInts = Int8Array.from(
    { length: mlp * dim }, (_, i) => ((i * 53) % 255) - 127);
  const downNibbles = Uint8Array.from(
    { length: dim * mlp }, (_, i) => (i * 11) % 16);

  const tensors = [
    { name: "embed", shape: [vocabN, dim] },
    { name: "pos", shape: [ctx, dim] },
    { name: "b0.norm1", shape: [dim] },
    { name: "b0.qkv", shape: [3 * dim, dim], dtype: "int8" },
    { name: "b0.proj", shape: [dim, dim], dtype: "int4" },
    { name: "b0.norm2", shape: [dim] },
    { name: "b0.up", shape: [mlp, dim], dtype: "int8" },
    { name: "b0.down", shape: [dim, mlp], dtype: "int4" },
    { name: "norm", shape: [dim] }
  ];
  const header = JSON.stringify({
    format: "hamr-url-model-v3", vocab: vocabN, dim, layers: 1, heads: 2,
    mlpDim: mlp, maxLen: ctx, linkVersion: 3, tokens: ["a", "b", "c", "d"],
    tensors
  });
  const hb = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(4 + hb.length + 4096);
  const view = new DataView(buf);
  view.setUint32(0, hb.length, true);
  new Uint8Array(buf, 4, hb.length).set(hb);
  let o = 4 + hb.length;
  const putF16 = (v) => { view.setUint16(o, F16[String(v)], true); o += 2; };
  // embed + pos: alternating 1 / 0.5 / -0.5 / 0.25
  const pattern = ["1", "0.5", "-0.5", "0.25"];
  for (let i = 0; i < (vocabN + ctx) * dim; i ++) putF16(pattern[i % 4]);
  for (let i = 0; i < dim; i ++) putF16("1");        // b0.norm1
  for (let r = 0; r < 3 * dim; r ++) putF16("0.5");  // qkv scales
  for (const q of qkvInts) { view.setInt8(o, q); o += 1; }
  for (let r = 0; r < dim; r ++) putF16("0.25");     // proj scales
  for (let b = 0; b < projNibbles.length; b += 2) {
    view.setUint8(o, projNibbles[b] | (projNibbles[b + 1] << 4)); o += 1;
  }
  for (let i = 0; i < dim; i ++) putF16("1");        // b0.norm2
  for (let r = 0; r < mlp; r ++) putF16("0.5");      // up scales
  for (const q of upInts) { view.setInt8(o, q); o += 1; }
  for (let r = 0; r < dim; r ++) putF16("2");        // down scales
  for (let b = 0; b < downNibbles.length; b += 2) {
    view.setUint8(o, downNibbles[b] | (downNibbles[b + 1] << 4)); o += 1;
  }
  for (let i = 0; i < dim; i ++) putF16("1");        // norm
  return { buf, qkvInts, projNibbles, upInts, downNibbles };
}

test("v3 loader dequantizes int8 and int4 tensors exactly", () => {
  const { buf, qkvInts, projNibbles, upInts, downNibbles } = buildTinyV3();
  const m = new URLModel(buf);
  assert.equal(m.linkVersion, 3);
  assert.equal(m.fastKernels, true);
  for (let i = 0; i < qkvInts.length; i ++) {
    assert.equal(m.tensors["b0.qkv"][i], qkvInts[i] * 0.5, `qkv[${i}]`);
  }
  for (let i = 0; i < projNibbles.length; i ++) {
    assert.equal(m.tensors["b0.proj"][i], (projNibbles[i] - 8) * 0.25,
      `proj[${i}]`);
  }
  for (let i = 0; i < upInts.length; i ++) {
    assert.equal(m.tensors["b0.up"][i], upInts[i] * 0.5, `up[${i}]`);
  }
  for (let i = 0; i < downNibbles.length; i ++) {
    assert.equal(m.tensors["b0.down"][i], (downNibbles[i] - 8) * 2,
      `down[${i}]`);
  }
  // f16 tensors load unchanged alongside quantized ones
  assert.equal(m.tensors["embed"][0], 1);
  assert.equal(m.tensors["embed"][1], 0.5);
  assert.equal(m.tensors["embed"][2], -0.5);
  assert.equal(m.tensors["embed"][3], 0.25);
});

test("v3 inference is deterministic and the v2 kernel is untouched", () => {
  const { buf } = buildTinyV3();
  const a = new URLModel(buf).session();
  const b = new URLModel(buf).session();
  for (const id of [0, 1, 3, 2, 4]) {
    const la = a.feed(id);
    const lb = b.feed(id);
    assert.deepEqual(Array.from(la), Array.from(lb));
  }
  // Existing (v2) models must never take the fast kernel: its
  // different summation order would break issued payloads
  assert.equal(model.fastKernels, false);
  assert.equal(modelV1.fastKernels, false);
});

/*
 * Chunked coding (payload version >= 3): URLs beyond the model
 * context are coded as EOS-terminated context-restarting chunks in
 * one arithmetic stream. Exercised here by re-versioning the shipped
 * model; the real v3 model gets its own pinned vectors.
 */
test("chunked coding round-trips beyond-context URLs", async () => {
  const raw = (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer;
  const chunkedModel = new URLModel(withLinkVersion(raw, 3));

  const long = "https://data.example-archive.org/collections/2024/expedition-photos/"
    + "region-north-atlantic/vessel-research-7/deck-camera-03/"
    + "capture-2024-06-19T14-22-51Z-frame-000482-exposure-auto-wb-daylight.jpg"
    + "?checksum=9f3a1c77d2e648b0&signature=vRt2LpQ8xYw4Nc6bJmH0aZsEuDkFgO51"
    + "&expires=1718900000&session=b81f2ce4a90d47e3";
  assert.ok(chunkedModel.tokenize(long.slice(8)).length + 1 >
    chunkedModel.maxLen, "test URL must exceed the model context");
  // The v2 model refuses it (falls back to classic in the hybrid)...
  assert.equal(neuralCompressToNumber(model, long), null);
  // ...the chunked model codes it and round-trips exactly
  const payload = compressHybrid(long, outputAlphabetASCII, chunkedModel);
  assert.equal(payloadSchemeVersion(payload, outputAlphabetASCII), 3);
  assert.equal(decompressHybrid(payload, outputAlphabetASCII, chunkedModel), long);
  // ...and it beats classic for this long, structured URL
  assert.ok(payload.length < compress(long, outputAlphabetASCII).length);

  // A URL whose token count lands exactly on a chunk boundary ends
  // with an empty final chunk and still round-trips
  const capacity = chunkedModel.maxLen - 2;
  let boundary = "https://example.com/";
  while (chunkedModel.tokenize(boundary.slice(8)).length % capacity !== 0 ||
         chunkedModel.tokenize(boundary.slice(8)).length === 0) {
    boundary += "x";
  }
  const bPayload = compressHybrid(boundary, outputAlphabetASCII, chunkedModel);
  assert.equal(decompressHybrid(bPayload, outputAlphabetASCII, chunkedModel),
    boundary);
});

test("chunked coding is bit-identical to v2 for short URLs", async () => {
  const raw = (await readFile(new URL("../model/url-model.bin", import.meta.url))).buffer;
  const chunkedModel = new URLModel(withLinkVersion(raw, 3));
  for (const link of ["https://www.example.com/some/path?a=1&b=2",
                      "https://en.wikipedia.org/wiki/Hammer"]) {
    const n2 = neuralCompressToNumber(model, link);
    const n3 = neuralCompressToNumber(chunkedModel, link);
    // Strip the unary version markers; the coded content must match
    assert.equal(n2 >> 3n, n3 >> 4n, link);
  }
});

test("v3 loader applies per-group scales", () => {
  // Minimal file: one int8 and one int4 tensor, 2x4, group size 2 -
  // scales laid out (rows x cols/group), row-major
  const F16 = { "1": 0x3c00, "0.5": 0x3800, "0.25": 0x3400, "2": 0x4000 };
  const header = JSON.stringify({
    format: "hamr-url-model-v3", vocab: 2, dim: 4, layers: 0, heads: 1,
    mlpDim: 4, maxLen: 4, linkVersion: 3, tokens: ["a"],
    tensors: [
      { name: "g8", shape: [2, 4], dtype: "int8", group: 2 },
      { name: "g4", shape: [2, 4], dtype: "int4", group: 2 }
    ]
  });
  const hb = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(4 + hb.length + 64);
  const view = new DataView(buf);
  view.setUint32(0, hb.length, true);
  new Uint8Array(buf, 4, hb.length).set(hb);
  let o = 4 + hb.length;
  // g8 scales: rows 2 x groups 2 = [1, 0.5, 0.25, 2]
  for (const s of ["1", "0.5", "0.25", "2"]) { view.setUint16(o, F16[s], true); o += 2; }
  const q8 = [10, -20, 30, -40, 50, -60, 70, -80];
  for (const q of q8) { view.setInt8(o, q); o += 1; }
  // g4 scales: [0.5, 2, 1, 0.25]
  for (const s of ["0.5", "2", "1", "0.25"]) { view.setUint16(o, F16[s], true); o += 2; }
  const q4 = [-8, 7, 3, -3, 5, -5, 0, 2]; // stored as value+8 nibbles
  for (let i = 0; i < q4.length; i += 2) {
    view.setUint8(o, (q4[i] + 8) | ((q4[i + 1] + 8) << 4)); o += 1;
  }
  const m = new URLModel(buf);
  const exp8 = [10 * 1, -20 * 1, 30 * 0.5, -40 * 0.5,
                50 * 0.25, -60 * 0.25, 70 * 2, -80 * 2];
  const exp4 = [-8 * 0.5, 7 * 0.5, 3 * 2, -3 * 2,
                5 * 1, -5 * 1, 0 * 0.25, 2 * 0.25];
  assert.deepEqual(Array.from(m.tensors["g8"]), exp8);
  assert.deepEqual(Array.from(m.tensors["g4"]), exp4);
});
