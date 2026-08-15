/**
 * @file Neural link compression: a tiny character-level transformer
 * predicts each character of a URL, and an arithmetic coder turns
 * those predictions into near-optimal bits.
 *
 * Inspired by the transformer experiment in
 * https://github.com/ansisg/hamr, which drove an arithmetic coder
 * with a 268MB GPT running on onnxruntime-web. This implementation
 * keeps the idea but replaces the model with a ~1.4MB character-level
 * transformer evaluated by the plain-JavaScript engine below - small
 * enough to ship with the site as a static file.
 *
 * DETERMINISM: arithmetic coding only works if the decoder reproduces
 * the encoder's probabilities bit-for-bit, potentially on a different
 * machine, browser, and CPU. Floating-point basic operations
 * (+ - * / sqrt) are IEEE-754 correctly-rounded and therefore identical
 * everywhere, but transcendental functions (Math.exp, Math.pow, ...)
 * are NOT guaranteed to be. This engine therefore only uses basic
 * operations plus its own exp() built from them. Do not "simplify" any
 * of this to Math.exp/Math.tanh/Math.pow - that would make payloads
 * encoded on one browser fail to decode on another.
 */

import { arithmeticEncode, arithmeticDecode } from "./arithmetic-coder.js";

// Character vocabulary: EOS = 0, printable ASCII 0x21..0x7E -> 1..94.
// URL.href output is always in this range (everything else is
// percent-encoded by the URL parser).
const EOS = 0;

/** Exact double constants (literals, not Math.* which may vary) */
const LN2 = 0.6931471805599453;
const INV_LN2 = 1.4426950408889634;

// Exact table of 2^n built by halving (exact for normal doubles)
const POW2 = new Float64Array(1101); // POW2[i] = 2^(-i)
POW2[0] = 1;
for (let i = 1; i < POW2.length; i ++) POW2[i] = POW2[i - 1] * 0.5;

/**
 * Deterministic exp() for non-positive arguments, using only
 * correctly-rounded operations: argument reduction by ln2, then a
 * fixed-order Taylor series.
 * @param {number} x Exponent, must be <= 0
 * @returns {number} e^x
 */
function detExp (x) {
  if (x < -708) return 0;
  const n = Math.floor(x * INV_LN2 + 0.5);
  const r = x - n * LN2; // |r| <= ~0.35
  let term = 1;
  let sum = 1;
  for (let i = 1; i <= 13; i ++) {
    term = term * r / i;
    sum += term;
  }
  return POW2[-n] * sum;
}

/**
 * Deterministic base-2 logarithm, likewise built only from
 * correctly-rounded operations: exact power-of-two range reduction,
 * then a fixed-order atanh series. Used to score candidate
 * tokenizations at encode time; scores feed payload selection, so
 * they must be bit-identical across engines for every encoder to
 * produce the same payload.
 * @param {number} x Argument, must be >= 0 and finite
 * @returns {number} log2(x), or -Infinity for zero
 */
function detLog2 (x) {
  if (x === 0) return -Infinity;
  let e = 0;
  while (x < 1) { x *= 2; e --; }
  while (x >= 2) { x *= 0.5; e ++; }
  // ln(m) = 2 atanh(z) with z = (m-1)/(m+1) in [0, 1/3); the series
  // gains ~3 bits per term, so 14 terms reach double precision
  const z = (x - 1) / (x + 1);
  const z2 = z * z;
  let term = z;
  let sum = z;
  for (let i = 3; i <= 29; i += 2) {
    term *= z2;
    sum += term / i;
  }
  return e + 2 * sum * INV_LN2;
}

/**
 * In-place softmax over a Float64Array using deterministic exp.
 * @param {Float64Array} x Logits, replaced by probabilities
 */
function softmax (x) {
  let max = -Infinity;
  for (let i = 0; i < x.length; i ++) {
    if (x[i] > max) max = x[i];
  }
  let sum = 0;
  for (let i = 0; i < x.length; i ++) {
    x[i] = detExp(x[i] - max);
    sum += x[i];
  }
  for (let i = 0; i < x.length; i ++) x[i] /= sum;
}

/**
 * Converts an IEEE-754 half-precision value to a double. Exact.
 * @param {number} h 16-bit representation
 * @returns {number} Equivalent double
 */
function halfToDouble (h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) return sign * mantissa * POW2[24];
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (1024 + mantissa) * POW2[10] * (exponent > 15 ? (1 << (exponent - 15)) : POW2[15 - exponent]);
}

/** RMS-normalizes `x` scaled by `g` into `out`. */
function rmsNorm (x, g, out) {
  let sum = 0;
  for (let i = 0; i < x.length; i ++) sum += x[i] * x[i];
  const scale = 1 / Math.sqrt(sum / x.length + 1e-5);
  for (let i = 0; i < x.length; i ++) out[i] = x[i] * scale * g[i];
}

/** Computes `weight (rows x cols) * x (cols)` into `out` (rows). */
function matmul (weight, rows, cols, x, out) {
  for (let r = 0; r < rows; r ++) {
    let sum = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c ++) {
      sum += weight[base + c] * x[c];
    }
    out[r] = sum;
  }
}

/**
 * The URL character model: a small causal transformer with learned
 * position embeddings, RMSNorm, and a ReLU MLP - all expressible in
 * correctly-rounded operations.
 */
export class URLModel {
  /**
   * Parses a model file (little-endian: uint32 header length, JSON
   * header, then float16 tensor data in header order).
   * @param {ArrayBuffer} buffer Model file contents
   */
  constructor (buffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    if (header.format !== "hamr-url-model-v1" &&
        header.format !== "hamr-url-model-v2") {
      throw `Unknown model format: "${header.format}"`;
    }
    /*
     * Tokenization: v1 models are character-level (EOS = 0, printable
     * ASCII 0x21..0x7E -> 1..94). v2 models carry a token string list
     * in the header, applied by greedy longest-match; ids follow list
     * order starting at 1, EOS stays 0. Both tokenizers are exact
     * string operations - determinism is unaffected.
     */
    if (header.tokens) {
      this.tokens = header.tokens;
      this.tokenIds = new Map(header.tokens.map((s, i) => [s, i + 1]));
      this.maxTokenLength = 1;
      for (const s of header.tokens) {
        if (s.length > this.maxTokenLength) this.maxTokenLength = s.length;
      }
    } else {
      this.tokens = null;
    }
    this.vocab = header.vocab;
    this.dim = header.dim;
    this.layers = header.layers;
    this.heads = header.heads;
    this.mlpDim = header.mlpDim;
    this.maxLen = header.maxLen;
    /*
     * The payload version this model encodes/decodes. Each retrained
     * model gets the next version; payloads carry the version in
     * their unary marker, so links made by older models stay
     * decodable as long as the older model file stays deployed
     * (model/url-model-v<N>.bin). Absent in early files = version 1.
     */
    this.linkVersion = header.linkVersion || 1;

    this.tensors = {};
    let offset = 4 + headerLength;
    for (const { name, shape } of header.tensors) {
      const count = shape.reduce((a, b) => a * b, 1);
      const data = new Float64Array(count);
      for (let i = 0; i < count; i ++) {
        data[i] = halfToDouble(view.getUint16(offset + i * 2, true));
      }
      this.tensors[name] = data;
      offset += count * 2;
    }
  }

  /**
   * Tokenizes text into symbol ids, or null if any part of the text
   * can't be represented. Character-level for v1 models; greedy
   * longest-match over the header's token list for v2.
   * @param {string} text Scheme-less URL text
   * @returns {number[]?} Symbol ids (without EOS)
   */
  tokenize (text) {
    const symbols = [];
    if (!this.tokens) {
      for (const char of text) {
        const code = char.charCodeAt(0);
        if (code < 0x21 || code > 0x7e) return null;
        symbols.push(code - 0x20);
      }
      return symbols;
    }
    let i = 0;
    while (i < text.length) {
      let matched = 0;
      const limit = Math.min(this.maxTokenLength, text.length - i);
      for (let length = limit; length > 0; length --) {
        const id = this.tokenIds.get(text.slice(i, i + length));
        if (id !== undefined) {
          symbols.push(id);
          matched = length;
          break;
        }
      }
      if (!matched) return null;
      i += matched;
    }
    return symbols;
  }

  /**
   * Converts symbol ids back into text.
   * @param {number[]} symbols Symbol ids (without EOS)
   * @returns {string} Reconstructed text
   */
  detokenize (symbols) {
    if (!this.tokens) {
      return symbols.map(s => String.fromCharCode(s + 0x20)).join("");
    }
    return symbols.map(s => this.tokens[s - 1]).join("");
  }

  /**
   * Starts an incremental inference session (with per-layer key/value
   * caches, so each fed token costs one transformer step, not a full
   * reprocessing of the context). `fork` clones the session state so
   * the encoder's tokenization search can branch hypotheses without
   * re-feeding their shared prefix; forks share the per-position
   * key/value arrays, which are written once and only read afterwards,
   * so a fork costs pointer copies, not recomputation.
   * @returns {{feed: (id: number) => Float64Array, fork: () => object}}
   *  `feed` consumes one token id and returns next-token logits
   */
  session () {
    const { dim, heads, layers, mlpDim, vocab, maxLen, tensors } = this;
    const headDim = dim / heads;
    const attnScale = 1 / Math.sqrt(headDim);

    const spawn = (kInit, vInit, positionInit) => {
      // Per-layer caches of past keys/values, laid out per position
      const kCache = kInit.map(layer => layer.slice());
      const vCache = vInit.map(layer => layer.slice());
      let position = positionInit;

      const x = new Float64Array(dim);
      const h = new Float64Array(dim);
      const qkv = new Float64Array(3 * dim);
      const attnOut = new Float64Array(dim);
      const proj = new Float64Array(dim);
      const mlp = new Float64Array(mlpDim);
      const logits = new Float64Array(vocab);

      const feed = (id) => {
        if (position >= maxLen) throw "Model context window exceeded.";
        const embed = tensors["embed"];
        const pos = tensors["pos"];
        for (let i = 0; i < dim; i ++) {
          x[i] = embed[id * dim + i] + pos[position * dim + i];
        }

        for (let l = 0; l < layers; l ++) {
          // Attention
          rmsNorm(x, tensors[`b${l}.norm1`], h);
          matmul(tensors[`b${l}.qkv`], 3 * dim, dim, h, qkv);
          kCache[l].push(qkv.slice(dim, 2 * dim));
          vCache[l].push(qkv.slice(2 * dim, 3 * dim));
          const keys = kCache[l];
          const values = vCache[l];
          const scores = new Float64Array(keys.length);
          for (let head = 0; head < heads; head ++) {
            const base = head * headDim;
            for (let j = 0; j < keys.length; j ++) {
              let sum = 0;
              const k = keys[j];
              for (let i = 0; i < headDim; i ++) {
                sum += qkv[base + i] * k[base + i];
              }
              scores[j] = sum * attnScale;
            }
            softmax(scores);
            for (let i = 0; i < headDim; i ++) attnOut[base + i] = 0;
            for (let j = 0; j < values.length; j ++) {
              const v = values[j];
              const weight = scores[j];
              for (let i = 0; i < headDim; i ++) {
                attnOut[base + i] += weight * v[base + i];
              }
            }
          }
          matmul(tensors[`b${l}.proj`], dim, dim, attnOut, proj);
          for (let i = 0; i < dim; i ++) x[i] += proj[i];

          // MLP
          rmsNorm(x, tensors[`b${l}.norm2`], h);
          matmul(tensors[`b${l}.up`], mlpDim, dim, h, mlp);
          for (let i = 0; i < mlpDim; i ++) {
            if (mlp[i] < 0) mlp[i] = 0;
          }
          matmul(tensors[`b${l}.down`], dim, mlpDim, mlp, proj);
          for (let i = 0; i < dim; i ++) x[i] += proj[i];
        }

        rmsNorm(x, tensors["norm"], h);
        // Output head is tied to the embedding table
        matmul(embed, vocab, dim, h, logits);
        position ++;
        return logits;
      };

      const fork = () => spawn(kCache, vCache, position);

      return { feed, fork };
    };

    const empty = [];
    for (let l = 0; l < layers; l ++) empty.push([]);
    return spawn(empty, empty, 0);
  }
}

/**
 * Wraps a model into the probability callback the arithmetic coder
 * expects. Feeding is incremental: the coder always extends the
 * context by one symbol, so each call feeds only the newest token.
 * @param {URLModel} model Loaded model
 * @returns {(context: number[]) => Float64Array} Probability callback
 */
export function modelProbabilities (model) {
  const session = model.session();
  let fed = 0;
  return (context) => {
    // The first call primes with EOS; afterwards feed new symbols
    let logits = null;
    if (fed === 0) {
      logits = session.feed(EOS);
      fed = 1;
    }
    while (fed <= context.length) {
      logits = session.feed(context[fed - 1]);
      fed ++;
    }
    if (logits === null) {
      throw "Arithmetic coder context out of sync with model session.";
    }
    const probs = logits.slice();
    softmax(probs);
    return probs;
  };
}

/*
 * Encoder-side tokenization search. Greedy longest-match is only one
 * of many segmentations whose concatenation equals the URL text, and
 * the decoder never re-tokenizes - it arithmetic-decodes a symbol
 * stream and concatenates the symbols' strings - so ANY segmentation
 * decodes to the same URL. The encoder is therefore free to search
 * segmentations for the one the model predicts most cheaply. This is
 * a pure encode-time improvement: payload format, decode behavior,
 * and already-issued links are untouched.
 */
const BEAM_WIDTH = 4;
const FINAL_CANDIDATES = 2;

/**
 * Beam-searches segmentations of `text` over the model's token list,
 * returning the estimated-cheapest candidates first. Costs are
 * accumulated -log2 probabilities including the terminating EOS - an
 * estimate, since the coder quantizes frequencies, which is why the
 * caller re-encodes the leading candidates with the real coder and
 * compares actual payloads.
 * @param {URLModel} model Loaded model with a token list
 * @param {string} text Scheme-less URL text
 * @returns {number[][]} Candidate symbol sequences (without EOS)
 */
function searchSegmentations (model, text) {
  // Same length budget the greedy path is subject to: the session
  // fits a priming EOS, the tokens, and a terminating EOS
  const maxSymbols = model.maxLen - 2;
  const length = text.length;

  const softmaxed = (logits) => {
    const probs = logits.slice(); // `feed` reuses its logits buffer
    softmax(probs);
    return probs;
  };

  // Hypotheses are compared by cost, then by symbol sequence: costs
  // are deterministic (detExp/detLog2 only), so the full ordering -
  // and with it the chosen payload - is identical on every engine
  const compare = (a, b) => {
    if (a.cost !== b.cost) return a.cost < b.cost ? -1 : 1;
    const n = Math.min(a.symbols.length, b.symbols.length);
    for (let i = 0; i < n; i ++) {
      if (a.symbols[i] !== b.symbols[i]) return a.symbols[i] - b.symbols[i];
    }
    return a.symbols.length - b.symbols.length;
  };

  // arrivals[p] collects hypotheses whose tokens cover text[0..p);
  // every parent sits at an earlier position, so by the time p is
  // processed the list is complete and can be pruned to the beam.
  // Sessions are forked lazily - only hypotheses that survive pruning
  // pay for a transformer step.
  const arrivals = [];
  for (let p = 0; p <= length; p ++) arrivals.push([]);
  const finals = [];

  const expand = (hyp, p) => {
    if (hyp.symbols.length >= maxSymbols) return;
    const limit = Math.min(model.maxTokenLength, length - p);
    for (let l = 1; l <= limit; l ++) {
      const id = model.tokenIds.get(text.slice(p, p + l));
      if (id === undefined) continue;
      arrivals[p + l].push({
        parent: hyp,
        id,
        cost: hyp.cost - detLog2(hyp.probs[id]),
        symbols: hyp.symbols.concat(id)
      });
    }
  };

  const root = { session: model.session(), symbols: [], cost: 0 };
  root.probs = softmaxed(root.session.feed(EOS));
  expand(root, 0);

  for (let p = 1; p <= length; p ++) {
    const candidates = arrivals[p];
    if (!candidates.length) continue;
    candidates.sort(compare);
    for (const candidate of candidates.slice(0, BEAM_WIDTH)) {
      const session = candidate.parent.session.fork();
      const probs = softmaxed(session.feed(candidate.id));
      if (p < length) {
        expand({ session, symbols: candidate.symbols,
          cost: candidate.cost, probs }, p);
      } else {
        finals.push({ symbols: candidate.symbols,
          cost: candidate.cost - detLog2(probs[EOS]) });
      }
    }
  }

  finals.sort(compare);
  return finals.slice(0, FINAL_CANDIDATES).map(final => final.symbols);
}

/*
 * Payload format (bits from least significant):
 *   marker   - unary payload version: N one-bits then a zero, where
 *              N is the encoding model's linkVersion (classic
 *              payloads are version 0 - a single zero bit - and are
 *              handled by compress.js unchanged)
 *   isHTTPS  - 1 bit
 *   payload  - arithmetic-coded bits of the scheme-less URL text,
 *              preceded by a sentinel 1 bit to preserve leading zeros
 */

/**
 * Arithmetic-codes one symbol sequence and frames it as a payload
 * number (sentinel bit, isHTTPS bit, unary version marker).
 * @param {URLModel} model Loaded model
 * @param {number[]} symbols Symbol ids (without EOS)
 * @param {boolean} isHTTPS Whether the link scheme is https
 * @returns {BigInt} Payload number
 */
function encodeSymbols (model, symbols, isHTTPS) {
  const bits = arithmeticEncode(symbols.concat(EOS), modelProbabilities(model));

  let number = 1n; // Sentinel to preserve leading zero bits
  for (const bit of bits) {
    number = (number << 1n) | BigInt(bit);
  }
  number = (number << 1n) | (isHTTPS ? 1n : 0n);
  // Unary version marker matching the model that encoded this payload
  const version = BigInt(model.linkVersion);
  number = (number << (version + 1n)) | ((1n << version) - 1n);
  return number;
}

/**
 * Compresses a link with the neural coder into a raw payload number.
 * @param {URLModel} model Loaded model
 * @param {string} input Link to compress
 * @returns {BigInt?} Payload number, or null if the link doesn't fit
 *  the model (too long, or contains bytes outside the vocabulary)
 */
export function neuralCompressToNumber (model, input) {
  let url;
  if (URL.canParse(input)) {
    url = new URL(input);
  } else {
    url = new URL("http://" + input);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  const isHTTPS = url.protocol === "https:";
  const text = url.href.slice(isHTTPS ? 8 : 7);

  // Greedy tokenization; bail out on anything the model can't
  // represent (every printable character has a single-char token, so
  // a greedy failure means no segmentation exists at all)
  const greedy = model.tokenize(text);
  if (greedy === null) return null;
  // Sequence overhead: one priming EOS plus the terminating EOS
  if (greedy.length + 2 > model.maxLen) return null;

  // Search for cheaper segmentations. Character-level models have
  // only one segmentation, and version-1 token payload bits are
  // pinned to greedy (that model stays deployed purely to decode old
  // links), so the search applies to version 2 onwards.
  const candidates = [greedy];
  if (model.tokens && model.linkVersion >= 2) {
    candidates.push(...searchSegmentations(model, text));
  }

  // Greedy comes first, so on a tie the payload matches the
  // pre-search encoder's output
  let best = null;
  const seen = new Set();
  for (const symbols of candidates) {
    const key = symbols.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    const number = encodeSymbols(model, symbols, isHTTPS);
    // Smaller payload number = same or fewer output symbols
    if (best === null || number < best) best = number;
  }
  return best;
}

/**
 * Decompresses a neural payload number back into a full link.
 * @param {URLModel} model Loaded model
 * @param {BigInt} number Payload number (version 1 marker included)
 * @returns {string} Full link
 */
export function neuralDecompressNumber (model, number) {
  let version = 0;
  while (number & 1n) {
    version ++;
    number >>= 1n;
  }
  number >>= 1n;
  if (version !== model.linkVersion) {
    throw `Payload version ${version} needs model version ${version}, `
      + `but this model is version ${model.linkVersion}.`;
  }
  const isHTTPS = number & 1n;
  number >>= 1n;

  const bitString = number.toString(2);
  if (bitString === "0") throw "Empty neural payload.";
  const bits = [];
  for (let i = 1; i < bitString.length; i ++) { // Skip sentinel bit
    bits.push(bitString.charCodeAt(i) - 0x30);
  }

  const symbols = arithmeticDecode(
    bits, modelProbabilities(model), (s) => s === EOS, model.maxLen);
  symbols.pop(); // Drop EOS
  return (isHTTPS ? "https://" : "http://") + model.detokenize(symbols);
}

/**
 * Reads the version marker of a payload number.
 * @param {BigInt} number Payload number
 * @returns {number} Version (0 = classic, 1 = neural)
 */
export function payloadVersion (number) {
  let version = 0;
  while (number & 1n) {
    version ++;
    number >>= 1n;
  }
  return version;
}
