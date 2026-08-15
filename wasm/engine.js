/**
 * @file JS glue for the production WebAssembly inference engine.
 *
 * Pairs engine.wasm (built from engine.c, recipe in wasm/README.md)
 * with a faithful port of the neural.js encode/decode pipeline.
 * neural.js is untouched and remains the reference implementation and
 * automatic fallback (see engine-select.js at the repo root, which
 * every production entry point goes through): this module re-uses its
 * model loader (URLModel - dequantization, tokenizer, payload
 * versioning) and the shared arithmetic coder, and swaps ONLY the
 * inference engine underneath. Every payload produced or consumed
 * here must be bit-identical to neural.js - that is the acceptance
 * criterion, not a hope (see wasm/wasm.test.mjs and wasm/verify.mjs).
 *
 * Memory model: the WASM module owns its linear memory. At load, the
 * dequantized f32 tensors are copied in once. Above the weights lives
 * a bump-allocated arena for per-session key/value slabs and the
 * per-session tables of k/v offsets that let forked sessions share
 * cached positions by pointer, exactly like the JS engine's
 * write-once per-position arrays. reset() reclaims the whole arena;
 * callers only invoke it when no session is live (between candidate
 * encodes, and at each chunk restart, where the previous chunk's
 * session is dead by construction).
 */

import { URLModel } from "../neural.js";
import { arithmeticEncode, arithmeticDecode } from "../arithmetic-coder.js";

const EOS = 0;

/** Exact double constant, same literal as neural.js */
const INV_LN2 = 1.4426950408889634;

/**
 * Deterministic base-2 logarithm, transcribed from neural.js detLog2
 * (not exported there). Only used for the encoder's search costs; the
 * source is identical, so results are bit-identical.
 * @param {number} x Argument, must be >= 0 and finite
 * @returns {number} log2(x), or -Infinity for zero
 */
function detLog2 (x) {
  if (x === 0) return -Infinity;
  let e = 0;
  while (x < 1) { x *= 2; e --; }
  while (x >= 2) { x *= 0.5; e ++; }
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

/** Tensor-name -> engine.c set_tensor kind (3..8 are per-layer). */
const TENSOR_KINDS = { norm1: 3, qkv: 4, proj: 5, norm2: 6, up: 7, down: 8 };

const align16 = (n) => (n + 15) & ~15;

/**
 * A loaded model bound to a WASM instance. One instance per engine;
 * weights are copied into linear memory once at construction.
 */
export class WasmEngine {
  /**
   * Instantiates the WASM module and copies the model's weights in.
   * @param {URLModel|ArrayBuffer} model Parsed model (or raw file)
   * @param {BufferSource|WebAssembly.Module} wasmBinary engine.wasm
   * @returns {Promise<WasmEngine>} Ready engine
   */
  static async create (model, wasmBinary) {
    if (!(model instanceof URLModel)) model = new URLModel(model);
    const module = wasmBinary instanceof WebAssembly.Module
      ? wasmBinary : await WebAssembly.compile(wasmBinary);
    const instance = await WebAssembly.instantiate(module, {});
    return new WasmEngine(model, instance);
  }

  /**
   * @param {URLModel} model Parsed model
   * @param {WebAssembly.Instance} instance Instantiated engine.wasm
   */
  constructor (model, instance) {
    this.model = model;
    this.exports = instance.exports;
    this.memory = instance.exports.memory;

    const entries = Object.entries(model.tensors);
    let offset = align16(this.exports.__heap_base.value);
    let total = offset;
    for (const [, data] of entries) total += align16(data.length * 4);
    this.grow(total);

    for (const [name, data] of entries) {
      new Float32Array(this.memory.buffer, offset, data.length).set(data);
      let kind;
      let layer = 0;
      if (name === "embed") kind = 0;
      else if (name === "pos") kind = 1;
      else if (name === "norm") kind = 2;
      else {
        const match = /^b(\d+)\.(\w+)$/.exec(name);
        layer = Number(match[1]);
        kind = TENSOR_KINDS[match[2]];
      }
      this.exports.set_tensor(kind, layer, offset);
      offset += align16(data.length * 4);
    }
    this.exports.init(model.dim, model.heads, model.layers, model.mlpDim,
      model.vocab, model.maxLen, model.fastKernels ? 1 : 0);
    this.probsPtr = this.exports.probs_ptr();
    this.arenaBase = align16(offset);
    this.arenaNext = this.arenaBase;
  }

  /** Grows linear memory so `bytes` total bytes are addressable. */
  grow (bytes) {
    const current = this.memory.buffer.byteLength;
    if (bytes > current) {
      // Grow with 16MB headroom so per-feed allocations don't grow
      // one page at a time
      this.memory.grow(Math.ceil((bytes - current) / 65536) + 256);
    }
  }

  /** Bump-allocates 16-aligned arena bytes; returns the byte offset. */
  alloc (bytes) {
    const ptr = this.arenaNext;
    this.arenaNext = ptr + align16(bytes);
    this.grow(this.arenaNext);
    return ptr;
  }

  /**
   * Reclaims the whole session arena. Only call with no live
   * sessions - every session's tables and k/v slabs live there.
   */
  reset () {
    this.arenaNext = this.arenaBase;
  }

  /**
   * Starts an inference session mirroring URLModel.session(): `feed`
   * runs one transformer step, `fork` clones the session so the
   * tokenization search can branch without re-feeding the shared
   * prefix (the k/v slabs are shared by table copy - written once,
   * read-only afterwards). Unlike the JS engine, `feed` returns the
   * softmaxed probabilities (both of neural.js's consumers softmax
   * the returned logits immediately, so this is the same math).
   * @returns {{feed: (id: number) => Float64Array, fork: () => object}}
   */
  session () {
    const { layers, maxLen, dim, vocab } = this.model;
    const tableBytes = layers * maxLen * 2 * 4;
    const engine = this;

    const spawn = (srcTable, positionInit) => {
      const table = engine.alloc(tableBytes);
      if (srcTable !== null) {
        new Uint8Array(engine.memory.buffer)
          .copyWithin(table, srcTable, srcTable + tableBytes);
      }
      let position = positionInit;

      const feed = (id) => {
        if (position >= maxLen) throw "Model context window exceeded.";
        const kv = engine.alloc(layers * 2 * dim * 8);
        const entries =
          new Uint32Array(engine.memory.buffer, table, layers * maxLen * 2);
        for (let l = 0; l < layers; l ++) {
          entries[(l * maxLen + position) * 2] = kv + l * 2 * dim * 8;
          entries[(l * maxLen + position) * 2 + 1] = kv + (l * 2 + 1) * dim * 8;
        }
        engine.exports.feed(id, position, table);
        position ++;
        return new Float64Array(
          engine.memory.buffer, engine.probsPtr, vocab).slice();
      };

      const fork = () => spawn(table, position);

      return { feed, fork };
    };

    return spawn(null, 0);
  }
}

/* ---- Pipeline ported from neural.js, engine swapped underneath ---- */

function chunkCapacity (model) {
  return model.maxLen - 2;
}

const CHUNKED_MAX_SYMBOLS = 4096;

/** Chunk framing, transcribed from neural.js chunkFrame. */
function chunkFrame (model, symbols) {
  const capacity = chunkCapacity(model);
  const framed = [];
  for (let i = 0; i < symbols.length; i += capacity) {
    framed.push(...symbols.slice(i, i + capacity), EOS);
  }
  if (symbols.length % capacity === 0) {
    framed.push(EOS);
  }
  return framed.length > CHUNKED_MAX_SYMBOLS ? null : framed;
}

/**
 * Probability callback for the arithmetic coder, transcribed from
 * neural.js modelProbabilities with a WASM session underneath. At a
 * chunk restart the previous chunk's session is dead, so the arena is
 * reclaimed before spawning the fresh session.
 * @param {WasmEngine} engine Loaded engine
 * @param {{context: number[]}} [chunkState] Latest-context out-param
 * @returns {(context: number[]) => Float64Array} Probability callback
 */
export function wasmProbabilities (engine, chunkState = null) {
  const chunked = engine.model.linkVersion >= 3;
  let session = engine.session();
  let fed = 0;
  return (context) => {
    let probs = null;
    if (fed === 0) {
      probs = session.feed(EOS);
      fed = 1;
    }
    while (fed <= context.length) {
      const symbol = context[fed - 1];
      if (chunked && symbol === EOS) {
        engine.reset();
        session = engine.session();
        probs = session.feed(EOS);
      } else {
        probs = session.feed(symbol);
      }
      fed ++;
    }
    if (probs === null) {
      throw "Arithmetic coder context out of sync with model session.";
    }
    if (chunkState) chunkState.context = context;
    return probs;
  };
}

/** Trailing-run length rule, transcribed from neural.js. */
function closedChunkLength (context) {
  let i = context.length - 1;
  if (i >= 0 && context[i] === EOS) i --;
  let run = 0;
  for (; i >= 0 && context[i] !== EOS; i --) {
    run ++;
  }
  return run;
}

const BEAM_WIDTH = 4;
const FINAL_CANDIDATES = 2;

/**
 * Encoder-side tokenization search, transcribed from neural.js
 * searchSegmentations. The only change: WASM sessions already return
 * softmaxed probabilities, so the explicit softmax step is gone (the
 * numbers are identical). Costs use the same detLog2, hypotheses
 * compare identically, so the chosen candidates - and therefore the
 * payload - match the JS encoder bit for bit.
 * @param {WasmEngine} engine Loaded engine (model has a token list)
 * @param {string} text Scheme-less URL text
 * @returns {number[][]} Candidate symbol sequences (without EOS)
 */
function searchSegmentations (engine, text) {
  const model = engine.model;
  const maxSymbols = model.maxLen - 2;
  const length = text.length;

  const compare = (a, b) => {
    if (a.cost !== b.cost) return a.cost < b.cost ? -1 : 1;
    const n = Math.min(a.symbols.length, b.symbols.length);
    for (let i = 0; i < n; i ++) {
      if (a.symbols[i] !== b.symbols[i]) return a.symbols[i] - b.symbols[i];
    }
    return a.symbols.length - b.symbols.length;
  };

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

  const root = { session: engine.session(), symbols: [], cost: 0 };
  root.probs = root.session.feed(EOS);
  expand(root, 0);

  for (let p = 1; p <= length; p ++) {
    const candidates = arrivals[p];
    if (!candidates.length) continue;
    candidates.sort(compare);
    for (const candidate of candidates.slice(0, BEAM_WIDTH)) {
      const session = candidate.parent.session.fork();
      const probs = session.feed(candidate.id);
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

/**
 * Arithmetic-codes one symbol sequence into a framed payload number,
 * transcribed from neural.js encodeSymbols. Resets the arena first:
 * any sessions from a preceding search or candidate encode are dead.
 * @param {WasmEngine} engine Loaded engine
 * @param {number[]} symbols Symbol ids (without EOS)
 * @param {boolean} isHTTPS Whether the link scheme is https
 * @returns {BigInt?} Payload number
 */
function encodeSymbols (engine, symbols, isHTTPS) {
  const model = engine.model;
  const coded = model.linkVersion >= 3
    ? chunkFrame(model, symbols)
    : symbols.concat(EOS);
  if (coded === null) return null;
  engine.reset();
  const bits = arithmeticEncode(coded, wasmProbabilities(engine));

  let number = 1n;
  for (const bit of bits) {
    number = (number << 1n) | BigInt(bit);
  }
  number = (number << 1n) | (isHTTPS ? 1n : 0n);
  const version = BigInt(model.linkVersion);
  number = (number << (version + 1n)) | ((1n << version) - 1n);
  return number;
}

/**
 * Compresses a link into a raw payload number - the WASM-backed
 * counterpart of neural.js neuralCompressToNumber, same options,
 * bit-identical output.
 * @param {WasmEngine} engine Loaded engine
 * @param {string} input Link to compress
 * @param {{search?: boolean}} [options] `search: false` = greedy only
 * @returns {BigInt?} Payload number, or null if the link doesn't fit
 */
export function wasmCompressToNumber (engine, input, { search = true } = {}) {
  const model = engine.model;
  let url;
  if (URL.canParse(input)) {
    url = new URL(input);
  } else {
    url = new URL("http://" + input);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  const isHTTPS = url.protocol === "https:";
  const text = url.href.slice(isHTTPS ? 8 : 7);

  const greedy = model.tokenize(text);
  if (greedy === null) return null;
  if (model.linkVersion < 3 && greedy.length + 2 > model.maxLen) return null;

  const candidates = [greedy];
  if (search && model.tokens && model.linkVersion >= 2 &&
      greedy.length + 2 <= model.maxLen) {
    engine.reset();
    candidates.push(...searchSegmentations(engine, text));
  }

  let best = null;
  const seen = new Set();
  for (const symbols of candidates) {
    const key = symbols.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    const number = encodeSymbols(engine, symbols, isHTTPS);
    if (number === null) continue;
    if (best === null || number < best) best = number;
  }
  return best;
}

/**
 * Decompresses a payload number back into a full link - the
 * WASM-backed counterpart of neural.js neuralDecompressNumber.
 * @param {WasmEngine} engine Loaded engine
 * @param {BigInt} number Payload number (version marker included)
 * @returns {string} Full link
 */
export function wasmDecompressNumber (engine, number) {
  const model = engine.model;
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
  for (let i = 1; i < bitString.length; i ++) {
    bits.push(bitString.charCodeAt(i) - 0x30);
  }

  engine.reset();
  let symbols;
  if (model.linkVersion >= 3) {
    const capacity = chunkCapacity(model);
    const chunkState = { context: [] };
    symbols = arithmeticDecode(
      bits, wasmProbabilities(engine, chunkState),
      (s) => s === EOS && closedChunkLength(chunkState.context) < capacity,
      CHUNKED_MAX_SYMBOLS);
    symbols = symbols.filter((s) => s !== EOS);
  } else {
    symbols = arithmeticDecode(
      bits, wasmProbabilities(engine), (s) => s === EOS, model.maxLen);
    symbols.pop();
  }
  return (isHTTPS ? "https://" : "http://") + model.detokenize(symbols);
}
