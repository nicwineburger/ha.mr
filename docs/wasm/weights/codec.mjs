/**
 * @file Prototype entropy-coded weight container ("wpack"): a
 * transparent recompression of a model file that reproduces it
 * byte-identically at load. Decode-at-load is NOT
 * determinism-sensitive (the reconstructed file is verified
 * byte-identical; inference math never changes), only load-time
 * matters - so the decoder is plain JS, measured by
 * wasm/weights/measure.mjs.
 *
 * Scheme: the file is split by the tensor manifest into
 *  - the JSON header, gzipped (browsers can DecompressionStream it);
 *  - f16 bytes (f16 tensors + quantization scales), phase-split into
 *    low-byte and high-byte streams, each static order-0 rANS coded
 *    (the high byte holds sign+exponent, which is highly skewed);
 *  - int4 nibbles, one stream, static order-1 rANS (context =
 *    previous nibble);
 *  - int8 bytes (none in the current v3 file), static order-0.
 *
 * The rANS coder is the classic byte-wise construction (32-bit state,
 * 12-bit frequencies, byte renormalization), written with plain
 * Number arithmetic - all intermediates stay far below 2^53.
 */

const PROB_BITS = 12;
const PROB_SCALE = 1 << PROB_BITS; // 4096
const RANS_L = 1 << 23;

/**
 * Scales symbol counts to a 12-bit frequency table; every present
 * symbol keeps a nonzero frequency.
 * @param {Uint32Array} counts Raw symbol counts
 * @returns {Uint16Array} Frequencies summing to 4096
 */
export function normalizeFreqs (counts) {
  let total = 0;
  for (const c of counts) total += c;
  const freqs = new Uint16Array(counts.length);
  if (total === 0) return freqs;
  let sum = 0;
  let largest = 0;
  for (let s = 0; s < counts.length; s ++) {
    if (counts[s] === 0) continue;
    freqs[s] = Math.max(1, Math.round(counts[s] / total * PROB_SCALE));
    sum += freqs[s];
    if (freqs[s] > freqs[largest]) largest = s;
  }
  // Push the rounding remainder into the most likely symbol; steal
  // from other symbols in the rare case the largest can't absorb it
  let excess = sum - PROB_SCALE;
  freqs[largest] -= excess;
  while (freqs[largest] < 1) {
    for (let s = 0; s < freqs.length && freqs[largest] < 1; s ++) {
      if (s !== largest && freqs[s] > 1) { freqs[s] --; freqs[largest] ++; }
    }
  }
  return freqs;
}

/**
 * rANS-encodes a symbol stream with per-context static frequencies.
 * @param {Uint8Array} symbols Symbol stream
 * @param {(i: number) => number} contextOf Context id for position i
 * @param {Uint16Array[]} freqs Frequency table per context
 * @returns {Uint8Array} Coded bytes
 */
export function ransEncode (symbols, contextOf, freqs) {
  const starts = freqs.map(f => {
    const start = new Uint32Array(f.length + 1);
    for (let s = 0; s < f.length; s ++) start[s + 1] = start[s] + f[s];
    return start;
  });
  const out = [];
  let x = RANS_L;
  // rANS encodes in reverse so the decoder streams forward
  for (let i = symbols.length - 1; i >= 0; i --) {
    const ctx = contextOf(i);
    const s = symbols[i];
    const freq = freqs[ctx][s];
    const xMax = (RANS_L >> PROB_BITS) * 256 * freq;
    while (x >= xMax) {
      out.push(x % 256);
      x = Math.floor(x / 256);
    }
    x = Math.floor(x / freq) * PROB_SCALE + (x % freq) + starts[ctx][s];
  }
  for (let i = 0; i < 4; i ++) {
    out.push(x % 256);
    x = Math.floor(x / 256);
  }
  return Uint8Array.from(out.reverse());
}

/**
 * Decodes `count` symbols from a ransEncode stream.
 * @param {Uint8Array} bytes Coded bytes
 * @param {number} count Symbols to decode
 * @param {(i: number, prev: number) => number} contextOf Context id
 *  for position i given the previously decoded symbol
 * @param {Uint16Array[]} freqs The encoder's frequency tables
 * @returns {Uint8Array} Decoded symbols
 */
export function ransDecode (bytes, count, contextOf, freqs) {
  const starts = freqs.map(f => {
    const start = new Uint32Array(f.length + 1);
    for (let s = 0; s < f.length; s ++) start[s + 1] = start[s] + f[s];
    return start;
  });
  // cum -> symbol lookup per context
  const lookup = freqs.map((f, ctx) => {
    const table = new Uint8Array(PROB_SCALE);
    for (let s = 0; s < f.length; s ++) {
      table.fill(s, starts[ctx][s], starts[ctx][s + 1]);
    }
    return table;
  });
  let ptr = 0;
  let x = 0;
  for (let i = 0; i < 4; i ++) x = x * 256 + bytes[ptr ++];
  const out = new Uint8Array(count);
  let prev = 0;
  for (let i = 0; i < count; i ++) {
    const ctx = contextOf(i, prev);
    const cum = x % PROB_SCALE;
    const s = lookup[ctx][cum];
    out[i] = s;
    x = freqs[ctx][s] * Math.floor(x / PROB_SCALE) + cum - starts[ctx][s];
    while (x < RANS_L && ptr < bytes.length) {
      x = x * 256 + bytes[ptr ++];
    }
    prev = s;
  }
  return out;
}

/**
 * Walks a model file's layout using its tensor manifest.
 * @param {Uint8Array} file Original model file bytes
 * @returns {{headerEnd: number, regions: {kind: string, offset: number,
 *  length: number}[]}} Regions in file order; kind is "f16" (f16
 *  tensor data or quantization scales), "int4", or "int8"
 */
export function fileRegions (file) {
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(file.subarray(4, 4 + headerLength)));
  const regions = [];
  let offset = 4 + headerLength;
  for (const { shape, dtype, group } of header.tensors) {
    const count = shape.reduce((a, b) => a * b, 1);
    if (dtype === "int8" || dtype === "int4") {
      const rows = shape[0];
      const cols = count / rows;
      const perRow = group ? cols / group : 1;
      const scaleBytes = rows * perRow * 2;
      regions.push({ kind: "f16", offset, length: scaleBytes });
      offset += scaleBytes;
      const dataBytes = dtype === "int8" ? count : count / 2;
      regions.push({ kind: dtype, offset, length: dataBytes });
      offset += dataBytes;
    } else {
      regions.push({ kind: "f16", offset, length: count * 2 });
      offset += count * 2;
    }
  }
  return { headerEnd: 4 + headerLength, regions };
}

/** Splits the file into the four symbol streams the container codes. */
export function buildStreams (file) {
  const { headerEnd, regions } = fileRegions(file);
  const lo = [];
  const hi = [];
  const nibbles = [];
  const int8 = [];
  for (const { kind, offset, length } of regions) {
    if (kind === "f16") {
      for (let i = 0; i < length; i += 2) {
        lo.push(file[offset + i]);
        hi.push(file[offset + i + 1]);
      }
    } else if (kind === "int4") {
      for (let i = 0; i < length; i ++) {
        nibbles.push(file[offset + i] & 0x0f, file[offset + i] >> 4);
      }
    } else {
      for (let i = 0; i < length; i ++) int8.push(file[offset + i]);
    }
  }
  return {
    header: file.subarray(0, headerEnd),
    lo: Uint8Array.from(lo),
    hi: Uint8Array.from(hi),
    nibbles: Uint8Array.from(nibbles),
    int8: Uint8Array.from(int8)
  };
}

const order0 = () => (i, prev) => 0;
const order1Nibble = (symbols) =>
  symbols
    ? (i) => (i === 0 ? 0 : symbols[i - 1])       // encode side
    : (i, prev) => (i === 0 ? 0 : prev);          // decode side

/** Counts per-context symbol frequencies for a stream. */
function countFreqs (symbols, contexts, alphabet) {
  const counts = [];
  for (let c = 0; c < contexts; c ++) counts.push(new Uint32Array(alphabet));
  let prev = 0;
  for (let i = 0; i < symbols.length; i ++) {
    counts[contexts === 1 ? 0 : (i === 0 ? 0 : prev)][symbols[i]] ++;
    prev = symbols[i];
  }
  return counts.map(normalizeFreqs);
}

/**
 * Packs a model file into the container. `gzipHeader` is injected so
 * the codec stays runtime-agnostic (node:zlib in tools, a
 * pre-gzipped blob in a browser).
 * @param {Uint8Array} file Original model file bytes
 * @param {(bytes: Uint8Array) => Uint8Array} gzipHeader Gzip function
 * @returns {{container: Uint8Array, breakdown: Object}} Packed bytes
 *  and per-stream size accounting
 */
export function pack (file, gzipHeader) {
  const streams = buildStreams(file);
  const headerGz = gzipHeader(streams.header);

  const loFreqs = countFreqs(streams.lo, 1, 256);
  const hiFreqs = countFreqs(streams.hi, 1, 256);
  const nibbleFreqs = countFreqs(streams.nibbles, 16, 16);
  const int8Freqs = countFreqs(streams.int8, 1, 256);

  const loCoded = ransEncode(streams.lo, order0(), loFreqs);
  const hiCoded = ransEncode(streams.hi, order0(), hiFreqs);
  const nibbleCoded = ransEncode(
    streams.nibbles, order1Nibble(streams.nibbles), nibbleFreqs);
  const int8Coded = ransEncode(streams.int8, order0(), int8Freqs);

  const chunks = [];
  const pushU32 = (n) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, n, true);
    chunks.push(b);
  };
  const pushFreqs = (freqs) => {
    for (const f of freqs) {
      const b = new Uint8Array(f.length * 2);
      const bv = new DataView(b.buffer);
      for (let s = 0; s < f.length; s ++) bv.setUint16(s * 2, f[s], true);
      chunks.push(b);
    }
  };
  chunks.push(new TextEncoder().encode("WPK1"));
  pushU32(file.length);
  pushU32(headerGz.length);
  chunks.push(headerGz);
  for (const [coded, stream, freqs] of [
    [loCoded, streams.lo, loFreqs],
    [hiCoded, streams.hi, hiFreqs],
    [nibbleCoded, streams.nibbles, nibbleFreqs],
    [int8Coded, streams.int8, int8Freqs]
  ]) {
    pushFreqs(freqs);
    pushU32(stream.length);
    pushU32(coded.length);
    chunks.push(coded);
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const container = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    container.set(c, offset);
    offset += c.length;
  }
  return {
    container,
    breakdown: {
      headerRaw: streams.header.length, headerGz: headerGz.length,
      f16Raw: streams.lo.length + streams.hi.length,
      f16Coded: loCoded.length + hiCoded.length,
      int4Raw: streams.nibbles.length / 2, int4Coded: nibbleCoded.length,
      int8Raw: streams.int8.length, int8Coded: int8Coded.length
    }
  };
}

/**
 * Unpacks a container back into the byte-identical model file.
 * @param {Uint8Array} container Packed bytes
 * @param {(bytes: Uint8Array) => Uint8Array} gunzipHeader Gunzip
 * @returns {Uint8Array} The original model file bytes
 */
export function unpack (container, gunzipHeader) {
  const view = new DataView(
    container.buffer, container.byteOffset, container.byteLength);
  let offset = 4;
  const readU32 = () => {
    const n = view.getUint32(offset, true);
    offset += 4;
    return n;
  };
  const fileLength = readU32();
  const headerGzLength = readU32();
  const header = gunzipHeader(
    container.subarray(offset, offset + headerGzLength));
  offset += headerGzLength;

  const readStream = (contexts, alphabet, contextOf) => {
    const freqs = [];
    for (let c = 0; c < contexts; c ++) {
      const f = new Uint16Array(alphabet);
      for (let s = 0; s < alphabet; s ++) {
        f[s] = view.getUint16(offset, true);
        offset += 2;
      }
      freqs.push(f);
    }
    const count = readU32();
    const codedLength = readU32();
    const coded = container.subarray(offset, offset + codedLength);
    offset += codedLength;
    return ransDecode(coded, count, contextOf, freqs);
  };

  const lo = readStream(1, 256, order0());
  const hi = readStream(1, 256, order0());
  const nibbles = readStream(16, 16, order1Nibble(null));
  const int8 = readStream(1, 256, order0());

  const file = new Uint8Array(fileLength);
  file.set(header, 0);
  const { regions } = fileRegions(file);
  let loPos = 0, hiPos = 0, nibblePos = 0, int8Pos = 0;
  for (const { kind, offset: at, length } of regions) {
    if (kind === "f16") {
      for (let i = 0; i < length; i += 2) {
        file[at + i] = lo[loPos ++];
        file[at + i + 1] = hi[hiPos ++];
      }
    } else if (kind === "int4") {
      for (let i = 0; i < length; i ++) {
        file[at + i] = nibbles[nibblePos] | (nibbles[nibblePos + 1] << 4);
        nibblePos += 2;
      }
    } else {
      for (let i = 0; i < length; i ++) file[at + i] = int8[int8Pos ++];
    }
  }
  return file;
}
