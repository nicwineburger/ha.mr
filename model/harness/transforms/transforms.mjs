/**
 * @file Structural transform search: detects encoded substructure in
 * URLs (percent-encoded nested URLs, base64/base64url text and JSON,
 * JWT segments, long hex ids), unwraps it into a transform tree, and
 * provides byte-exact inversion plus a compact bit-level tree
 * serialization for honest payload accounting.
 *
 * This is a MEASUREMENT PROTOTYPE for the experiment harness. Nothing
 * here is wired into the shipped codecs; the point is to quantify how
 * much a transform-based payload scheme would gain before designing
 * one for real.
 *
 * Tree model: a string is a list of parts; each part is either a
 * literal ({ kind: "lit", text }) or a span ({ kind: "span", type,
 * start, end, raw, meta, ... }) covering [start, end) of the parent
 * text. Span types:
 *   - "pct": percent-decoded region. `decoded` holds the decoded
 *     bytes (latin-1 string), `escapes[i]` is -1 for a literal char or
 *     a 2-bit hex-case code for a decoded escape, so re-encoding is
 *     byte-exact even for mixed-case escapes. `parts` recurses into
 *     the decoded text.
 *   - "b64": base64/base64url region whose decoded bytes are text.
 *     `meta` records alphabet and exact padding; non-canonical
 *     encodings (stray trailing bits, inner '=') are rejected at
 *     detection time by a re-encode check. `parts` recurses.
 *   - "b64" with `binary: true`: base64 whose decoded bytes are
 *     high-entropy binary (session tokens, signatures). Routed to the
 *     8-bits/byte binary channel. In theory that is break-even
 *     (base64 is 6 bits/char = 8 bits/byte), but measured against the
 *     real model (run.mjs entropy) random base64 costs ~7.3 bits/char
 *     through the coder, so the channel saves ~1.3 bits/char minus
 *     tree overhead. Candidates containing '/' are excluded (usually
 *     glued path segments, not blobs).
 *   - "hex": long hex run stored as raw bytes (8 bits/byte in the
 *     hypothetical payload = 4 bits/hex char). Uniform case only.
 * JWTs need no dedicated type: the dots split the three segments into
 * separate base64url candidates, the JSON header/payload pass the
 * text gate and unwrap, and the random signature fails it and stays
 * literal text. Spans that sit inside an x.y.z JWT shape are tagged
 * `tag: "jwt"` for reporting.
 */

const MAX_DEPTH = 4;

/** Unreserved set used by the canonical percent-encoding variants
 * (matches JavaScript's encodeURIComponent). */
const UNRESERVED = new Set(
  ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" +
   "-_.!~*'()").split(""));

const HEX_UPPER = "0123456789ABCDEF";
const HEX_LOWER = "0123456789abcdef";

function isHexDigit (c) {
  return (c >= "0" && c <= "9") || (c >= "a" && c <= "f") ||
    (c >= "A" && c <= "F");
}

function isPrintable (code) {
  return code >= 0x21 && code <= 0x7e;
}

/**
 * Percent-decodes a raw span, recording per-character inversion
 * metadata.
 * @param {string} raw Span text possibly containing %XX escapes
 * @returns {{decoded: string, escapes: number[], nEscapes: number,
 *  nPrintableEscapes: number}} Decoded bytes as a latin-1 string;
 *  `escapes[i]` is -1 for a literal char, else a case code
 *  (bit 1: first hex digit lowercase, bit 0: second lowercase)
 */
export function pctDecode (raw) {
  const decoded = [];
  const escapes = [];
  let nEscapes = 0;
  let nPrintableEscapes = 0;
  let i = 0;
  while (i < raw.length) {
    if (raw[i] === "%" && i + 2 < raw.length + 1 &&
        isHexDigit(raw[i + 1] || "") && isHexDigit(raw[i + 2] || "")) {
      const h1 = raw[i + 1];
      const h2 = raw[i + 2];
      const byte = parseInt(h1 + h2, 16);
      decoded.push(String.fromCharCode(byte));
      escapes.push(((h1 >= "a" && h1 <= "f") ? 2 : 0) |
                   ((h2 >= "a" && h2 <= "f") ? 1 : 0));
      nEscapes ++;
      if (isPrintable(byte)) nPrintableEscapes ++;
      i += 3;
    } else {
      decoded.push(raw[i]);
      escapes.push(-1);
      i ++;
    }
  }
  return { decoded: decoded.join(""), escapes, nEscapes, nPrintableEscapes };
}

/**
 * Re-encodes a decoded string using recorded escape metadata.
 * Byte-exact inverse of pctDecode.
 * @param {string} decoded Decoded bytes as latin-1 string
 * @param {number[]} escapes Per-character metadata from pctDecode
 * @returns {string} Original raw span
 */
export function pctEncode (decoded, escapes) {
  const out = [];
  for (let i = 0; i < decoded.length; i ++) {
    const e = escapes[i];
    if (e < 0) {
      out.push(decoded[i]);
    } else {
      const byte = decoded.charCodeAt(i);
      const table1 = (e & 2) ? HEX_LOWER : HEX_UPPER;
      const table2 = (e & 1) ? HEX_LOWER : HEX_UPPER;
      out.push("%" + table1[byte >> 4] + table2[byte & 15]);
    }
  }
  return out.join("");
}

/**
 * Canonically percent-encodes decoded bytes (escape everything
 * outside the unreserved set) - used to test whether a span's escape
 * pattern is the common encodeURIComponent one, which serializes to a
 * 2-bit variant flag instead of a per-character mask.
 * @param {string} decoded Decoded bytes as latin-1 string
 * @param {boolean} lower Use lowercase hex digits
 * @returns {string} Canonical encoding
 */
export function pctEncodeCanonical (decoded, lower) {
  const table = lower ? HEX_LOWER : HEX_UPPER;
  const out = [];
  for (let i = 0; i < decoded.length; i ++) {
    const c = decoded[i];
    if (UNRESERVED.has(c)) {
      out.push(c);
    } else {
      const byte = decoded.charCodeAt(i);
      out.push("%" + table[byte >> 4] + table[byte & 15]);
    }
  }
  return out.join("");
}

/**
 * Strictly decodes a base64/base64url span and verifies the decoding
 * is canonical by re-encoding.
 * @param {string} raw Candidate span (padding included)
 * @returns {{bytes: Buffer, urlSafe: boolean, pad: number}?} Decoded
 *  bytes and inversion metadata, or null if the span is not clean,
 *  canonical base64
 */
export function b64Decode (raw) {
  const firstPad = raw.indexOf("=");
  const core = firstPad === -1 ? raw : raw.slice(0, firstPad);
  const pad = raw.length - core.length;
  if (pad > 2 || core.includes("=")) return null;
  const hasStd = /[+/]/.test(core);
  const hasUrl = /[-_]/.test(core);
  if (hasStd && hasUrl) return null;
  if (pad > 0 && (core.length + pad) % 4 !== 0) return null;
  if (pad === 0 && core.length % 4 === 1) return null;
  const std = hasUrl
    ? core.replace(/-/g, "+").replace(/_/g, "/")
    : core;
  const bytes = Buffer.from(std, "base64");
  if (b64Encode(bytes, hasUrl, pad) !== raw) return null; // Non-canonical
  return { bytes, urlSafe: hasUrl, pad };
}

/**
 * Re-encodes bytes to base64 with recorded variant metadata.
 * @param {Buffer} bytes Decoded bytes
 * @param {boolean} urlSafe Use the base64url alphabet
 * @param {number} pad Number of '=' characters in the original
 * @returns {string} Base64 span
 */
export function b64Encode (bytes, urlSafe, pad) {
  let s = bytes.toString("base64").replace(/=+$/, "");
  if (urlSafe) s = s.replace(/\+/g, "-").replace(/\//g, "_");
  return s + "=".repeat(pad);
}

/** True if every byte is printable ASCII or common whitespace, i.e.
 * the decoded content is text the model can meaningfully predict. */
function isTexty (bytes) {
  for (const b of bytes) {
    if (!isPrintable(b) && b !== 0x20 && b !== 0x09 && b !== 0x0a &&
        b !== 0x0d) return false;
  }
  return bytes.length > 0;
}

const PCT_LEFT_DELIMS = new Set(["&", "?", "#", "=", ";"]);
const PCT_RIGHT_DELIMS = new Set(["&", "#"]);

/** Finds maximal percent-encoded candidate regions around escapes,
 * bounded by URL structure delimiters. Returns [start, end) pairs. */
function findPctCandidates (text) {
  const spans = [];
  let i = 0;
  while (i < text.length) {
    const p = text.indexOf("%", i);
    if (p === -1) break;
    if (isHexDigit(text[p + 1] || "") && isHexDigit(text[p + 2] || "")) {
      let start = p;
      while (start > 0 && !PCT_LEFT_DELIMS.has(text[start - 1])) start --;
      let end = p + 3;
      while (end < text.length && !PCT_RIGHT_DELIMS.has(text[end])) end ++;
      if (spans.length && start <= spans[spans.length - 1][1]) {
        spans[spans.length - 1][1] = Math.max(spans[spans.length - 1][1], end);
      } else {
        spans.push([start, end]);
      }
      i = end;
    } else {
      i = p + 1;
    }
  }
  return spans;
}

const B64_CANDIDATE = /[A-Za-z0-9+/=_-]{16,}/g;
const HEX_CANDIDATE = /[0-9A-Fa-f]{16,}/g;
const JWT_SHAPE = /eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g;

function overlaps (a, b) {
  return a.start < b.end && b.start < a.end;
}

/**
 * Parses text into a transform tree: literals plus detected,
 * invertibly-unwrapped spans (recursing into decoded content).
 * @param {string} text Input text (URL href without scheme at the top
 *  level, decoded content at deeper levels)
 * @param {object} [opts] Options: `stats` collects detection counts
 *  ({pct,b64,hex,jwt,b64bin}), `noHex` disables the hex transform,
 *  `depth` is internal
 * @returns {object[]} Parts array (see file header)
 */
export function parseText (text, opts = {}) {
  const depth = opts.depth || 0;
  const stats = opts.stats;
  if (depth >= MAX_DEPTH || text.length === 0) {
    return text.length ? [{ kind: "lit", text }] : [];
  }
  const accepted = [];
  const tryAccept = (span) => {
    for (const a of accepted) {
      if (overlaps(a, span)) return false;
    }
    accepted.push(span);
    return true;
  };

  // Percent spans first: they subsume base64/hex content that will be
  // re-detected inside the decoded text on recursion
  for (const [start, end] of findPctCandidates(text)) {
    const raw = text.slice(start, end);
    const { decoded, escapes, nEscapes, nPrintableEscapes } = pctDecode(raw);
    // Worth transforming only if enough escapes decode to printable
    // characters the model can exploit (a lone %20, or a run of UTF-8
    // escapes, saves nothing once re-escaped for the content stream)
    if (nPrintableEscapes < 2 || nPrintableEscapes * 2 < nEscapes) continue;
    let variant = 2; // 0 = canonical upper, 1 = canonical lower, 2 = mask
    if (pctEncodeCanonical(decoded, false) === raw) variant = 0;
    else if (pctEncodeCanonical(decoded, true) === raw) variant = 1;
    tryAccept({
      kind: "span", type: "pct", start, end, raw,
      decoded, escapes, nEscapes, variant
    });
  }

  // Base64 / base64url spans decoding to text. The candidate regex
  // includes '/' and '=' because standard base64 uses them, but URL
  // paths and query '=' glue unrelated text onto real spans - so try
  // the whole candidate first, then fall back to '/'- and
  // '='-delimited sub-candidates (which covers base64url JWT segments
  // and query values)
  const tryB64 = (raw, start) => {
    // Real base64 of text virtually always mixes cases and digits;
    // this rejects ordinary long words and hex runs (which have no
    // uppercase or no lowercase and fall through to the hex detector)
    if (!/[0-9]/.test(raw) || !/[a-z]/.test(raw) || !/[A-Z]/.test(raw)) {
      return false;
    }
    if (raw.indexOf("=") === -1 && raw.length % 4 === 1) {
      raw = raw.slice(0, -1); // Trim to a decodable length
    }
    const dec = b64Decode(raw);
    if (!dec) return false;
    if (!isTexty(dec.bytes)) {
      // Binary payload -> binary channel. Candidates containing '/'
      // are usually glued path segments, not blobs - unless they also
      // contain '+' or '=', which never appear in plain path text
      if (raw.includes("/") && !/[+=]/.test(raw)) return false;
      return tryAccept({
        kind: "span", type: "b64", binary: true,
        start, end: start + raw.length, raw,
        bytes: dec.bytes, urlSafe: dec.urlSafe, pad: dec.pad
      });
    }
    return tryAccept({
      kind: "span", type: "b64", start, end: start + raw.length, raw,
      decoded: dec.bytes.toString("latin1"),
      urlSafe: dec.urlSafe, pad: dec.pad
    });
  };
  for (const m of text.matchAll(B64_CANDIDATE)) {
    if (tryB64(m[0], m.index)) continue;
    if (!/[/=]/.test(m[0])) continue;
    for (const sub of m[0].matchAll(/[A-Za-z0-9+_-]{16,}={0,2}/g)) {
      tryB64(sub[0], m.index + sub.index);
    }
  }

  // Long uniform-case hex runs -> raw bytes (4 bits/hex char)
  if (!opts.noHex) {
    for (const m of text.matchAll(HEX_CANDIDATE)) {
      let raw = m[0];
      if (raw.length % 2 === 1) raw = raw.slice(0, -1);
      const hasLower = /[a-f]/.test(raw);
      const hasUpper = /[A-F]/.test(raw);
      if (hasLower && hasUpper) continue; // Mixed case: skip (rare)
      const nDigits = (raw.match(/[0-9]/g) || []).length;
      const nLetters = raw.length - nDigits;
      if (nDigits < 2 || nLetters < 2) continue; // Words / decimal runs
      tryAccept({
        kind: "span", type: "hex", start: m.index,
        end: m.index + raw.length, raw,
        bytes: Buffer.from(raw, "hex"), upper: hasUpper
      });
    }
  }

  accepted.sort((a, b) => a.start - b.start);

  // Tag spans sitting inside an x.y.z JWT shape, for reporting
  const jwtRanges = [];
  for (const m of text.matchAll(JWT_SHAPE)) {
    jwtRanges.push({ start: m.index, end: m.index + m[0].length });
  }
  for (const span of accepted) {
    if (jwtRanges.some(r => overlaps(r, span))) span.tag = "jwt";
  }

  if (stats) {
    for (const span of accepted) {
      stats[spanTypeKey(span)] = (stats[spanTypeKey(span)] || 0) + 1;
    }
  }

  // Recurse into decoded text of pct/text-b64 spans
  for (const span of accepted) {
    if (span.type === "pct" || (span.type === "b64" && !span.binary)) {
      span.parts = parseText(span.decoded, { ...opts, depth: depth + 1 });
    }
  }

  // Assemble literals + spans
  const parts = [];
  let pos = 0;
  for (const span of accepted) {
    if (span.start > pos) {
      parts.push({ kind: "lit", text: text.slice(pos, span.start) });
    }
    parts.push(span);
    pos = span.end;
  }
  if (pos < text.length) parts.push({ kind: "lit", text: text.slice(pos) });
  return parts;
}

/**
 * Replays a transform tree forward, rebuilding the original text from
 * decoded content and inversion metadata only (stored raw span text
 * is deliberately not used, so this verifies invertibility for real).
 * @param {object[]} parts Transform tree from parseText
 * @returns {string} Reconstructed original text
 */
export function reconstruct (parts) {
  const out = [];
  for (const part of parts) {
    if (part.kind === "lit") {
      out.push(part.text);
    } else if (part.type === "pct") {
      out.push(pctEncode(reconstruct(part.parts), part.escapes));
    } else if (part.type === "b64") {
      out.push(b64Encode(
        part.binary ? part.bytes : Buffer.from(reconstruct(part.parts), "latin1"),
        part.urlSafe, part.pad));
    } else if (part.type === "hex") {
      let s = part.bytes.toString("hex");
      if (part.upper) s = s.toUpperCase();
      out.push(s);
    } else {
      throw `Unknown part: ${part.kind}/${part.type}`;
    }
  }
  return out.join("");
}

/**
 * Parses a URL text and verifies the tree inverts byte-exactly.
 * @param {string} text URL href without scheme
 * @param {object} [opts] Options forwarded to parseText
 * @returns {{parts: object[], ok: boolean, hasTransforms: boolean}}
 *  Tree, inversion status, and whether any span was detected
 */
export function verifyInvertible (text, opts = {}) {
  const parts = parseText(text, opts);
  const hasTransforms = treeHasSpans(parts);
  let ok;
  try {
    ok = reconstruct(parts) === text;
  } catch {
    ok = false;
  }
  return { parts, ok, hasTransforms };
}

/** True if the tree contains at least one span at any depth. */
export function treeHasSpans (parts) {
  return parts.some(p => p.kind === "span");
}

/** Reporting key for a span: jwt tag wins, binary base64 is split out
 * from text base64. */
export function spanTypeKey (span) {
  if (span.tag === "jwt") return "jwt";
  if (span.type === "b64" && span.binary) return "b64bin";
  return span.type;
}

/** Collects span type counts ({pct,b64,b64bin,hex,jwt}) at all depths. */
export function treeSpanCounts (parts) {
  const counts = {};
  const walk = (ps) => {
    for (const p of ps) {
      if (p.kind !== "span") continue;
      const key = spanTypeKey(p);
      counts[key] = (counts[key] || 0) + 1;
      if (p.parts) walk(p.parts);
    }
  };
  walk(parts);
  return counts;
}

/**
 * Escapes decoded content for the model-facing content stream:
 * printable ASCII passes through, '%' becomes %25 and everything else
 * %XX (uppercase), so the mapping is injective and a decoder can
 * recover the exact decoded bytes from the stream.
 * @param {string} text Decoded bytes as latin-1 string
 * @returns {string} Escaped content text
 */
export function escapeContent (text) {
  const out = [];
  for (let i = 0; i < text.length; i ++) {
    const code = text.charCodeAt(i);
    if (isPrintable(code) && text[i] !== "%") {
      out.push(text[i]);
    } else {
      out.push("%" + HEX_UPPER[code >> 4] + HEX_UPPER[code & 15]);
    }
  }
  return out.join("");
}

/**
 * Builds the transformed representation to be scored: the model-facing
 * content text (URL with encoded spans replaced by their unwrapped
 * content, re-escaped injectively below the top level) plus the raw
 * bytes routed to the 8-bits/byte binary channel (hex span contents).
 * @param {object[]} parts Transform tree
 * @param {boolean} [nested] Internal: escaping applies below top level
 * @returns {{text: string, bytes: number[]}} Content text and binary
 *  channel bytes
 */
export function buildContent (parts, nested = false) {
  let text = "";
  const bytes = [];
  for (const part of parts) {
    if (part.kind === "lit") {
      text += nested ? escapeContent(part.text) : part.text;
    } else if (part.parts) { // pct / text b64
      const inner = buildContent(part.parts, true);
      text += inner.text;
      bytes.push(...inner.bytes);
    } else { // hex / binary b64
      bytes.push(...part.bytes);
    }
  }
  return { text, bytes };
}

/** Appends an Elias gamma code for n >= 1 to a bit array. */
function pushGamma (bits, n) {
  let len = 0;
  for (let m = n; m > 1; m >>= 1) len ++;
  for (let i = 0; i < len; i ++) bits.push(0);
  for (let i = len; i >= 0; i --) bits.push((n >> i) & 1);
}

/**
 * Serializes a transform tree into the compact binary encoding whose
 * size we charge against the scheme: span counts, gap positions and
 * lengths as Elias gamma codes, 2-bit types, and per-type variant
 * metadata (percent escape masks fall back to 1 bit/char + 2 bits per
 * escape only when the canonical-encoder variants don't match).
 * @param {object[]} parts Transform tree
 * @returns {number[]} Serialized bits (0/1)
 */
export function serializeTree (parts) {
  const bits = [];
  const emit = (ps) => {
    const spans = ps.filter(p => p.kind === "span");
    pushGamma(bits, spans.length + 1);
    let prev = 0;
    for (const span of spans) {
      pushGamma(bits, span.start - prev + 1);
      const type = span.type === "pct" ? 0 : span.type === "b64" ? 1 : 2;
      bits.push(type >> 1, type & 1);
      if (span.type === "pct") {
        bits.push(span.variant >> 1, span.variant & 1);
        pushGamma(bits, span.decoded.length + 1);
        if (span.variant === 2) {
          for (const e of span.escapes) {
            bits.push(e < 0 ? 0 : 1);
            if (e >= 0) bits.push(e >> 1, e & 1);
          }
        }
        emit(span.parts);
      } else if (span.type === "b64") {
        bits.push(span.urlSafe ? 1 : 0);
        bits.push(span.pad >> 1, span.pad & 1);
        bits.push(span.binary ? 1 : 0);
        if (span.binary) {
          pushGamma(bits, span.bytes.length + 1);
        } else {
          pushGamma(bits, span.decoded.length + 1);
          emit(span.parts);
        }
      } else { // hex
        bits.push(span.upper ? 1 : 0);
        pushGamma(bits, span.bytes.length + 1);
      }
      prev = span.end;
    }
  };
  emit(parts);
  return bits;
}

/**
 * Returns a copy of the tree with binary-channel spans (hex and
 * binary base64) reverted to literals - used to score the scheme with
 * and without the binary channel, which only pays off when 8
 * bits/byte beat the model on the original span text.
 * @param {object[]} parts Transform tree
 * @returns {object[]} Tree without binary-channel spans
 */
export function stripBinary (parts) {
  return parts.map(part => {
    if (part.kind !== "span") return part;
    if (part.type === "hex" || (part.type === "b64" && part.binary)) {
      return { kind: "lit", text: part.raw };
    }
    if (part.parts) return { ...part, parts: stripBinary(part.parts) };
    return part;
  });
}
