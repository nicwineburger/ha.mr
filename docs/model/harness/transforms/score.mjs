/**
 * @file Honest payload accounting for the transform-search scheme.
 *
 * For each URL we compare:
 *   - current:     the shipped hybrid payload (compressHybrid), in
 *                  output-alphabet symbols
 *   - chunked:     a hypothetical "next version" that codes the
 *                  UNtransformed URL text with the shipped model but
 *                  splits it into context-sized chunks (so URLs longer
 *                  than the model context stop falling back to
 *                  classic) - this isolates how much of any win comes
 *                  from chunking alone rather than from transforms
 *   - transformed: the transform scheme - serialized transform tree
 *                  + chunk-coded content text + 8 bits/byte binary
 *                  channel, all packed into a real payload number and
 *                  rendered in the output alphabet
 *
 * Content bits are REAL arithmetic-coded bits from the shipped model
 * (the same coder path the site uses), not -log2 estimates. The
 * payload number mirrors the shipped format: sentinel bit, coded
 * bits, isHTTPS bit, and a unary version marker one past the current
 * model's linkVersion (what a shipped transform scheme would pay).
 */

import { arithmeticEncode } from "../../../arithmetic-coder.js";
import { compressToNumber, numberToString } from "../../../compress.js";
import { neuralCompressToNumber } from "../../../neural.js";
import { outputAlphabetASCII } from "../../../alphabets.js";
import {
  buildContent,
  parseText,
  reconstruct,
  serializeTree,
  stripBinary,
  treeHasSpans,
  treeSpanCounts
} from "./transforms.mjs";

/*
 * detExp/softmax duplicate neural.js internals (they are not
 * exported, and the shipped file must not change for a harness
 * experiment). Bit-exactness across engines is irrelevant here - the
 * coder only needs to be self-consistent within this measurement.
 */
const LN2 = 0.6931471805599453;
const INV_LN2 = 1.4426950408889634;
const POW2 = new Float64Array(1101);
POW2[0] = 1;
for (let i = 1; i < POW2.length; i ++) POW2[i] = POW2[i - 1] * 0.5;

function detExp (x) {
  if (x < -708) return 0;
  const n = Math.floor(x * INV_LN2 + 0.5);
  const r = x - n * LN2;
  let term = 1;
  let sum = 1;
  for (let i = 1; i <= 13; i ++) {
    term = term * r / i;
    sum += term;
  }
  return POW2[-n] * sum;
}

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

const EOS = 0;

/** Probability callback over a fresh model session (same incremental
 * feeding contract as neural.js modelProbabilities). */
function modelProbabilities (model) {
  const session = model.session();
  let fed = 0;
  return (context) => {
    let logits = null;
    if (fed === 0) {
      logits = session.feed(EOS);
      fed = 1;
    }
    while (fed <= context.length) {
      logits = session.feed(context[fed - 1]);
      fed ++;
    }
    if (logits === null) throw "Context out of sync with model session.";
    const probs = logits.slice();
    softmax(probs);
    return probs;
  };
}

/**
 * Codes text with the model in real arithmetic-coded bits, splitting
 * into independent EOS-terminated chunks when the text exceeds the
 * model context (each chunk restarts the context; the EOS terminators
 * make the chunking self-framing, at a few bits per chunk).
 * @param {URLModel} model Loaded model
 * @param {string} text Content text (printable ASCII)
 * @returns {{bits: number, chunks: number}?} Total coded bits and
 *  chunk count, or null if the model can't represent the text
 */
export function codeTextBits (model, text) {
  const symbols = model.tokenize(text);
  if (symbols === null) return null;
  const chunkSize = model.maxLen - 2; // Priming EOS + terminating EOS
  let bits = 0;
  let chunks = 0;
  for (let i = 0; i < symbols.length || chunks === 0; i += chunkSize) {
    const chunk = symbols.slice(i, i + chunkSize);
    chunk.push(EOS);
    bits += arithmeticEncode(chunk, modelProbabilities(model)).length;
    chunks ++;
  }
  return { bits, chunks };
}

/**
 * Packs coded bits into a payload number the way the shipped codecs
 * do and measures its rendered size in output-alphabet symbols.
 * @param {number[][]} bitArrays Bit arrays to concatenate
 * @param {number} extraBits Number of additional (binary channel)
 *  bits, charged as 8 bits/byte; modeled as literal bits
 * @param {boolean} isHTTPS Scheme bit
 * @param {number} version Unary version marker value
 * @returns {number} Payload length in outputAlphabetASCII symbols
 */
export function payloadSymbols (bitArrays, extraBits, isHTTPS, version) {
  let number = 1n; // Sentinel preserves leading zero bits
  for (const bits of bitArrays) {
    for (const bit of bits) number = (number << 1n) | BigInt(bit);
  }
  number <<= BigInt(extraBits); // Binary channel: exact bit count
  number = (number << 1n) | (isHTTPS ? 1n : 0n);
  const v = BigInt(version);
  number = (number << (v + 1n)) | ((1n << v) - 1n);
  return numberToString(number, outputAlphabetASCII).length;
}

/**
 * Scores one URL: current hybrid payload vs chunked-neural baseline
 * vs the transform scheme.
 * @param {URLModel} model Loaded model
 * @param {string} raw URL (scheme optional)
 * @returns {object} Per-URL result record
 */
export function scoreURL (model, raw) {
  const link = /^https?:\/\//.test(raw) ? raw : "https://" + raw;
  if (!URL.canParse(link)) return { skip: "unparsable" };
  const url = new URL(link);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { skip: "scheme" };
  }
  const isHTTPS = url.protocol === "https:";
  const text = url.href.slice(isHTTPS ? 8 : 7);
  const version = model.linkVersion + 1; // What a shipped scheme would pay

  // Current hybrid = min(classic, neural), computed separately so the
  // neural payload's coded bit count can be reused for the chunked
  // baseline instead of running the model twice
  let classicNum = null;
  let neuralNum = null;
  try { classicNum = compressToNumber(link); } catch {}
  try { neuralNum = neuralCompressToNumber(model, link); } catch {}
  if (classicNum === null && neuralNum === null) {
    return { skip: "hybrid-failure" };
  }
  const lengths = [classicNum, neuralNum].filter(n => n !== null)
    .map(n => numberToString(n, outputAlphabetASCII).length);
  const current = Math.min(...lengths);

  const result = { url: link, text, current, chars: text.length };

  // Chunked-neural baseline on the untransformed text (empty tree).
  // When the URL fits the model context the shipped neural payload
  // already contains the coded bits: strip sentinel, isHTTPS, and the
  // unary marker to recover the count. Otherwise chunk-code for real.
  let baseCoded = null;
  if (neuralNum !== null) {
    const bitLen = neuralNum.toString(2).length;
    baseCoded = { bits: bitLen - 3 - model.linkVersion, chunks: 1 };
  } else {
    baseCoded = codeTextBits(model, text);
  }
  if (baseCoded) {
    result.chunked = payloadSymbolsFor(
      baseCoded, serializeTree([]), 0, isHTTPS, version);
  }

  const stats = {};
  const parts = parseText(text, { stats });
  result.detected = treeSpanCounts(parts);
  result.b64bin = stats.b64bin || 0;
  if (!treeHasSpans(parts)) return result;

  if (reconstruct(parts) !== text) {
    result.inversionFailure = true;
    return result;
  }

  // Score the full tree, and (when binary-channel spans exist) the
  // tree without them - the channel only pays where 8 bits/byte beat
  // the model on the original span text
  const hasBinarySpan = (ps) => ps.some(p => p.kind === "span" &&
    (p.type === "hex" || (p.type === "b64" && p.binary) ||
     (p.parts && hasBinarySpan(p.parts))));
  const binaryPresent = hasBinarySpan(parts);
  const variants = [{ tree: parts, binary: binaryPresent }];
  if (binaryPresent) {
    const stripped = stripBinary(parts);
    if (treeHasSpans(stripped)) {
      variants.push({ tree: stripped, binary: false });
    }
  }
  let best = null;
  for (const { tree, binary } of variants) {
    const content = buildContent(tree);
    const coded = codeTextBits(model, content.text);
    if (!coded) continue;
    const treeBits = serializeTree(tree);
    const symbols = payloadSymbolsFor(
      coded, treeBits, content.bytes.length, isHTTPS, version);
    if (best === null || symbols < best.symbols) {
      best = {
        symbols,
        contentText: content.text,
        contentBits: coded.bits,
        treeBits: treeBits.length,
        binaryBytes: content.bytes.length,
        usedBinaryChannel: binary
      };
    }
  }
  if (best) result.transformed = best;
  return result;
}

/** Assembles content + tree + binary channel into rendered symbols. */
function payloadSymbolsFor (coded, treeBits, binaryBytes, isHTTPS, version) {
  // Content bits are a real bit array only inside codeTextBits; the
  // count is what matters for size, so charge them as literal bits
  return payloadSymbols([treeBits], coded.bits + 8 * binaryBytes,
    isHTTPS, version);
}
