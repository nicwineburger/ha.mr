/**
 * @file Hybrid compression: every link is encoded with the classic
 * dictionary/Huffman scheme and (when the model is available and the
 * link fits it) the neural coder - whichever payload is smaller wins.
 * The payload's version marker records which scheme was used, so
 * decoding is unambiguous and classic links from before the neural
 * coder existed keep working.
 */

import {
  compressToNumber,
  decompressNumber,
  numberToString,
  stringToNumber
} from "./compress.js";
import {
  neuralCompressToNumber,
  neuralDecompressNumber,
  payloadVersion
} from "./neural.js";

// The classic scheme's intentional normalizations, applied to both
// sides of the round-trip comparison in compressHybrid so that only
// real corruption differs: escapes of unreserved characters decode to
// the literal, all other escape hex is uppercased, stray "%" becomes
// "%25", valueless query parameters gain "=", and a bare trailing "?"
// is dropped. Escapes of reserved characters deliberately stay
// escapes - "%2F" and "/" are different URLs, and conflating them
// would hide exactly the corruption this check exists to catch.
const unreservedCharacter = /^[A-Za-z0-9\-_.!~*'()]$/;
function comparableLink (link) {
  const url = new URL(link);
  url.search = url.search.replace(/=(?=&|$)/g, "");
  let href = url.href;
  if (url.search === "") href = href.replace(/\?(?=#|$)/, "");
  return href
    .replace(/%(?![0-9a-fA-F]{2})/g, "%25")
    .replace(/%[0-9a-fA-F]{2}/g, (escape) => {
      const char = String.fromCharCode(parseInt(escape.slice(1), 16));
      return unreservedCharacter.test(char) ? char : escape.toUpperCase();
    });
}

/**
 * Compresses the input link with the best available scheme.
 * @param {string} input Link to compress
 * @param {string[]} alphabet Output alphabet as array of characters/strings
 * @param {URLModel?} model Loaded model, or null for classic-only
 * @param {{search?: boolean}} [options] Neural encoder options
 *  (see neuralCompressToNumber); omitted = full tokenization search
 * @param {{compress: (input: string, options?: object) => BigInt?}}
 *  [engine] Inference engine bound to `model` (see
 *  engine-select.js's selectEngine) - omitted defaults to the plain
 *  JS engine via neuralCompressToNumber, unchanged from before this
 *  parameter existed. Callers pass this to run the neural half of the
 *  hybrid scheme through WASM; payloads are bit-identical either way.
 * @returns {string} Output payload (not a full link!)
 */
export function compressHybrid (input, alphabet, model, options, engine = null) {
  // Neither scheme can represent non-http(s) protocols or credentials,
  // and each mangles them differently (the classic coder rewrites the
  // protocol and drops credentials; the neural coder keeps credentials
  // the classic one drops). Reject such input up front rather than
  // issue a link that points somewhere else. A missing protocol is
  // tolerated - both schemes already assume "http://" for it.
  const hasProtocol = /^\w+:\/\//.test(input);
  const url = new URL(hasProtocol ? input : "http://" + input);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Unsupported protocol "${url.protocol}" - only http and https links can be encoded`);
  }
  if (url.username || url.password) {
    throw new Error("Credentials in links are not supported");
  }
  // Either scheme may fail where the other succeeds (e.g. the classic
  // domain dictionary can't encode hostnames containing "_", which
  // are invalid DNS but do occur in the wild) - only fail if both do.
  let best = null;
  let classicError = null;
  try {
    best = compressToNumber(input);
    // The classic coder can't report an unrepresentable link - an
    // encoder bug just yields a payload that decodes to a different
    // URL. Verify the round-trip (up to the scheme's intentional
    // normalizations) so a corrupt candidate is discarded instead of
    // issued; the neural scheme, when available, still covers the link.
    if (comparableLink(decompressNumber(best)) !== comparableLink(url.href)) {
      best = null;
      classicError = new Error("Classic encoding failed round-trip verification");
    }
  } catch (e) {
    // A decode so corrupt its URL doesn't parse lands here with best
    // already assigned - discard it along with ordinary failures
    best = null;
    classicError = e;
  }
  if (model) {
    try {
      const neural = engine
        ? engine.compress(input, options)
        : neuralCompressToNumber(model, input, options);
      // Smaller payload number = same or fewer output symbols
      if (neural !== null && (best === null || neural < best)) best = neural;
    } catch (e) {
      console.warn("Neural compression failed, using classic:", e);
    }
  }
  if (best === null) throw classicError;
  return numberToString(best, alphabet);
}

/**
 * Decompresses a payload produced by any scheme version.
 * @param {string} payload Compressed payload
 * @param {string[]} alphabet Ordered alphabet used by payload
 * @param {URLModel?} model Loaded model, or null for classic-only
 * @param {{decompress: (number: BigInt) => string}} [engine] Inference
 *  engine bound to `model` (see engine-select.js's selectEngine) -
 *  omitted defaults to the plain JS engine via
 *  neuralDecompressNumber, unchanged from before this parameter
 *  existed.
 * @returns {string} Full link containing payload contents.
 */
export function decompressHybrid (payload, alphabet, model, engine = null) {
  const number = stringToNumber(payload, alphabet);
  const version = payloadVersion(number);
  if (version === 0) return decompressNumber(number);
  // Neural payloads name the model that made them; the caller is
  // responsible for loading the matching model file (the latest one
  // for current links, model/url-model-v<N>.bin for older versions)
  if (!model) {
    throw `This link requires model version ${version} to decode.`;
  }
  return engine ? engine.decompress(number) : neuralDecompressNumber(model, number);
}

/**
 * Reads which scheme a payload was encoded with.
 * @param {string} payload Compressed payload
 * @param {string[]} alphabet Ordered alphabet used by payload
 * @returns {number} Version (0 = classic, 1 = neural)
 */
export function payloadSchemeVersion (payload, alphabet) {
  return payloadVersion(stringToNumber(payload, alphabet));
}
