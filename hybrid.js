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

/**
 * Compresses the input link with the best available scheme.
 * @param {string} input Link to compress
 * @param {string[]} alphabet Output alphabet as array of characters/strings
 * @param {URLModel?} model Loaded model, or null for classic-only
 * @returns {string} Output payload (not a full link!)
 */
export function compressHybrid (input, alphabet, model) {
  let best = compressToNumber(input);
  if (model) {
    try {
      const neural = neuralCompressToNumber(model, input);
      // Smaller payload number = same or fewer output symbols
      if (neural !== null && neural < best) best = neural;
    } catch (e) {
      // The classic result is always available as a fallback
      console.warn("Neural compression failed, using classic:", e);
    }
  }
  return numberToString(best, alphabet);
}

/**
 * Decompresses a payload produced by any scheme version.
 * @param {string} payload Compressed payload
 * @param {string[]} alphabet Ordered alphabet used by payload
 * @param {URLModel?} model Loaded model, or null for classic-only
 * @returns {string} Full link containing payload contents.
 */
export function decompressHybrid (payload, alphabet, model) {
  const number = stringToNumber(payload, alphabet);
  const version = payloadVersion(number);
  if (version === 0) return decompressNumber(number);
  if (version === 1) {
    if (!model) throw "This link requires the model file to decode.";
    return neuralDecompressNumber(model, number);
  }
  throw `Unsupported payload version: ${version}.`;
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
