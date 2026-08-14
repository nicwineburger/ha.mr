/**
 * @file Binary arithmetic coder driven by a probability model.
 *
 * Adapted from the neural-compression experiment in
 * https://github.com/ansisg/hamr - the coder follows the classic
 * 64-bit integer arithmetic coding scheme (Witten/Neal/Cleary style
 * with underflow handling), with probabilities quantized to 32-bit
 * integer frequencies so that encode and decode stay bit-exact.
 *
 * The probability model is a callback: `probabilities(context)` takes
 * the symbol ids seen so far and returns an array of probabilities
 * (any non-negative values; they're normalized by quantization). The
 * caller must guarantee the callback is deterministic - the decoder
 * reconstructs the exact same probabilities to undo the encoding.
 */

const PRECISION = 64n;
const MAX_RANGE = 1n << PRECISION;
const HALF = MAX_RANGE >> 1n;
const QUARTER = HALF >> 1n;
const THREE_QUARTER = QUARTER * 3n;
const FREQ_TOTAL = 1n << 32n;

/**
 * Quantizes a probability distribution into a cumulative frequency
 * table. Every symbol gets a frequency of at least 1 so it stays
 * encodable; the rounding remainder is assigned to the most likely
 * symbol.
 * @param {Float64Array|number[]} probs Symbol probabilities
 * @returns {BigInt[]} Cumulative frequencies, length `probs.length + 1`
 */
function buildCDF (probs) {
  const freqs = new Array(probs.length);
  let sum = 0n;
  let largest = 0;
  for (let i = 0; i < probs.length; i ++) {
    const f = Math.floor(probs[i] * 4294967296);
    freqs[i] = f > 1 ? BigInt(f) : 1n;
    sum += freqs[i];
    if (freqs[i] > freqs[largest]) largest = i;
  }
  freqs[largest] += FREQ_TOTAL - sum;

  const cdf = new Array(freqs.length + 1);
  cdf[0] = 0n;
  for (let i = 0; i < freqs.length; i ++) {
    cdf[i + 1] = cdf[i] + freqs[i];
  }
  return cdf;
}

/**
 * Encodes a symbol sequence into bits using the given model.
 * @param {number[]} symbols Symbol ids to encode, in order
 * @param {(context: number[]) => Float64Array} probabilities
 *  Model callback: probabilities of the next symbol given a context
 * @returns {number[]} Encoded bits (0/1), most significant first
 */
export function arithmeticEncode (symbols, probabilities) {
  let low = 0n;
  let high = MAX_RANGE - 1n;
  let pending = 0;
  const bits = [];

  const output = (bit) => {
    bits.push(bit);
    while (pending > 0) {
      bits.push(1 - bit);
      pending --;
    }
  };

  const context = [];
  for (const symbol of symbols) {
    const cdf = buildCDF(probabilities(context));
    context.push(symbol);

    const span = high - low + 1n;
    high = low + (span * cdf[symbol + 1]) / FREQ_TOTAL - 1n;
    low = low + (span * cdf[symbol]) / FREQ_TOTAL;

    while (true) {
      if (high < HALF) {
        output(0);
      } else if (low >= HALF) {
        output(1);
        low -= HALF;
        high -= HALF;
      } else if (low >= QUARTER && high < THREE_QUARTER) {
        pending ++;
        low -= QUARTER;
        high -= QUARTER;
      } else {
        break;
      }
      low <<= 1n;
      high = (high << 1n) | 1n;
    }
  }

  // Flush enough bits to disambiguate the final interval
  pending ++;
  output(low < QUARTER ? 0 : 1);

  return bits;
}

/**
 * Decodes bits back into symbols using the given model. Decoding stops
 * when `isTerminal` returns true for a decoded symbol.
 * @param {number[]} bits Encoded bits (0/1), most significant first
 * @param {(context: number[]) => Float64Array} probabilities
 *  The same model callback used to encode
 * @param {(symbol: number) => boolean} isTerminal Stop condition
 * @param {number} maxSymbols Safety limit on output length
 * @returns {number[]} Decoded symbol ids, including the terminal symbol
 */
export function arithmeticDecode (bits, probabilities, isTerminal, maxSymbols) {
  let low = 0n;
  let high = MAX_RANGE - 1n;
  let value = 0n;
  let bitIndex = 0;

  const readBit = () => bitIndex < bits.length ? bits[bitIndex ++] : 0;

  for (let i = 0n; i < PRECISION; i ++) {
    value = (value << 1n) | BigInt(readBit());
  }

  const symbols = [];
  while (true) {
    if (symbols.length >= maxSymbols) {
      throw `Arithmetic decode exceeded ${maxSymbols} symbols.`;
    }
    const cdf = buildCDF(probabilities(symbols));

    const span = high - low + 1n;
    const scaled = ((value - low + 1n) * FREQ_TOTAL - 1n) / span;

    // Binary search: cdf[symbol] <= scaled < cdf[symbol + 1]
    let lo = 0;
    let hi = cdf.length - 2;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid + 1] <= scaled) lo = mid + 1;
      else hi = mid;
    }
    const symbol = lo;
    symbols.push(symbol);
    if (isTerminal(symbol)) break;

    high = low + (span * cdf[symbol + 1]) / FREQ_TOTAL - 1n;
    low = low + (span * cdf[symbol]) / FREQ_TOTAL;

    while (true) {
      if (high < HALF) {
        // Nothing to adjust
      } else if (low >= HALF) {
        low -= HALF;
        high -= HALF;
        value -= HALF;
      } else if (low >= QUARTER && high < THREE_QUARTER) {
        low -= QUARTER;
        high -= QUARTER;
        value -= QUARTER;
      } else {
        break;
      }
      low <<= 1n;
      high = (high << 1n) | 1n;
      value = (value << 1n) | BigInt(readBit());
    }
  }

  return symbols;
}
