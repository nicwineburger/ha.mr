/**
 * @file Evaluation driver for the transform-search prototype.
 *
 * Usage:
 *   node run.mjs score <urls-file> [--limit N] [--offset N] [--out f.json]
 *   node run.mjs incidence <urls-file> [--sample N] [--seed N]
 *     [--out f.json] [--detected-out urls.txt]
 *   node run.mjs entropy   (base64-of-random-bytes bits/char check)
 *
 * The urls-file must be the first argument after the mode. `score`
 * runs the full accounting (real coded bits via the shipped model)
 * over a holdout file; `incidence` runs detection + inversion
 * verification only (no model) over a deterministic seeded sample of
 * a larger corpus slice; `entropy` measures what the model spends on
 * base64 of random bytes - the naive expectation is 6 bits/char
 * (break-even with an 8-bits/byte binary channel), and anything above
 * that is headroom for unwrapping binary base64.
 */

import { readFile, writeFile } from "node:fs/promises";
import { URLModel } from "../../../neural.js";
import { treeSpanCounts, verifyInvertible } from "./transforms.mjs";
import { codeTextBits, scoreURL } from "./score.mjs";

const mode = process.argv[2];
const args = process.argv.slice(3);
const positional = args.filter(a => !a.startsWith("--"));
const flag = (name, dflt) => {
  const i = args.indexOf("--" + name);
  return i === -1 ? dflt : args[i + 1];
};

async function loadModel () {
  const buffer = await readFile(
    new URL("../../url-model.bin", import.meta.url));
  return new URLModel(buffer.buffer);
}

async function loadURLs (file) {
  return (await readFile(file, "utf8"))
    .split("\n").map(u => u.trim()).filter(Boolean);
}

/** Deterministic PRNG (mulberry32) for seeded sampling. */
function mulberry32 (seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile (sorted, p) {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1,
    Math.floor(p / 100 * sorted.length))];
}

if (mode === "score") {
  const urls = await loadURLs(positional[0]);
  const offset = parseInt(flag("offset", "0"), 10);
  const limit = parseInt(flag("limit", String(urls.length)), 10);
  const slice = urls.slice(offset, offset + limit);
  const model = await loadModel();

  const out = {
    n: 0, skipped: 0, inversionFailures: 0,
    sumCurrent: 0, sumScheme: 0, sumChunked: 0, sumCombined: 0,
    detectedURLs: 0, transformedURLs: 0,
    byType: {}, // type -> { urls, wins, winSum }
    wins: [], // per affected URL: current - transformed
    affectedDetails: [],
    b64bin: 0,
    examples: [],
    failures: []
  };
  const t0 = performance.now();
  for (const raw of slice) {
    const r = scoreURL(model, raw);
    if (r.skip) {
      out.skipped ++;
      continue;
    }
    out.n ++;
    out.b64bin += r.b64bin || 0;
    const types = Object.keys(r.detected || {});
    if (types.length) out.detectedURLs ++;
    if (r.inversionFailure) {
      out.inversionFailures ++;
      out.failures.push(r.url);
      // Excluded from claimed wins: charge the current payload
      out.sumCurrent += r.current;
      out.sumScheme += r.current;
      out.sumChunked += Math.min(r.current, r.chunked ?? Infinity);
      out.sumCombined += Math.min(r.current, r.chunked ?? Infinity);
      continue;
    }
    const transformed = r.transformed ? r.transformed.symbols : null;
    const scheme = Math.min(r.current, transformed ?? Infinity);
    const chunked = Math.min(r.current, r.chunked ?? Infinity);
    out.sumCurrent += r.current;
    out.sumScheme += scheme;
    out.sumChunked += chunked;
    out.sumCombined += Math.min(scheme, chunked);
    if (transformed !== null) {
      out.transformedURLs ++;
      const win = r.current - transformed;
      out.wins.push(win);
      if (out.affectedDetails.length < 500) {
        out.affectedDetails.push({
          url: r.url, win, current: r.current, transformed,
          chunked: r.chunked, detected: r.detected,
          usedBinaryChannel: r.transformed.usedBinaryChannel
        });
      }
      for (const type of types) {
        const t = out.byType[type] ||= { urls: 0, wins: 0, winSum: 0 };
        t.urls ++;
        if (win > 0) t.wins ++;
        t.winSum += win;
      }
      if (win > 0) {
        out.examples.push({
          url: r.url, win, current: r.current, transformed,
          chunked: r.chunked, detected: r.detected,
          content: r.transformed.contentText,
          treeBits: r.transformed.treeBits,
          contentBits: r.transformed.contentBits,
          binaryBytes: r.transformed.binaryBytes
        });
        out.examples.sort((a, b) => b.win - a.win);
        out.examples.length = Math.min(out.examples.length, 12);
      }
    } else if (types.length) {
      // Detected but not scoreable (model can't code the content)
      for (const type of types) {
        const t = out.byType[type] ||= { urls: 0, wins: 0, winSum: 0 };
        t.urls ++;
      }
    }
    if (out.n % 200 === 0) {
      console.error(`  ${out.n}/${slice.length} ` +
        `(${((performance.now() - t0) / 1000).toFixed(0)}s)`);
    }
  }

  const wins = out.wins.slice().sort((a, b) => a - b);
  const affected = wins.length;
  const positive = wins.filter(w => w > 0).length;
  const report = {
    n: out.n,
    skipped: out.skipped,
    inversionFailures: out.inversionFailures,
    avgCurrent: out.sumCurrent / out.n,
    avgScheme: out.sumScheme / out.n,
    avgChunked: out.sumChunked / out.n,
    avgCombined: out.sumCombined / out.n,
    schemeGainPct: 100 * (1 - out.sumScheme / out.sumCurrent),
    chunkedGainPct: 100 * (1 - out.sumChunked / out.sumCurrent),
    combinedGainPct: 100 * (1 - out.sumCombined / out.sumCurrent),
    detectedURLs: out.detectedURLs,
    transformedURLs: out.transformedURLs,
    affected,
    positiveWins: positive,
    falsePositiveRate: affected ? (affected - positive) / affected : null,
    winMean: affected ? wins.reduce((a, b) => a + b, 0) / affected : null,
    winMedian: percentile(wins, 50),
    winP90: percentile(wins, 90),
    positiveWinMean: positive
      ? wins.filter(w => w > 0).reduce((a, b) => a + b, 0) / positive : null,
    byType: out.byType,
    b64bin: out.b64bin,
    wins: out.wins,
    affectedDetails: out.affectedDetails,
    examples: out.examples,
    failures: out.failures.slice(0, 20)
  };
  console.log(JSON.stringify(report, null, 2));
  const outFile = flag("out", null);
  if (outFile) await writeFile(outFile, JSON.stringify(report));
} else if (mode === "incidence") {
  const urls = await loadURLs(positional[0]);
  const sampleSize = Math.min(
    parseInt(flag("sample", "50000"), 10), urls.length);
  const seed = parseInt(flag("seed", "42"), 10);
  const rand = mulberry32(seed);
  const chosen = new Set();
  while (chosen.size < sampleSize) {
    chosen.add(Math.floor(rand() * urls.length));
  }

  const out = {
    n: 0, unparsable: 0, inversionFailures: 0,
    detectedURLs: 0, byType: {}, b64binURLs: 0, spanCounts: {}
  };
  const detected = []; // URLs with transforms, for follow-up scoring
  for (const i of chosen) {
    const raw = urls[i];
    const link = /^https?:\/\//.test(raw) ? raw : "https://" + raw;
    if (!URL.canParse(link)) {
      out.unparsable ++;
      continue;
    }
    const url = new URL(link);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      out.unparsable ++;
      continue;
    }
    const text = url.href.replace(/^https?:\/\//, "");
    out.n ++;
    const stats = {};
    const { parts, ok, hasTransforms } = verifyInvertible(text, { stats });
    if (hasTransforms && !ok) {
      out.inversionFailures ++;
      continue;
    }
    if (stats.b64bin) out.b64binURLs ++;
    if (!hasTransforms) continue;
    out.detectedURLs ++;
    detected.push(raw);
    const counts = treeSpanCounts(parts);
    for (const [type, count] of Object.entries(counts)) {
      out.byType[type] = (out.byType[type] || 0) + 1;
      out.spanCounts[type] = (out.spanCounts[type] || 0) + count;
    }
  }
  const report = {
    sample: sampleSize, seed, ...out,
    detectedPct: 100 * out.detectedURLs / out.n,
    byTypePct: Object.fromEntries(Object.entries(out.byType)
      .map(([t, c]) => [t, 100 * c / out.n]))
  };
  console.log(JSON.stringify(report, null, 2));
  const outFile = flag("out", null);
  if (outFile) await writeFile(outFile, JSON.stringify(report));
  const detectedOut = flag("detected-out", null);
  if (detectedOut) await writeFile(detectedOut, detected.join("\n") + "\n");
} else if (mode === "entropy") {
  // Base64 of random bytes through the model: confirms ~6 bits/char,
  // i.e. ~8 bits per underlying byte - unwrapping buys nothing
  const model = await loadModel();
  const rand = mulberry32(7);
  let chars = 0;
  let bits = 0;
  for (let i = 0; i < 40; i ++) {
    const bytes = Buffer.from(
      Array.from({ length: 33 }, () => Math.floor(rand() * 256)));
    const span = bytes.toString("base64");
    const coded = codeTextBits(model, span);
    if (!coded) continue;
    chars += span.length;
    bits += coded.bits;
  }
  console.log(`base64(random bytes) through the model: ` +
    `${(bits / chars).toFixed(2)} bits/char over ${chars} chars ` +
    `(binary channel: 6.00 bits/char; anything above that is ` +
    `headroom for unwrapping binary base64)`);
} else {
  console.error("Usage: node run.mjs score|incidence|entropy <urls-file>");
  process.exit(1);
}
