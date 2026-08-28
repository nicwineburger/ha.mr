#!/usr/bin/env node
// Prose audit for the voice-print skill.
//
// Measures a text against the documented AI-register tells: the
// excess-vocabulary lists (Kobak et al. 2025, 15M PubMed abstracts),
// Wikipedia's "Signs of AI writing" patterns, and the burstiness
// (sentence-length variance) feature the detection literature weights
// most. Thresholds are heuristics tuned on known specimens, not
// science — and targets are "match the author's corpus baseline",
// not "score zero": run this on the author's own writing first.
//
// Word lists age as models drift (2023's "delve" faded; "showcasing"
// persisted) — re-tune them, don't trust them forever.
//
// Usage: node prose-audit.mjs <files...>   (.html is tag-stripped)
// Exit: 0 clean, 1 any file had STRONG findings, 2 usage error.

import { readFileSync } from 'node:fs'

const STRONG_LEX = [
  /\bdel(ve|ves|ving)\b/gi, /\btapestr(y|ies)\b/gi,
  /\bunderscor(e|es|ed|ing)\b/gi, /\bshowcas(e|es|ed|ing)\b/gi,
  /\bpivotal\b/gi, /\bmeticulous(ly)?\b/gi, /\bseamless(ly)?\b/gi,
  /\bintricate\b/gi, /\bintricacies\b/gi, /\bbolster(s|ed|ing)?\b/gi,
  /\bgarner(s|ed|ing)?\b/gi, /\bfoster(s|ed|ing)?\b/gi,
  /\belucidat(e|es|ed|ing)\b/gi, /\bvibrant\b/gi, /\bboasts?\b/gi,
  /\btestament to\b/gi, /\bgroundbreaking\b/gi, /\brenowned\b/gi,
  /\bnestled\b/gi, /\bin the heart of\b/gi, /\bevolving landscape\b/gi,
  /\brich tapestry\b/gi, /\bvaluable insights\b/gi, /\bdeep dive\b/gi,
  /\bgame-?changer\b/gi, /\btreasure trove\b/gi, /\bever-evolving\b/gi,
  /\bit(?:'|’)?s important to note\b/gi,
  /\bit is important to note\b/gi, /\bit(?:'|’)?s worth noting\b/gi,
  /\bin conclusion\b/gi, /\bin summary\b/gi, /\bmoreover\b/gi,
  /\bfurthermore\b/gi, /\bharness(es|ing)? the\b/gi,
  /\bunlock(s|ing)? the (power|potential)\b/gi
]

const WEAK_LEX = [
  /\bcrucial(ly)?\b/gi, /\bnotably\b/gi, /\bcomprehensive\b/gi,
  /\bremarkable\b/gi, /\binnovative\b/gi, /\badditionally\b/gi,
  /\brobust\b/gi, /\bleverag(e|es|ed|ing)\b/gi,
  /\bstreamlin(e|es|ed|ing)\b/gi, /\belevat(e|es|ed|ing)\b/gi,
  /\bempower(s|ed|ing)?\b/gi, /\bjourney\b/gi, /\brealm\b/gi,
  /\blandscape\b/gi, /\bholistic\b/gi, /\binsights\b/gi,
  /\bdive into\b/gi, /\bat its core\b/gi
]

const COPULA = /\b(serves|stands|functions)\s+as\b|\brepresents\s+a\b/gi
const NEG_PARALLEL = new RegExp(
  "\\bnot\\s+(just|only|merely|simply)\\b[^.!?\\n]{0,90}\\bbut\\b" +
  "|\\bisn(?:'|’)?t\\s+(just|only)\\b[^.!?\\n]{0,90}\\bit(?:'|’)?s\\b", 'gi')
const PARTICIPIAL_TAIL = new RegExp(
  ',\\s*(highlighting|underscoring|emphasizing|showcasing|reflecting' +
  '|demonstrating|ensuring|fostering|signaling|signalling|marking' +
  '|cementing|solidifying|paving the way)\\b', 'gi')
const TRIAD = /\b[\w'’-]+,\s+[\w'’-]+,?\s+and\s+[\w'’-]+/g
const BOLD_LEAD = /^[ \t]*[-*•][ \t]*\*\*[^*\n]{2,60}\*\*[ \t]*[:—–-]/gm
const CURLY = /[“”‘’]/g
const EM_DASH = /—/g

function stripHtml (raw) {
  return raw
    .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<(br|\/p|\/li|\/h[1-6]|\/div)[^>]*>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&(#38|amp);/g, '&').replace(/&(#39|apos);/g, "'")
    .replace(/&(#34|quot);/g, '"').replace(/&nbsp;/g, ' ')
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
}

function count (text, re) {
  const m = text.match(re)
  return m ? m.length : 0
}

// Sentence split requires two letters before the terminator so periods
// in versions ("v1.5.2") and initials ("e.g.", "U.S.") don't
// manufacture fake short sentences and inflate the variance we
// measure — and it must not require an uppercase follower, or
// all-lowercase informal prose would collapse into one sentence.
function sentences (text) {
  const out = []
  for (const para of text.split(/\n\s*\n|\n/)) {
    for (const s of para.split(/(?<=[A-Za-z]{2}[.!?…])\s+/)) {
      const words = s.split(/\s+/).filter(Boolean)
      if (words.length >= 2) out.push(words.length)
    }
  }
  return out
}

function cv (xs) {
  if (xs.length < 2) return NaN
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length)
  return sd / mean
}

function audit (file) {
  const raw = readFileSync(file, 'utf8')
  const text = /\.html?$/i.test(file) ? stripHtml(raw) : raw
  const words = text.split(/\s+/).filter(w => /\w/.test(w)).length
  if (words < 50) {
    console.log(`${file}: only ${words} words — too short to measure`)
    return 0
  }
  const per1k = n => (n * 1000) / words
  const sl = sentences(text)
  const slCV = cv(sl)
  const slMean = sl.reduce((a, b) => a + b, 0) / (sl.length || 1)
  const sum = res => res.reduce((a, re) => a + count(text, re), 0)

  const m = {
    strongLex: sum(STRONG_LEX), weakLex: sum(WEAK_LEX),
    copula: count(text, COPULA), negPar: count(text, NEG_PARALLEL),
    tail: count(text, PARTICIPIAL_TAIL), triad: count(text, TRIAD),
    emDash: count(text, EM_DASH), curly: count(text, CURLY),
    boldLead: count(raw, BOLD_LEAD)
  }

  const findings = []
  const flag = (sev, label, detail) => findings.push([sev, label, detail])

  if (m.strongLex >= 2 && per1k(m.strongLex) >= 1.5) {
    flag('STRONG', 'excess vocabulary', `x${m.strongLex} (${per1k(m.strongLex).toFixed(1)}/1k)`)
  } else if (m.strongLex >= 1) flag('WEAK', 'excess vocabulary', `x${m.strongLex}`)
  if (m.weakLex >= 3 && per1k(m.weakLex) >= 4) flag('WEAK', 'weak-tier vocabulary', `x${m.weakLex}`)
  if (m.copula >= 2 && per1k(m.copula) >= 1.2) {
    flag('STRONG', 'copula replacement (serves/stands as)', `x${m.copula}`)
  } else if (m.copula >= 1) flag('WEAK', 'copula replacement', `x${m.copula}`)
  if (m.negPar >= 2) flag('STRONG', 'negative parallelism (not just X, but Y)', `x${m.negPar}`)
  else if (m.negPar === 1) flag('WEAK', 'negative parallelism', 'x1')
  if (m.tail >= 2 && per1k(m.tail) >= 1) flag('STRONG', 'participial tails (, highlighting ...)', `x${m.tail}`)
  else if (m.tail >= 1) flag('WEAK', 'participial tails', `x${m.tail}`)
  if (per1k(m.emDash) > 3) flag('STRONG', 'em-dash density', `${per1k(m.emDash).toFixed(1)}/1k`)
  else if (per1k(m.emDash) > 1.5) flag('WEAK', 'em-dash density', `${per1k(m.emDash).toFixed(1)}/1k`)
  if (per1k(m.triad) >= 6) flag('WEAK', 'rule-of-three lists', `${per1k(m.triad).toFixed(1)}/1k`)
  if (sl.length >= 25 && slCV < 0.42) flag('STRONG', 'metronomic rhythm (sentence-length CV)', slCV.toFixed(2))
  else if (sl.length >= 25 && slCV < 0.52) flag('WEAK', 'low burstiness (sentence-length CV)', slCV.toFixed(2))
  if (m.boldLead >= 3) flag('STRONG', 'bold-lead bullet lists (**X:** ...)', `x${m.boldLead}`)
  if (!/\.html?$/i.test(file) && m.curly > 0) flag('WEAK', 'curly quotes in plain text', `x${m.curly}`)

  console.log(`\n${file} — ${words} words, ${sl.length} sentences` +
    ` (mean ${slMean.toFixed(1)}, CV ${Number.isNaN(slCV) ? 'n/a' : slCV.toFixed(2)})`)
  console.log('-------------------------------------------------------------')
  let strong = 0
  for (const [sev, label, detail] of findings) {
    if (sev === 'STRONG') strong++
    console.log(`${sev.padEnd(7)} ${label.padEnd(44)} ${detail}`)
  }
  const weak = findings.length - strong
  if (findings.length === 0) console.log('verdict: clean — no AI-register signature detected')
  else if (strong === 0 && weak < 3) console.log(`verdict: clean enough — ${weak} weak signals`)
  else if (strong === 0) console.log(`verdict: leaning AI register — ${weak} weak signals`)
  else console.log(`verdict: AI register detected — ${strong} strong, ${weak} weak`)
  return strong
}

const files = process.argv.slice(2)
if (files.length === 0) {
  console.error('usage: node prose-audit.mjs <text/markdown/html files...>')
  process.exit(2)
}
let worst = 0
for (const f of files) {
  try { worst = Math.max(worst, audit(f)) } catch (e) {
    console.error(`skip ${f}: ${e.message}`)
  }
}
process.exit(worst > 0 ? 1 : 0)
