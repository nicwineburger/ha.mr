---
name: voice-print
description: Use whenever writing user-facing prose for a project — site copy, About pages, changelogs, blog posts, READMEs, error messages, marketing text. Replaces the default "AI register" (excess-vocabulary words, negative parallelisms, participial tails, em-dash density, metronomic sentence rhythm) with a voice derived from the author's own writing corpus, then measures the result with the bundled prose-audit.mjs. Not for code, and not for matching an existing editorial style guide — follow that instead.
---

# Voice Print

Unconstrained prose sampling lands in the model's modal register, and
that register is now empirically mapped: the excess-vocabulary study
of 15M PubMed abstracts (Kobak et al., Science Advances 2025)
isolated 379 style words whose frequency jumped when LLM assistance
arrived, and Wikipedia's editor-curated "Signs of AI writing" page
documents the phrase-level and structural patterns. This skill turns
those findings into (a) a writing procedure that derives voice from
the human author instead, and (b) `prose-audit.mjs`, which measures
a draft against the documented tells.

Two failure modes to avoid from the start. **Caricature**: asking for
"more human" writing produces fake typos and forced slang, which is
its own fingerprint — imperfection must be *sampled from the author's
corpus, never invented*. **Word-swapping**: replacing flagged words
with synonyms leaves rhythm, structure, and framing intact — the
detection literature's paraphrase-attack results show that only
restructuring at the sentence level moves the signal.

## Procedure — profile, write, restructure, audit

**1. Profile the author first.** Collect 1,000–5,000 words of their
real writing (old READMEs, blog posts, long messages — pre-AI if
possible) and run the audit on it:

```
node .claude/skills/voice-print/prose-audit.mjs corpus.txt
```

The corpus's numbers ARE the voice print: its em-dash and semicolon
rates, sentence-length mean and variance, paragraph rhythm,
contraction rate, how it opens sentences, what it does instead of
transitions. Targets are "match the corpus", not "score zero" — an
author who genuinely writes em-dashes keeps them.

**2. Write to the profile under hard bans.** The bans (below) are the
documented tells; everything else follows the corpus. Two rules do
most of the work: lead with the particular, not the category (the
model's instinct is category-first: "a comprehensive guide to X" —
the author's version names the thing); and put one detail per section
that only this author could know — a real date, a real complaint, a
real reader. Genericity is the deepest tell and the one no word list
catches.

**3. Revise by restructuring, not swapping.** On a revision pass,
change sentence *shapes*: merge two, split one, reorder a paragraph,
turn a triad into one item, delete a topic sentence. Follow the
corpus's burstiness — if it mixes 40-word sentences with 4-word ones,
do that; the model's uniform 15–25-word rhythm is measurable
(sentence-length CV is in the audit).

**4. Place human ink where eyes land.** The author personally
rewrites the sampling points: the headline, the first paragraph, the
About page, one anecdote per major page. Readers sample; they don't
audit. This is the only step that breaks the fixed point of a model
editing its own tells.

**5. Audit and iterate against the corpus baseline**, not against
zero. Fix STRONG findings or argue for them (the escape hatch below).

## The tells (what the audit measures)

Grounded in the excess-vocabulary list, the Wikipedia signs page, and
the burstiness feature of the detection literature. Era note: the
excess vocabulary drifts as models change (2023's "delve/tapestry"
faded; "showcasing/highlighting" persisted) — the word lists in the
audit are a snapshot to re-tune, not scripture.

- **Excess vocabulary.** delve, tapestry, underscore, showcase,
  pivotal, meticulous, seamless, intricate, bolster, garner, foster,
  vibrant, boasts, testament, groundbreaking, renowned, nestled,
  "evolving landscape", "rich tapestry", "valuable insights", "deep
  dive", "game-changer", "it's important to note", "in conclusion",
  moreover, furthermore. Weaker but counted: crucial, notably,
  comprehensive, remarkable, innovative, additionally.
- **Copula replacement.** "serves as", "stands as", "functions as",
  "represents a", "boasts" where the author would write "is".
- **Negative parallelism.** "not just X, but Y", "isn't only X — it's
  Y". One is rhetoric; a pattern is a fingerprint.
- **Participial tails.** Sentences that end ", highlighting its…",
  ", underscoring the…", ", ensuring that…" — significance-padding
  bolted onto a fact.
- **Rule of three.** Triad lists ("X, Y, and Z") at a rate no human
  sustains.
- **Punctuation register.** Em-dash density; curly quotes/apostrophes
  in hand-typed-looking web copy; flawless punctuation with zero
  variance as an *absence* tell.
- **Rhythm.** Low sentence-length variance (metronomic 15–25-word
  sentences) and uniform paragraph sizes — the burstiness feature
  detectors weight most.
- **Formatting reflexes.** Bold-lead bullet lists ("**X:** …"),
  Title Case headings, "X and Y"-shaped section headers, a summary
  paragraph that restates the section it ends.

## Escape hatch

Any tell may stand when the author's corpus shows they genuinely
write that way — the baseline run proves it. The offense is the
model's register leaking through, not the constructions themselves.

## Honest limits

This targets the casual read, not forensic stylometry — determined
analysis with a corpus wins regardless, and the detection literature
cuts both ways: detectors are brittle, and their false positives
concentrate on formal and non-native writers, so "scores clean" and
"reads human" are neither necessary nor sufficient for each other.
Word lists age as models drift. And a model applying this skill is
still a model — step 4 is load-bearing, not optional.

## When not to use

An editorial style guide or house voice exists — follow it. The text
is meant to be plainly AI-authored (disclosed assistant replies,
generated reference docs) — don't disguise those. Code comments are
`divergent-design`/engineering territory, not this skill's.
