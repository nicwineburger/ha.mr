# Transform search: measurement report

Would a "structural transform" payload scheme - detect encoded
substructure in URLs (percent-encoded nested URLs, base64 text/JSON,
JWTs, hex ids), unwrap it invertibly, and compress the unwrapped form
with the existing model - be worth shipping as a new payload version?

**Answer: no, not with the current model.** Detection works and
inversion is byte-exact (zero failures across 54,000 URLs), but the
v2 model already codes encoded substructure *as it appears in real
URLs* well enough that unwrapping usually costs bits instead of saving
them. Corpus-level gain on the shared 4,000-URL holdout: **+0.23%**,
and more than half of even that is captured by plain chunked coding of
long URLs with no transforms at all (+0.27% on its own). Details and
the one genuinely promising follow-up below.

## What was measured

Everything is real, end-to-end accounting - no `-log2` estimates:

- **Content bits**: the transformed representation is coded with the
  shipped v2 model through the real arithmetic coder (`score.mjs`
  reuses `URLModel` + `arithmeticEncode`). Text longer than the model
  context is split into independent EOS-terminated chunks
  (self-framing, each chunk restarts context).
- **Tree bits**: the transform tree is serialized to an actual bit
  string (`serializeTree`): Elias-gamma span counts/gaps/lengths,
  2-bit types, per-type variant flags, percent-escape masks
  (1 bit/char + 2 bits/escape) when the escape pattern isn't one of
  the two canonical-encoder variants.
- **Binary channel**: hex spans and binary base64 decode to raw bytes
  charged at 8 bits/byte.
- **Payload**: sentinel + content + tree + binary + isHTTPS + unary
  version marker (version 3, one past the current model), packed into
  a number and rendered in `outputAlphabetASCII` - the same path the
  shipped codecs use. Compared against the current hybrid payload
  (`min(classic, neural)`) per URL; scheme value =
  `min(current, transformed)`.
- **Attribution baseline**: a "chunked" variant that codes the
  *untransformed* URL the same way (empty tree). Long URLs fall back
  to classic today because they exceed the model context; chunking
  alone fixes that, so any win the transforms share with it is not a
  transform win.

The transformed representation replaces each encoded span with its
decoded content; below the top level, bytes outside printable ASCII
(and `%`) are re-escaped as `%XX` so the mapping stays injective and
model-representable. Every scored URL's tree is verified by replaying
the transforms forward (`reconstruct`) and asserting byte equality
with the original; failures would be excluded from claimed wins -
there were none.

## Incidence (50,000-URL seeded sample of the CC-derived slice)

| type | URLs affected | share |
|---|---|---|
| any transform | 1,065 | **2.13%** |
| base64 → binary | 557 | 1.11% |
| hex run (≥16 chars) | 291 | 0.58% |
| percent-encoded span | 211 | 0.42% |
| base64 → text/JSON | 32 | 0.06% |
| JWT | 0 | 0.00% |

Inversion failures: **0** of 50,000. Real machine-generated blobs are
mostly *binary* base64 (tokens, signatures) - base64 that decodes to
text or JSON, the case the transform idea is built around, occurs in
**0.06%** of URLs, and not a single JWT appeared in the sample.

## Holdout scoring (4,000 URLs, same set as the model campaigns)

| scheme | avg payload symbols | vs current |
|---|---|---|
| current hybrid | 19.48 | - |
| + transform scheme | 19.44 | **-0.23%** |
| + chunked coding only (no transforms) | 19.43 | -0.27% |
| + both | 19.43 | -0.30% |

- 97/4,000 URLs (2.4%) had a scoreable transform tree; **0 inversion
  failures, 0 skips**.
- Win distribution on affected URLs (payload symbols, positive =
  smaller): mean **-13.8**, median -12, p90 -2. Only **4 of 97** won
  at all - a **95.9% false-positive rate** (detected but no win).
  `min()` per URL keeps the losses out of the aggregate, but nearly
  all detections are noise.
- The 4 winners won big (median +69 symbols) - but they are long-blob
  URLs where chunked coding alone captures almost everything: against
  the chunked baseline their transform-specific margin is only +2 to
  +7 symbols.
- Per-type (URLs where only that type fired): hex 2/17 positive
  (mean +4.5, carried entirely by two big outliers); binary base64
  1/54 (mean -22); percent 0/25 (mean -9.6).

## Why unwrapping loses: the model already knows the encodings

The premise was that the model sees encoded spans as ~6 bits/char
noise. Measured reality:

- **Percent-encoded URLs are in-distribution.** The v2 model was
  trained on raw crawl URLs, which are full of
  `?url=https%3A%2F%2F...` patterns. Example (real holdout URL):
  `linkedin.com/login?session_redirect=https%3A%2F%2F...` codes to
  **17 symbols** as-is; unwrapped, **41** - the decoded form is
  *out* of distribution, and this span's `%2E` escapes (unreserved
  chars, so not a canonical-encoder pattern) force the per-character
  mask, bloating the tree.
- **JSON is out-of-distribution.** A textbook JWT URL (synthetic,
  since the corpus had none): 176 chars, current hybrid **173**
  symbols, chunked **157**, transformed **218**. The unwrapped JSON
  content cost **10.0 bits/char** through the URL model - far more
  than the ~6.5 bits/char of the base64 it replaced. Unwrapping JSON
  is a *loss* until a model is trained on unwrapped text.
- **Binary base64 has real but small headroom.** `run.mjs entropy`:
  the model spends **7.27 bits/char** on base64 of random bytes vs
  the binary channel's 6.00 (= 8 bits/byte), so ~17% headroom exists
  on such spans - contrary to the naive "already 6 bits/char"
  assumption. But tree overhead (~25-40 bits) eats it at typical span
  lengths (16-40 bytes), which is why binary base64 still went
  1-for-54 on the holdout.

## Worked examples

**Win - long tracking URL with a 32-hex id** (holdout):
`amazon.com/Echo-Dot/dp/B08YT2N5SX?...&linkId=956377554b182fa39a44015b148b2450&...`
(139 chars) - current **129** (exceeds model context, falls back to
classic), chunked **58**, transformed **54** (hex id → 16 bytes on
the binary channel). Win vs current: +75 symbols; transform-specific
part: +4.

**Loss - percent-encoded nested URL** (holdout):
`linkedin.com/login?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2F...`
- current **17**, transformed **41**. The model codes the encoded
form at ~0.7 bits/char (memorized pattern); unwrapping only destroys
that.

**Loss - JWT** (synthetic):
`api.site.io/v1/token/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIi...`
- current **173**, chunked **157**, transformed **218** (tree 71
bits, content 1,060 bits over 106 chars, signature 32 bytes binary).
The JSON header/payload cost 10 bits/char through the URL model.

**Win - base64-wrapped return URL** (CC sample; the concept's
best case, see next section):
`ssowelfare.fnmgroup.it/UI/RichiediPassword?authority=Store&returnUrl=aHR0cHM6Ly93ZWxmYXJlLmZubWdyb3VwLml0L2xvZ2luP1JldHVyblVybD0vY29udGFjdHVzJnNzb1Rva2VuPQ%3D%3D`
(172 chars) unwraps to
`...returnUrl=https://welfare.fnmgroup.it/login?ReturnUrl=/contactus&ssoToken=`
- current **147**, chunked **102**, transformed **58** (tree 57 bits;
content 133 chars at 2.3 bits/char - decoded URL text is exactly what
the model knows). Win +89, of which +44 is transform-specific.

## Scoring every detected URL (1,065 from the CC sample)

The holdout only had 97 affected URLs, so all 1,065 detected URLs
from the incidence sample were also scored end-to-end (this set is
*conditional on detection* - it says nothing about corpus-level gain
by itself):

| scheme | avg payload symbols | vs current |
|---|---|---|
| current hybrid | 38.14 | - |
| + transform scheme | 36.59 | **-4.07%** |
| + chunked coding only | 36.83 | -3.44% |
| + both | 36.54 | -4.18% |

- **0 inversion failures**, 0 skips (55,065 URLs verified in total
  across all runs).
- 45/1,065 positive - the same ~96% detected-but-no-win rate as the
  holdout. Wins: mean -13.2, median -13; among the 45 positive: mean
  +36.7, median +14, p90 +87 symbols.
- Transform-specific margin over plain chunking averages **0.24
  symbols** on affected URLs (~0.6%): even where the scheme wins,
  chunked coding of the raw URL captures most of it.
- Per type: base64→text is the only detector that *usually* wins when
  it fires (22/32 positive) - the concept works precisely where its
  premise holds (base64-wrapped URLs/JSON) - but that's 0.06% of the
  corpus. Hex: 14/291. Binary base64: 19/557. Percent: 5/211.
- Incidence-weighted, this slice implies a corpus-level gain of
  roughly 2.13% incidence x 4.07% relative win x ~2x payload-mass
  ratio of affected URLs = **~0.17%** - consistent with the +0.23%
  measured directly on the holdout.

## What overhead dominates

For typical spans the serialized tree costs 24-45 bits (4-7 payload
symbols): gamma-coded gap + length are ~10-20 bits each at realistic
positions, plus type/variant flags. Percent spans whose escape set
isn't exactly a canonical encoder's (e.g. `%2E` for `.`) fall back to
the mask - 1 bit per decoded char + 2 per escape - which for a
50-char span adds ~7 more symbols and is the single biggest metadata
cost in practice. But overhead is *not* the reason the scheme loses;
even with a free tree, unwrapped content usually costs more model
bits than the encoded original.

## Limitations

- The holdout skews short (avg 60 chars, max 180): heavy-blob URLs
  (multi-KB SAML/consent redirects) are underrepresented, and they are
  where both chunking and transforms would help most. The CC-sample
  scoring above partially addresses this.
- Detection is heuristic: the base64 gates (mixed case + digit,
  length ≥16, canonical re-encode) still fire on random-looking path
  slugs; nearly all such detections decode to "binary" and lose.
  Precision could be improved, but that only reduces wasted work -
  `min()` already keeps false positives out of the payload.
- Percent-span boundaries follow URL delimiters (`& = ? #`);
  partially-encoded values can split awkwardly. No attempt was made
  to transform `+`-as-space or HTML-entity encodings.
- Scoring reuses the shipped model file; a scheme shipped for real
  would pay the same version-marker bit we charged (version 3).

## Verdict

**Not worth shipping as a payload scheme against the current model.**
Incidence is low (2.1%), the false-positive rate is extreme (96%),
and the corpus-level gain (+0.23%) is smaller than what plain chunked
coding - a far simpler scheme with no tree, no detectors, and no
inversion risk - already delivers (+0.27%).

The measurement does point at two real follow-ups:

1. **Chunked coding** of long URLs is cheap, transform-free, and
   turns the model's context ceiling into a non-event; it accounts
   for most of what the transform scheme appeared to win. Worth
   considering as (part of) a future payload version on its own.
2. If a **future model is trained on unwrapped representations**
   (decoded JSON/nested URLs), the transform tree becomes the bridge
   between wire format and model distribution, and this prototype's
   inversion machinery and accounting are directly reusable. That is
   a training-data experiment first, not a payload-format one.

## Reproducing

```sh
node run.mjs incidence <slice.txt> --sample 50000 --seed 42
node run.mjs score <holdout.txt> [--offset N --limit N]
node run.mjs entropy
node --test model/harness/transforms/   # invertibility suite
```
