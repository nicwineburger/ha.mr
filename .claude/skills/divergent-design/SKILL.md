---
name: divergent-design
description: Use whenever designing or styling a web page, site, or UI from scratch — HTML/CSS artifacts, mockups, landing pages, dashboards. Breaks the default "AI house style" (stock palette hexes, tinted near-black surface ladders, hairline card dashboards, spaced-uppercase micro-labels) by deriving a bespoke visual language from the subject first, then linting the result with the bundled audit script. Not for projects with an existing brand or design system — match those instead.
---

# Divergent Design

Unconstrained "make it look good" samples the model's modal design: the
same skin on every project with a different accent hue. The result is
attributable on sight — token ladder, hairline cards, uppercase
micro-labels, Tailwind hexes, `translateY(-2px)` hovers. This skill
exists to force a *derived* design instead of the *default* one.

The tell has two layers. The **vocabulary** (specific hexes, radii,
label styles, hover moves) is mechanically checkable — `audit.sh`
greps for it. The **sameness** (one system reused across unrelated
projects) is avoided by deriving the system from the subject before
writing any CSS. Consistency *within* a project is good design; keep
it. Sameness *across* projects is the fingerprint; that's what this
kills.

## Procedure — derive, commit, execute, audit

**1. Name the subject's native visual world.** Before thinking about
CSS, list three real-world artifacts people already associate with the
domain — physical ones, not websites. A Marvel tracker: comic panel
gutters, film end-credits, S.H.I.E.L.D. dossiers. A bakery: kraft
paper, flour dust, hand-chalked price boards. A finance tool: ledger
rules, receipt paper, terminal phosphor. Steal structure, texture, and
palette from those artifacts. If you catch yourself reaching for
"modern dashboard" as the world, stop — that's the default talking.

**2. Write the stance block before any CSS.** Paste it as a comment at
the top of the stylesheet so it ships with the artifact and future
edits argue with it instead of drifting back to defaults:

```css
/* DESIGN STANCE — argue with this block before changing the look.
   WORLD:     <the three artifacts>
   PALETTE:   <where each color came from — sampled from the world,
              or derived in oklch. Never a framework's named values.>
   MODE:      <light or dark, chosen from context — not dark-by-default>
   TYPE:      <faces and why they belong to this subject>
   SHAPE:     <radius stance, border stance, density — one stance, stated>
   DEPTH:     <flat / shadows / overlap / printed rules — pick one;
              never a surface-lightness ladder by reflex>
   LAYOUT:    <the skeleton; if it's a centered column of cards, justify>
   SIGNATURE: <2–3 moves unique to this project a designer would be
              recognized by>
   MOTION:    <stance, possibly "none">
*/
```

**3. Pick the signature moves.** Two or three deliberate, project-
specific decisions carry more identity than fifty token names: an
asymmetric column split, borders on only one edge, a repeated angle,
a texture, ruled lines instead of boxes, oversized numerals, a single
accent used exactly once per screen. If every move on the list could
appear on any site, they aren't signatures yet.

**4. Execute with full consistency, then audit.**

```
bash .claude/skills/divergent-design/audit.sh <css/html files>
```

Fix every STRONG finding or add a line to the stance block arguing why
it stays. WEAK findings matter only in numbers — three or more means
the defaults are creeping back.

## The fingerprint (what the audit greps for)

Banned as *reflexes*, not as techniques — see the escape hatch below.

- **Memorized palette hexes.** Tailwind's 300–500 range (`#38bdf8`,
  `#4ade80`, `#c084fc`, `#f59e0b`, `#ef4444`, …) and the `#667eea` →
  `#764ba2` gradient pair. The model emits memorized values; a derived
  palette almost never lands on them exactly. Specify color in
  `oklch()` or from sampled sources — it also breaks the habit at the
  syntax level.
- **Tinted near-black by default.** `#0a0a0a`, `#0d0d0d`, `#07070d`,
  `#0f172a`, `#1a1a2e` and kin. If dark is right for the subject, pick
  a *chromatic* dark from the world (ink blue, oxblood, bottle green,
  warm brown-black) — and decide light vs dark at all from context.
- **The surface ladder.** `--surface` / `--surface2` / `--surface3`
  elevation-by-lightening, plus paired `--accent` / `--accent-dim`
  badge tokens. Choose a depth strategy in the stance block instead.
- **Hairline cards.** `1px solid` white-alpha or `var(--border)` on
  every container. Boxes are not the only grouping device: rules,
  whitespace, background shifts, and overlap all group.
- **Spaced-uppercase micro-labels.** `text-transform: uppercase` +
  `letter-spacing` on small muted text over every stat and section.
- **The radius comfort zone.** Everything at 6–12px with `999px`
  pills. Take a stance: sharp, or nearly-sharp, or huge, or mixed on
  purpose — stated in SHAPE.
- **Default motion.** `transition: all .2s`, hover `translateY(-2px)`,
  `scale(1.02)`. Motion is designed per project or omitted.
- **The gradient habits.** `linear-gradient(135deg …)` backgrounds,
  gradient-filled hero text via `background-clip: text`, low-opacity
  radial "glow blobs".
- **The font shortlist.** Inter, Space Grotesk, Outfit, Manrope,
  DM Sans, Sora, Plus Jakarta Sans, Poppins, Bebas Neue + Barlow.
  Weak signal alone — humans use these too — but combined with the
  above it completes the picture. TYPE must say why a face belongs to
  *this* subject; "clean and modern" is not a reason.
- **Misc tells.** Emoji-in-SVG data-URI favicons; `backdrop-filter`
  blurred sticky headers; `box-shadow: 0 4px 12px rgba(0,0,0,…)`
  card shadows; the centered max-width column of stat tiles.

## Escape hatch

Any banned item may be used when the stance block argues it *from the
subject*: spaced-uppercase labels genuinely belong on a military
dossier UI; a near-black belongs to a film-photography portfolio;
pills belong where the world contains actual pills. The offense is
the reflex, not the technique. No argument in the stance block, no
use.

## Honest limits

This removes the visual signature, not every signature. Still
attributable, out of scope here: microcopy voice (em-dashes, "No
items yet" empty states, self-aware error messages), semantic
class-naming discipline, comment style, and the eerie uniformity of a
system applied hundreds of times without drift. And the derivation
step is itself performed by the model — expect divergence from the
*default*, not perfect anonymity.

## When not to use

The project already has a brand, design system, or an existing site
to match — match it, and none of this applies. Likewise when the
user asks for a specific look, including "the usual AI dashboard
look": what they ask for wins.
