#!/usr/bin/env bash
# Fingerprint audit for the divergent-design skill.
#
# Exact-hex matching is the point, not a shortcut: the house style is
# built from MEMORIZED values (Tailwind's 300-500 hexes, the same few
# near-blacks), so literal matches are the habit itself. A palette
# derived from the subject almost never lands on these strings.
#
# STRONG findings are the signature; fix them or argue for them in the
# stylesheet's DESIGN STANCE block. WEAK findings are common in human
# work too and only matter in numbers (3+ means drift).
#
# Occurrences are counted with `grep -o | wc -l`, never `grep -c` —
# minified CSS is one line and would count as one hit.
set -u

if [ $# -lt 1 ]; then
  echo "usage: audit.sh <css/html/js files...>" >&2
  exit 2
fi

files=()
for f in "$@"; do
  if [ -f "$f" ]; then files+=("$f"); else echo "skip (not a file): $f" >&2; fi
done
if [ ${#files[@]} -lt 1 ]; then exit 2; fi

strong=0
weak=0

hits () {
  grep -Eio -- "$1" "${files[@]}" 2>/dev/null | wc -l | tr -d ' '
}

check () {
  local sev="$1" label="$2" pat="$3" n
  n=$(hits "$pat")
  if [ "$n" -gt 0 ]; then
    printf '%-7s %-42s x%s\n' "$sev" "$label" "$n"
    if [ "$sev" = "STRONG" ]; then strong=$((strong + 1)); else weak=$((weak + 1)); fi
  fi
}

echo "divergent-design audit: ${files[*]}"
echo "-------------------------------------------------------------"

check STRONG "stock palette hex (Tailwind 300-500)" \
  "#(38bdf8|0ea5e9|4ade80|22c55e|34d399|10b981|c084fc|a78bfa|8b5cf6|7c3aed|6366f1|818cf8|f472b6|ec4899|e879f9|fbbf24|f59e0b|f97316|ef4444|dc2626|60a5fa|3b82f6|2563eb|06b6d4|14b8a6)\b"
check STRONG "the #667eea->#764ba2 gradient pair" \
  "#(667eea|764ba2)\b"
check STRONG "default tinted near-black background" \
  "#(0a0a0a|0d0d0d|07070d|0a0a0f|0b0b10|0f0f14|16161d|0f172a|111827|1a1a2e|0e0e1c|13132a|191930)\b"
check STRONG "surface elevation ladder token" \
  "--surface[23]\b|--bg-(secondary|tertiary)\b"
check STRONG "accent/-dim paired badge tokens" \
  "--[a-z][a-z-]*-dim ?:"
check STRONG "hover lift translateY(-1/2/4px)" \
  "translateY\(-(1|2|4)px\)"
check STRONG "transition: all .2s/.15s" \
  "transition: ?all +0?\.(15|2)s"
check STRONG "linear-gradient(135deg ...)" \
  "linear-gradient\( ?135deg"
check STRONG "gradient text via background-clip" \
  "background-clip: ?text"
check STRONG "999px pill radius" \
  "border-radius: ?9{3,}px"
check STRONG "white-alpha hairline border" \
  "1px solid rgba\(255, ?255, ?255, ?(0?\.0[5-9]|0?\.1[0-5]?)\)"
check STRONG "emoji-in-SVG data-URI favicon" \
  "data:image/svg\+xml[^\"']{0,200}(<text|%3Ctext)"

check WEAK "shortlist font" \
  "(family=|font-family[^;}]{0,80})(Inter|Space.?Grotesk|Outfit|Manrope|DM.?Sans|Sora|Plus.?Jakarta|Poppins|Bebas.?Neue|Barlow)\b"
check WEAK "hairline border on var(--border...)" \
  "1px solid var\(--border"
check WEAK "backdrop-filter blur" \
  "backdrop-filter: ?blur"
check WEAK "soft black card shadow" \
  "box-shadow: ?0 [248]px [0-9]+px rgba\(0, ?0, ?0"
check WEAK "hover scale(1.02/1.05)" \
  "scale\(1\.0[25]\)"
check WEAK "tight hero tracking (-0.02/-0.03em)" \
  "letter-spacing: ?-0?\.0[23]em"

# The spaced-uppercase micro-label is a pairing, not a property: either
# half alone is ordinary, three or more of both in one file is the tell.
for f in "${files[@]}"; do
  u=$(grep -Eio -- "text-transform: ?uppercase" "$f" 2>/dev/null | wc -l | tr -d ' ')
  l=$(grep -Eio -- "letter-spacing: ?[0-9.]+px" "$f" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$u" -ge 3 ] && [ "$l" -ge 3 ]; then
    printf '%-7s %-42s %s\n' "STRONG" "spaced-uppercase micro-labels" "$f (uppercase x$u, tracked x$l)"
    strong=$((strong + 1))
  fi
done

# Radius comfort zone: 6-12px is unremarkable in any one rule; a
# stylesheet where 60%+ of radii sit there took no stance on shape.
total=$(hits "border-radius: ?[^;}]+")
cluster=$(hits "border-radius: ?(6|8|10|12)px")
if [ "$total" -ge 6 ] && [ $((cluster * 10)) -ge $((total * 6)) ]; then
  printf '%-7s %-42s %s\n' "STRONG" "radius comfort zone (6-12px)" "$cluster of $total radii"
  strong=$((strong + 1))
fi

echo "-------------------------------------------------------------"
if [ "$strong" -eq 0 ] && [ "$weak" -lt 3 ]; then
  echo "verdict: clean — no house-style signature detected"
  exit 0
elif [ "$strong" -eq 0 ]; then
  echo "verdict: leaning default — $weak weak signals, no strong ones"
  exit 0
else
  echo "verdict: house style detected — $strong strong signals, $weak weak"
  echo "fix each STRONG finding or argue for it in the DESIGN STANCE block"
  exit 1
fi
