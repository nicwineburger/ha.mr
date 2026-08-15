/**
 * Opt-in "clean link" preprocessing: strips a curated set of
 * cross-site/campaign tracking query parameters from a link before it
 * is compressed. This is deliberately lossy — the short link decodes
 * to the cleaned URL — so callers must only apply it when the user
 * explicitly asked for it (it is never on by default).
 *
 * The list is conservative on purpose: it only contains parameters
 * that are pure tracking identifiers and never affect page content.
 * Ambiguous or functional parameters (e.g. "ref", "q", "id", "page",
 * "spm", "source") are never stripped, even though some sites use
 * them for tracking — on other sites those same names select the
 * content, and a cleaned link must always load the same page. When in
 * doubt, a parameter stays.
 */

// Any parameter whose name starts with one of these is stripped:
// utm_* (Google Analytics), oly_* (Omeda), _hs* (HubSpot: _hsenc,
// _hsmi), vero_* (Vero), matomo_*/mtm_*/pk_* (Matomo/Piwik)
const trackingPrefixes = [
  "utm_",
  "oly_",
  "_hs",
  "vero_",
  "matomo_",
  "mtm_",
  "pk_"
];

// Parameters stripped on an exact name match: click identifiers and
// campaign tokens set by ad and email platforms
const trackingNames = new Set([
  "gclid", "gclsrc", "dclid", "gbraid", "wbraid", "wickedid", // Google Ads
  "gad_source", // Google Ads
  "srsltid", // Google Merchant Center
  "fbclid", "igshid", "igsh", "sfnsn", // Meta
  "mc_cid", "mc_eid", // Mailchimp
  "msclkid", "cvid", // Microsoft Ads / Bing
  "twclid", // Twitter/X
  "ttclid", // TikTok
  "li_fat_id", // LinkedIn
  "mkt_tok", // Marketo
  "yclid" // Yandex
]);

// Matching is case-insensitive: these names identify trackers
// regardless of spelling, and no known site uses a recased variant
// for anything functional
function isTrackingParam (name) {
  const lower = name.toLowerCase();
  if (trackingNames.has(lower)) return true;
  return trackingPrefixes.some(prefix => lower.startsWith(prefix));
}

/**
 * Strips known tracking query parameters from a link.
 *
 * Everything except the removed parameters is preserved byte-exact:
 * surviving parameters keep their order and spelling, and the path
 * and fragment are untouched. If stripping removes every parameter,
 * the "?" is dropped too. The link is handled as a raw string rather
 * than through `new URL(...)`, which would normalize unrelated parts
 * and silently change what the payload decodes to.
 * @param {string} link the link to clean
 * @returns {{cleaned: string, removed: string[]}} the cleaned link and
 *   the names of the stripped parameters (deduplicated, in order)
 */
export function cleanLink (link) {
  // The fragment starts at the first "#"; a "?" inside the fragment
  // does not start a query, so only a "?" before the "#" counts
  const hashIndex = link.indexOf("#");
  const queryEnd = hashIndex === -1 ? link.length : hashIndex;
  const queryIndex = link.slice(0, queryEnd).indexOf("?");
  if (queryIndex === -1) return { cleaned: link, removed: [] };

  const before = link.slice(0, queryIndex);
  const query = link.slice(queryIndex + 1, queryEnd);
  const after = link.slice(queryEnd);

  const removed = [];
  const kept = query.split("&").filter(param => {
    const name = param.split("=", 1)[0];
    if (!isTrackingParam(name)) return true;
    if (!removed.includes(name)) removed.push(name);
    return false;
  });

  const cleaned = kept.length > 0
    ? `${before}?${kept.join("&")}${after}`
    : before + after;
  return { cleaned, removed };
}
