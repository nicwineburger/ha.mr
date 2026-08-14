import { test } from "node:test";
import assert from "node:assert/strict";
import { compress, decompress } from "../compress.js";
import {
  outputAlphabetASCII,
  outputAlphabetQR,
  outputAlphabetEmoji
} from "../alphabets.js";

const alphabets = {
  ascii: outputAlphabetASCII,
  qr: outputAlphabetQR,
  emoji: outputAlphabetEmoji
};

function roundtrip (link, alphabet) {
  return decompress(compress(link, alphabet), alphabet);
}

/**
 * Links that must survive a compress/decompress cycle byte-for-byte.
 */
const exactCases = [
  "https://example.com",
  "http://example.com",
  "https://www.example.com",
  "https://example.com/some/path",
  "https://www.example.com/some/path?a=1&b=2#frag",
  "https://en.wikipedia.org/wiki/Hammer_(disambiguation)",
  "https://www.amazon.com/dp/B08N5WRWNW?ref_=cm_sw_r_cp_ud_dp",
  "https://example.com/index.html",
  "https://example.com/a/index.php",
  "https://sub.domain.example.co.uk/x",
  "https://example.com/UPPER/lower/12345/MiXeD123",
  "http://localhost:8080/test",
  "https://example.com/a?x=1#h",
  "https://example.com/a=b/c",
  "https://example.com/na%C3%AFve",
  "https://example.com/a%0Db%0Ac",
  "https://example.com/%00%01%0F",
  "https://example.com/docs/",
  "https://example.com/a/b/c/",
  "https://example.com/docs/?a=b",
  "https://example.com/docs/#frag",
  "https://xn--nxasmq6b.example/x"
];

for (const [name, alphabet] of Object.entries(alphabets)) {
  test(`exact round-trips (${name})`, () => {
    for (const link of exactCases) {
      assert.equal(roundtrip(link, alphabet), link, `round-trip of ${link}`);
    }
  });
}

/**
 * Intentional normalizations: the output is a different string, but an
 * equivalent link. These pin down known behavior so accidental changes
 * to it show up in review.
 */
const normalizedCases = [
  // Root-path slash is implied
  ["https://example.com/", "https://example.com"],
  // Escapes of unreserved characters are decoded
  ["https://example.com/%7Euser", "https://example.com/~user"],
  ["https://example.com/%41", "https://example.com/A"],
  // Escape hex is uppercased
  ["https://example.com/na%c3%afve", "https://example.com/na%C3%AFve"],
  // Tilde is percent-encoded internally but must decode to the same char
  ["https://example.com/~user/page", "https://example.com/~user/page"],
  // Spaces become %20
  ["https://example.com/a b", "https://example.com/a%20b"],
  // Valueless query params gain "="
  ["https://example.com/?foo", "https://example.com?foo="],
  // Stray "%" is escaped rather than crashing
  ["https://example.com/50%_off", "https://example.com/50%25_off"],
  ["https://example.com/100%", "https://example.com/100%25"],
  // Hex-valid escape that isn't valid UTF-8 survives verbatim
  ["https://example.com/a%C3z", "https://example.com/a%C3z"],
  // Hostname is lowercased
  ["https://EXAMPLE.COM/Path", "https://example.com/Path"],
  // Square brackets are percent-encoded (they're reserved for IPv6 hosts)
  ["https://example.com/a[b]c", "https://example.com/a%5Bb%5Dc"],
  // Query "+" becomes "%20" (form-encoding equivalence); path "+" is
  // a literal plus and is preserved
  ["https://example.com/a+b?q=c+d", "https://example.com/a+b?q=c%20d"]
];

test("intentional normalizations (ascii)", () => {
  for (const [input, expected] of normalizedCases) {
    assert.equal(roundtrip(input, outputAlphabetASCII), expected,
      `normalization of ${input}`);
  }
});

test("low-byte percent escapes decode to the original bytes", () => {
  // Regression: "%0A" used to come back as a bare "%a", which swallowed
  // the next character as a hex digit and corrupted the URL
  const link = "https://example.com/a%0Db%0Ac";
  const output = roundtrip(link, outputAlphabetASCII);
  assert.equal(decodeURIComponent(new URL(output).pathname), "/a\rb\nc");
});

test("query parameters with encoded characters", () => {
  const cases = [
    "https://example.com/?q=100%25",
    "https://example.com/?a=%0A",
    "https://example.com/search?q=a%20b&lang=en"
  ];
  for (const link of cases) {
    const output = roundtrip(link, outputAlphabetASCII);
    const a = new URL(link).searchParams;
    const b = new URL(output).searchParams;
    assert.deepEqual(Array.from(b), Array.from(a), `params of ${link}`);
  }
});

test("trailing slashes on non-root paths are preserved", () => {
  assert.equal(roundtrip("https://example.com/docs/", outputAlphabetASCII),
    "https://example.com/docs/");
  assert.equal(roundtrip("https://example.com/a/b/", outputAlphabetQR),
    "https://example.com/a/b/");
});

test("every character of the URL alphabet round-trips in a path", () => {
  // Square brackets are excluded: they normalize to %5B/%5D (see above)
  const urlAlphabet =
    "!$&'()*+,-.0123456789:;=@ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~";
  for (const ch of urlAlphabet) {
    const link = `https://example.com/a${ch}b`;
    const output = roundtrip(link, outputAlphabetASCII);
    assert.equal(new URL(output).href, new URL(link).href, `char "${ch}"`);
  }
});

/**
 * Deterministic fuzzing: random-but-reproducible URLs, compared after
 * WHATWG URL normalization so equivalent spellings don't count as
 * failures.
 */
function mulberry32 (seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

test("fuzz: random URLs round-trip (ascii + qr)", () => {
  const rand = mulberry32(42);
  const rnd = n => Math.floor(rand() * n);
  const hosts = [
    "example.com", "www.google.com", "sub.foo.co.uk", "localhost",
    "a-b.xyz", "deep.sub.domain.example.org", "127.0.0.1"
  ];
  const segChars = "abcdefghijABCDE0123456789-_.!$'()*,;=@";
  for (let i = 0; i < 2000; i++) {
    const host = hosts[rnd(hosts.length)];
    let path = "";
    const segments = rnd(4);
    for (let s = 0; s < segments; s++) {
      path += "/";
      const len = 1 + rnd(8);
      for (let c = 0; c < len; c++) path += segChars[rnd(segChars.length)];
    }
    if (segments && rand() < 0.3) path += "/";
    const query = rand() < 0.3 ? `?k${rnd(9)}=v${rnd(9)}&x=${rnd(99)}` : "";
    const hash = rand() < 0.3 ? `#sec${rnd(9)}` : "";
    const port = rand() < 0.1 ? `:${1 + rnd(65534)}` : "";
    const proto = rand() < 0.5 ? "https" : "http";
    const link = `${proto}://${host}${port}${path}${query}${hash}`;
    if (!URL.canParse(link)) continue;
    const alphabet = rand() < 0.5 ? outputAlphabetASCII : outputAlphabetQR;
    const output = roundtrip(link, alphabet);
    assert.equal(new URL(output).href, new URL(link).href, `fuzz #${i}: ${link}`);
  }
});

test("compress rejects garbage input", () => {
  assert.throws(() => compress("", outputAlphabetASCII));
});

test("decompress rejects characters outside the alphabet", () => {
  assert.throws(() => decompress("\u{1F600}", outputAlphabetASCII));
});
