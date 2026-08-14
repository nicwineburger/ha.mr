import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const cli = fileURLToPath(new URL("../standalone.js", import.meta.url));

function run (...args) {
  return execFileSync(process.execPath, [cli, ...args], {
    encoding: "utf8"
  }).trim();
}

test("CLI compresses and decodes text links", () => {
  const link = "https://www.example.com/some/path?a=1#b";
  const short = run(link);
  assert.match(short, /^https:\/\/ha\.mr#/);
  assert.equal(run(short), link);
});

test("CLI compresses and decodes QR links", () => {
  // Regression: QR links used to be mis-detected and decoded with the
  // wrong alphabet, producing garbage output
  const link = "https://www.example.com/test";
  const short = run(link, "qr");
  assert.match(short, /^HTTP:\/\/HA\.MR\//);
  assert.equal(run(short), link);
});

test("CLI decodes emoji links", () => {
  const link = "https://example.com/emoji-test";
  const short = run(link, "emoji");
  assert.equal(run(short), link);
});

test("CLI accepts links without a protocol prefix", () => {
  const short = run("https://example.com/x");
  assert.equal(run(short.replace("https://", "")), "https://example.com/x");
});

test("CLI rejects unknown alphabets", () => {
  assert.throws(() => run("https://example.com", "bogus"));
});

test("CLI requires an input link", () => {
  assert.throws(() => run());
});
