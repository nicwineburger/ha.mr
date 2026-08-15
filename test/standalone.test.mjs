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

function runWithDomain (domain, ...args) {
  return execFileSync(process.execPath, [cli, ...args], {
    encoding: "utf8",
    env: { ...process.env, HAMR_DOMAIN: domain }
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

test("CLI builds and decodes links on a custom domain", () => {
  const link = "https://www.example.com/some/path?a=1#b";
  const short = runWithDomain("short.example", link);
  assert.match(short, /^https:\/\/short\.example#/);
  assert.equal(runWithDomain("short.example", short), link);
});

test("CLI builds and decodes QR links on a custom domain", () => {
  const link = "https://www.example.com/test";
  const short = runWithDomain("short.example", link, "qr");
  assert.match(short, /^HTTP:\/\/SHORT\.EXAMPLE\//);
  assert.equal(runWithDomain("short.example", short), link);
});

test("CLI treats domain-prefixed hostnames as links, not payloads", () => {
  // "ha.mrs.example" starts with "ha.mr" but is a regular link
  const short = run("https://ha.mrs.example/path");
  assert.equal(run(short), "https://ha.mrs.example/path");
});

test("CLI decodes links from archived model versions", async () => {
  // Version-1 links must keep decoding after the latest model moved
  // to version 2 - the CLI lazy-loads model/url-model-v1.bin
  const { compressHybrid } = await import("../hybrid.js");
  const { URLModel } = await import("../neural.js");
  const { outputAlphabetASCII } = await import("../alphabets.js");
  const { readFile } = await import("node:fs/promises");
  const modelV1 = new URLModel(
    (await readFile(new URL("../model/url-model-v1.bin", import.meta.url))).buffer);
  const link = "https://www.example.com/archived/version";
  const payload = compressHybrid(link, outputAlphabetASCII, modelV1);
  assert.equal(run(`https://ha.mr#${payload}`), link);
});

test("CLI rejects unknown alphabets", () => {
  assert.throws(() => run("https://example.com", "bogus"));
});

test("CLI requires an input link", () => {
  assert.throws(() => run());
});
