#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { compressHybrid, decompressHybrid, payloadSchemeVersion } from "./hybrid.js";
import { cleanLink } from "./clean.js";
import { URLModel } from "./neural.js";
import {
  outputAlphabetASCII,
  outputAlphabetQR,
  outputAlphabetEmoji
} from "./alphabets.js";

// Load the neural model shipped alongside the CLI; fall back to
// classic-only compression if it's unavailable
async function loadModel (file) {
  const buffer = await readFile(new URL(file, import.meta.url));
  return new URLModel(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}
let model = null;
try {
  model = await loadModel("./model/url-model.bin");
} catch (e) {
  console.error("Warning: neural model unavailable, using classic compression only.");
}

// The domain used to build and recognize short links; set HAMR_DOMAIN
// to use a self-hosted deployment (defaults to the original site)
const domain = (process.env.HAMR_DOMAIN || "ha.mr").toLowerCase();

// "--clean" may appear anywhere in argv; the remaining arguments keep
// their positional meaning
const args = process.argv.slice(2);
const clean = args.includes("--clean");
const positional = args.filter(arg => arg !== "--clean");

const input = positional[0]?.trim();
const alphabetName = positional[1]?.trim() || "ascii";
if (!input) {
  console.error(`Usage: hamr [--clean] <link> [ascii|qr|emoji]`);
  console.error(`  --clean  strip known tracking parameters before compressing`);
  console.error(`           (lossy: the short link decodes to the cleaned URL)`);
  console.error(`Set HAMR_DOMAIN to use a domain other than ha.mr.`);
  process.exit(1);
}

let payload = "";
const inputLower = input.toLowerCase();
for (const prefix of [`https://${domain}`, `http://${domain}`, domain]) {
  if (inputLower.startsWith(prefix)) {
    const rest = input.slice(prefix.length);
    // Only treat this as a short link if a payload separator follows,
    // so e.g. "ha.mrs.example" still compresses as a regular link
    if (rest[0] === "/" || rest[0] === "#") payload = rest;
    break;
  }
}

if (payload) {
  // QR links carry the payload in the path ("/"); text links use the hash ("#")
  const isQRCode = payload[0] === "/";
  payload = payload.slice(1);
  const useEmoji = Array.from(payload).some(c => !outputAlphabetASCII.includes(c));
  const alpha = isQRCode ? outputAlphabetQR
    : useEmoji ? outputAlphabetEmoji : outputAlphabetASCII;
  // Links made by older models decode with their archived model file
  let decodeModel = model;
  const version = payloadSchemeVersion(payload, alpha);
  if (version >= 1 && (!decodeModel || decodeModel.linkVersion !== version)) {
    try {
      decodeModel = await loadModel(`./model/url-model-v${version}.bin`);
    } catch (e) {
      console.error(`This link requires model version ${version} (model/url-model-v${version}.bin).`);
      process.exit(3);
    }
  }
  console.log(decompressHybrid(payload, alpha, decodeModel));
  process.exit(0);
}

// Cleaning applies only when compressing; the note goes to stderr so
// stdout stays the bare short link for piping
let link = input;
if (clean) {
  const result = cleanLink(input);
  link = result.cleaned;
  if (result.removed.length > 0) {
    console.error(`cleaned: removed ${result.removed.join(", ")}`);
  }
}

let alphabet = outputAlphabetASCII;
if (alphabetName === "qr") alphabet = outputAlphabetQR;
else if (alphabetName === "emoji") alphabet = outputAlphabetEmoji;
else if (alphabetName !== "ascii") {
  console.error(`Unknown alphabet "${alphabetName}".`);
  console.error("Select one of: ascii, qr, emoji");
  process.exit(2);
}

if (alphabetName === "qr") {
  console.log(`HTTP://${domain.toUpperCase()}/` + compressHybrid(link, alphabet, model));
} else {
  console.log(`https://${domain}#` + compressHybrid(link, alphabet, model));
}
