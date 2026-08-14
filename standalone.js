#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { compressHybrid, decompressHybrid } from "./hybrid.js";
import { URLModel } from "./neural.js";
import {
  outputAlphabetASCII,
  outputAlphabetQR,
  outputAlphabetEmoji
} from "./alphabets.js";

// Load the neural model shipped alongside the CLI; fall back to
// classic-only compression if it's unavailable
let model = null;
try {
  const buffer = await readFile(new URL("./model/url-model.bin", import.meta.url));
  model = new URLModel(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
} catch (e) {
  console.error("Warning: neural model unavailable, using classic compression only.");
}

// The domain used to build and recognize short links; set HAMR_DOMAIN
// to use a self-hosted deployment (defaults to the original site)
const domain = (process.env.HAMR_DOMAIN || "ha.mr").toLowerCase();

const input = process.argv[2]?.trim();
const alphabetName = process.argv[3]?.trim() || "ascii";
if (!input) {
  console.error(`Usage: hamr <link> [ascii|qr|emoji]`);
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
  if (isQRCode) console.log(decompressHybrid(payload, outputAlphabetQR, model));
  else console.log(decompressHybrid(payload, useEmoji ? outputAlphabetEmoji : outputAlphabetASCII, model));
  process.exit(0);
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
  console.log(`HTTP://${domain.toUpperCase()}/` + compressHybrid(input, alphabet, model));
} else {
  console.log(`https://${domain}#` + compressHybrid(input, alphabet, model));
}
