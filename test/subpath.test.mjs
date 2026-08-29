import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { compress } from "../docs/compress.js";
import { outputAlphabetQR } from "../docs/alphabets.js";

/*
 * End-to-end check that the site works when served under a base path,
 * the way a GitHub Pages project site serves it (e.g. /ha.mr/ on
 * user.github.io): assets must load, generated links must carry the
 * base path, and both hash and QR-path payloads must decode. The
 * server below mimics Pages' project-site routing: files under the
 * base path, the site's own 404.html (with a 404 status) for unknown
 * paths under it, and a bare 404 for everything else.
 */

const BASE = "/ha.mr";
const DOCS = fileURLToPath(new URL("../docs", import.meta.url));
const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream",
  ".woff2": "font/woff2"
};

const CHROMIUM_PATH = "/opt/pw-browsers/chromium";
const PLAYWRIGHT_PATH = process.env.PLAYWRIGHT
  || "/opt/node22/lib/node_modules/playwright/index.mjs";

// Same skip convention as wasm/browser-production.test.mjs: only
// `undefined` means "run normally"
let unavailableReason = undefined;
let chromium;
try {
  ({ chromium } = await import(PLAYWRIGHT_PATH));
  await readFile(CHROMIUM_PATH).catch(() => {});
} catch (e) {
  unavailableReason = `Playwright unavailable: ${e && e.message ? e.message : e}`;
}

function startPagesMimic () {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (path === BASE) {
      res.writeHead(301, { location: `${BASE}/` });
      return res.end();
    }
    const serve = async (file, status) => {
      const data = await readFile(join(DOCS, normalize(file)));
      res.writeHead(status, {
        "content-type": TYPES[extname(file)] || "application/octet-stream"
      });
      res.end(data);
    };
    try {
      if (path === `${BASE}/`) return await serve("/index.html", 200);
      if (!path.startsWith(`${BASE}/`)) throw new Error("outside base");
      await serve(path.slice(BASE.length), 200);
    } catch {
      try {
        if (path.startsWith(`${BASE}/`)) return await serve("/404.html", 404);
      } catch {}
      res.writeHead(404, { "content-type": "text/html" });
      res.end("<h1>404</h1>");
    }
  });
  return new Promise(resolve =>
    server.listen(0, () => resolve({ server, port: server.address().port })));
}

test("site works when served under a base path",
  { skip: unavailableReason }, async () => {
    const link = "https://en.wikipedia.org/wiki/Hammer";
    const { server, port } = await startPagesMimic();
    const origin = `http://localhost:${port}`;
    const browser = await chromium.launch({ executablePath: CHROMIUM_PATH });
    try {
      const page = await browser.newPage();
      const failed = new Set();
      page.on("response", r => {
        if (r.status() >= 400) failed.add(new URL(r.url()).pathname);
      });

      // The compressor view: every asset resolves under the base path
      await page.goto(`${origin}${BASE}/`, { waitUntil: "networkidle" });
      assert.deepEqual([...failed], [], "no asset requests may fail");
      assert.equal(await page.title(),
        `localhost:${port}${BASE} - link compressor`);

      // Generated links carry the base path
      await page.fill("#input-link", link);
      await page.waitForFunction(
        `document.querySelector("#output-link").textContent.includes("#")`,
        null, { timeout: 10000 });
      const short = await page.textContent("#output-link");
      assert.ok(short.startsWith(`${origin}${BASE}#`),
        `short link "${short}" should start with ${origin}${BASE}#`);

      // A hash payload decodes back to the link (classic payloads
      // need no model, so the prompt shows immediately). Leave the
      // page first: same-path hash navigation wouldn't reload it.
      await page.goto("about:blank");
      await page.goto(`${origin}${BASE}/#${short.split("#")[1]}`,
        { waitUntil: "load" });
      await page.waitForSelector("#redirect-container",
        { state: "visible", timeout: 15000 });
      assert.equal(await page.textContent("#redirect-link"), link);

      // A QR-style payload rides in the path below the base, served
      // through the 404 fallback; the page must still find its assets
      // and decode it
      failed.clear();
      const qrPayload = compress(link, outputAlphabetQR);
      await page.goto(`${origin}${BASE}/${qrPayload}`, { waitUntil: "load" });
      await page.waitForSelector("#redirect-container",
        { state: "visible", timeout: 15000 });
      assert.equal(await page.textContent("#redirect-link"), link);
      assert.ok([...failed].every(p => p === `${BASE}/${qrPayload}`),
        `only the payload path itself may 404, got: ${[...failed]}`);
    } finally {
      await browser.close();
      server.close();
    }
  });
