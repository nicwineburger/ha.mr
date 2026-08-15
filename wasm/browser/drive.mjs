/**
 * @file Shared Playwright driver: serves the repo root over HTTP
 * (ES modules and fetch need same-origin URLs) and opens the harness
 * page in headless Chromium. Playwright is a dev-time tool, not a
 * dependency - it is resolved from the environment (global install or
 * PLAYWRIGHT env override), never from package.json.
 */

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { fileURLToPath } from "node:url";

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream"
};

/**
 * Serves the repo root, launches Chromium, navigates to a page, runs
 * `fn(page)`, and tears everything down.
 * @param {(page: object) => Promise<void>} fn Driver body
 * @param {string} [path] Page to open, relative to the repo root
 *  served at "/" - defaults to the measurement harness; the
 *  production browser test passes "/" (mapped to index.html, like a
 *  real static host) to drive the real site instead
 */
export async function withHarnessPage (fn, path = "/wasm/browser/harness.html") {
  const root = fileURLToPath(new URL("../..", import.meta.url));
  const server = createServer(async (req, res) => {
    try {
      // "/" maps to index.html, like GitHub Pages/any static host -
      // this matters for the production browser test: main.js reads
      // location.pathname to detect QR-code payloads, so the served
      // path must actually be "/", not "/index.html"
      let path = new URL(req.url, "http://localhost").pathname;
      if (path === "/") path = "/index.html";
      const body = await readFile(root + path);
      res.writeHead(200,
        { "content-type": TYPES[extname(path)] || "text/plain" });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;

  process.env.PLAYWRIGHT_BROWSERS_PATH ||= "/opt/pw-browsers";
  const playwright = process.env.PLAYWRIGHT
    || "/opt/node22/lib/node_modules/playwright/index.mjs";
  const { chromium } = await import(playwright);
  const browser = await chromium.launch({
    headless: true,
    executablePath: "/opt/pw-browsers/chromium",
    args: ["--no-sandbox"]
  });
  try {
    const page = await browser.newPage();
    page.on("console", msg => console.log("[page]", msg.text()));
    page.on("pageerror", err => console.error("[pageerror]", err));
    await page.goto(`http://127.0.0.1:${port}${path}`);
    // The measurement harness signals readiness explicitly; the
    // production site (index.html/404.html) has no such flag - `fn`
    // is responsible for waiting on whatever DOM/global state it needs
    if (path.endsWith("harness.html")) {
      await page.waitForFunction("window.hamrReady === true");
    }
    await fn(page);
  } finally {
    await browser.close();
    server.close();
  }
}
