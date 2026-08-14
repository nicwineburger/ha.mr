import { compressHybrid, decompressHybrid, payloadSchemeVersion } from "./hybrid.js";
import { URLModel } from "./neural.js";
import {
  outputAlphabetASCII,
  outputAlphabetQR,
  outputAlphabetEmoji
} from "./alphabets.js";

/**
 * The neural model loads in the background. Until it's ready (or if
 * the file is missing), everything runs on the classic scheme.
 */
let model = null;
const modelReady = (async () => {
  try {
    const response = await fetch("/model/url-model.bin");
    if (!response.ok) throw `HTTP ${response.status}`;
    model = new URLModel(await response.arrayBuffer());
    // Upgrade whatever is currently displayed
    updateOutput();
  } catch (e) {
    console.warn("Neural model unavailable, using classic compression only:", e);
  }
})();

/**
 * The site adapts to whatever domain it's hosted on: output links and
 * the displayed title are derived from the current origin, so forks
 * don't need to change any code. When there's no usable origin (e.g.
 * the page was opened from disk), fall back to the canonical domain.
 */
const isHosted = location.protocol === "http:" || location.protocol === "https:";
const siteHost = (isHosted && location.host) || "ha.mr";
const siteOrigin = isHosted && location.host
  ? `${location.protocol}//${location.host}`
  : `https://${siteHost}`;

if (siteHost !== "ha.mr") {
  document.title = `${siteHost} - link compressor`;
  document.querySelector(".title").textContent = siteHost;
  // The pronunciation hint only makes sense for the original domain
  document.querySelector(".subtitle").style.display = "none";
}

const settings = {
  emoji: false,
  qr: false
};

const settingsElements = {
  emoji: "#settings-emoji",
  qr: "#settings-qr"
};

for (const setting in settingsElements) {
  const element = document.querySelector(settingsElements[setting]);
  settings[setting] = element.checked;
  element.addEventListener("change", (event) => {
    settings[setting] = element.checked;
    updateOutput();
  });
}

function countSymbols (string, alphabet) {
  let count = 0;
  while (string) {
    const symbol = alphabet.find(c => string.endsWith(c));
    string = string.slice(0, symbol ? -symbol.length : -1);
    count ++;
  }
  return count;
}

const inputLinkElement = document.querySelector("#input-link");
const outputLinkElement = document.querySelector("#output-link");
const outputRatioElement = document.querySelector("#output-ratio");
const queryWarningElement = document.querySelector("#query-warning");

const qrCodeImage = document.querySelector("#qrcode");
const qrCodeCorrectionLevelContainer = document.querySelector("#qr-correct-level-container");
const qrCodeCorrectionLevelElement = document.querySelector("#qr-correct-level");
qrCodeCorrectionLevelElement.addEventListener("change", updateOutput);

/**
 * Neural encoding takes a few hundred milliseconds, too slow to run on
 * every keystroke. Each input change renders the classic result
 * immediately, then upgrades to the hybrid (best-of-both) result once
 * the input has been idle briefly.
 */
let neuralTimer = null;
function updateOutput () {
  renderOutput(null);
  if (model) {
    clearTimeout(neuralTimer);
    neuralTimer = setTimeout(() => renderOutput(model), 200);
  }
}

function renderOutput (activeModel) {
  const input = inputLinkElement.value.trim();
  try {
    const alphabet = settings.emoji ? outputAlphabetEmoji : outputAlphabetASCII;
    const output = compressHybrid(input, alphabet, activeModel);
    let inputNormalized = input;
    if (input.startsWith("https://")) {
      inputNormalized = input.slice(8);
    } else if (input.startsWith("http://")) {
      inputNormalized = input.slice(7);
    }
    let excessiveParams = false;
    if (URL.canParse("http://" + inputNormalized)) {
      const url = new URL("http://" + inputNormalized);
      if (url.searchParams.size > 1) {
        excessiveParams = true;
      }
    }
    if (excessiveParams) {
      queryWarningElement.style.display = "inline";
    } else {
      queryWarningElement.style.display = "none";
    }
    // Overhead of the short link, not counting the protocol: the host
    // plus the "#" separator
    const ratio = (1 - (countSymbols(output, alphabet) + siteHost.length + 1) / inputNormalized.length) * 100;
    if (ratio < -300) {
      outputRatioElement.textContent = `Output is much larger than the input`;
      outputRatioElement.style.color = "rgb(255, 50, 50)";
    } else if (ratio < 0) {
      outputRatioElement.textContent = `Output is ${Math.floor(-ratio)}% larger than the input`;
      outputRatioElement.style.color = "rgb(255, 50, 50)";
    } else if (ratio > 0) {
      outputRatioElement.textContent = `Output is ${Math.ceil(ratio)}% smaller than the input`;
      outputRatioElement.style.color = "rgb(15, 190, 15)";
    } else {
      outputRatioElement.textContent = "Output is the same length as the input";
      outputRatioElement.style.color = "gray";
    }
    outputLinkElement.textContent = `${siteOrigin}#${output}`;
    outputLinkElement.href = `${siteOrigin}#${output}`;
    outputLinkElement.style.color = "";
    if (settings.qr) {
      const errorCorrection = ["L", "M", "Q", "H"][qrCodeCorrectionLevelElement.value];
      qrCodeImage.style.display = "inline";
      qrCodeCorrectionLevelContainer.style.display = "inline";
      // Uppercase keeps the QR code in alphanumeric mode; hostnames
      // only contain [a-z0-9.-], which all fit that character set
      let qrCodeLink = `HTTP://${siteHost.toUpperCase()}/${compressHybrid(input, outputAlphabetQR, activeModel)}`;
      QRCode.toDataURL(qrCodeLink, {
        errorCorrectionLevel: errorCorrection,
        scale: 8
      }, (err, url) => {
        if (err) {
          qrCodeImage.style.display = "none";
          qrCodeCorrectionLevelContainer.style.display = "none";
          return;
        }
        qrCodeImage.src = url;
        qrCodeImage.title = qrCodeLink;
      });
    } else {
      qrCodeImage.style.display = "none";
      qrCodeCorrectionLevelContainer.style.display = "none";
    }
  } catch (e) {
    if (!input.trim()) {
      outputLinkElement.textContent = "Enter a link above to compress";
    } else {
      outputLinkElement.textContent = "Invalid link";
      outputLinkElement.style.color = "rgb(255, 50, 50)";
      console.error(e);
    }
    qrCodeImage.style.display = "none";
    qrCodeCorrectionLevelContainer.style.display = "none";
    outputRatioElement.style.color = "rgba(255, 255, 255, 0)";
    outputLinkElement.removeAttribute("href");
    queryWarningElement.style.display = "none";
  }
}
inputLinkElement.addEventListener("input", updateOutput);

(async () => {
  let payload = null;
  let alphabet = outputAlphabetASCII;

  // Get hash value of current address bar
  if (window.location.hash) {
    // Decode hash value in case it's non-ASCII
    payload = decodeURIComponent(window.location.hash.slice(1));
    // Remove all whitespace - we never use whitespace when encoding hash values
    payload = payload.replace(/\s+/g, "");
    // Check if input is pure ASCII - potentially unreliable?
    const useEmoji = Array.from(payload).some(c => !outputAlphabetASCII.includes(c));
    alphabet = useEmoji ? outputAlphabetEmoji : outputAlphabetASCII;
  } else {
    // If no hash value, we're likely reading a QR code
    // For that, use the path instead
    payload = decodeURIComponent(window.location.pathname.slice(1));
    alphabet = outputAlphabetQR;
  }

  if (payload && payload.trim()) {
    try {
      // Classic payloads redirect immediately; neural ones need the
      // model matching their payload version. The latest model is
      // already being fetched; links made by older models lazy-load
      // their archived model file instead.
      let decodeModel = null;
      const version = payloadSchemeVersion(payload, alphabet);
      if (version >= 1) {
        await modelReady;
        decodeModel = model;
        if (!decodeModel || decodeModel.linkVersion !== version) {
          const response = await fetch(`/model/url-model-v${version}.bin`);
          if (!response.ok) throw `HTTP ${response.status} fetching model version ${version}`;
          decodeModel = new URLModel(await response.arrayBuffer());
        }
      }
      const target = decompressHybrid(payload, alphabet, decodeModel);
      window.location.href = target;
      return;
    } catch (e) {
      console.warn(`Redirect failed. Could not decode input.`);
      console.error(e);
    }
  }

  updateOutput();

  document.querySelector("#loader").style.opacity = 0;
  document.querySelector("#content").style.opacity = 1;
  document.querySelector("#content").style.pointerEvents = "auto";

})();
