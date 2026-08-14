# ha.mr
Compresses links and optimizes QR codes entirely in the browser, without a back-end database.

## How
Every link is compressed with two schemes, and the smaller payload wins (a version marker in the payload records which one was used):

### Classic scheme
1. Common parts of the link (e.g. protocol, `www.` prefix, `index.html`) are manually detected and reduced to individual bits. If present, the port is encoded as a raw numeric value.
2. Second-level and top-level domains are matched against a Huffman-coded dictionary of the most common websites and TLDs.
3. The rest of the link is split into parts, and each segment is either fitted to a predefined character set, or Huffman coded.

### Neural scheme
A ~1.4MB character-level transformer (see [`model/README.md`](model/README.md)) predicts each character of the link, and an arithmetic coder turns those predictions into near-optimal bits. Inference runs in ~200 lines of dependency-free JavaScript restricted to IEEE correctly-rounded operations, so encoding and decoding are bit-identical on every browser and platform. The model ships with the site as a static file — any fork hosts it automatically. If it fails to load, everything falls back to the classic scheme.

### Output
- For links, the output is encoded in the full character set of a URL. (I've been informed that square brackets `[]` are not supposed to be a part of this set, but it's too late to change that now.)
- For QR codes, the output uses the alphanumeric character set to remove overhead compared to other QR code generators.

## Usage

### Website
Paste a link at [ha.mr](https://ha.mr) to get a compressed link back. Optionally enable emoji output or a QR code.

The compressed payload lives entirely inside the short link:
- Text links carry it in the fragment: `https://ha.mr#<payload>`
- QR codes carry it in the path: `HTTP://HA.MR/<payload>` (uppercase keeps the QR code in alphanumeric mode)

Opening either link decodes the payload in your browser and redirects — no server ever sees or stores the original link.

### Command line
With [Node.js](https://nodejs.org) installed:

```sh
node standalone.js <link> [ascii|qr|emoji]   # compress a link
node standalone.js "https://ha.mr#..."        # decode a compressed link
```

Or install it as a global `hamr` command with `npm install -g .`.

### Tests
```sh
npm test
```

## Hosting your own
The site is fully portable: nothing about the compression or the UI is tied to the `ha.mr` domain. Output links, QR codes, and the displayed title all adapt to whatever domain the site is served from.

To run your own instance:
1. Fork this repository and enable GitHub Pages (or copy the files to any static web host).
2. Replace the contents of `CNAME` with your own domain, or delete the file if you're not using a custom domain.
3. Serve the site from the **root** of the domain. Text links (`https://your.domain#...`) work from any path, but QR-code links carry their payload in the URL path and rely on the `404.html` fallback at the domain root to decode them.

Links are only decodable by a deployment of this codebase, but they are not tied to the domain that created them: the payload format is identical everywhere, so a link's path/fragment can be decoded by any instance (or by the CLI). The one caveat is the neural model: version 1 payloads can only be decoded with the exact `model/url-model.bin` that encoded them, so don't swap that file for a retrained one unless you're prepared to break your own previously issued links (see [`model/README.md`](model/README.md)).

For the command line tool, set the `HAMR_DOMAIN` environment variable to build and recognize short links on your domain:

```sh
HAMR_DOMAIN=your.domain node standalone.js "https://some-long.link/"
```

## Known normalizations
The compressor reproduces links exactly in the common case, but a few equivalent spellings are normalized (see `test/roundtrip.test.mjs` for the pinned-down list):
- Percent-escape hex is uppercased, and escapes of unreserved characters (e.g. `%7E`) are decoded.
- Query parameters without a value (`?foo`) gain an equals sign (`?foo=`).
- Stray `%` characters that aren't part of a valid escape are encoded as `%25`.
- Square brackets are percent-encoded.
- Repeated slashes in paths are collapsed.

QR-code links whose payload happens to contain a `/../` or `/./` sequence may be normalized away by the browser before decoding; this is rare, but such links can fail to resolve.

## Acknowledgements
- https://www.npmjs.com/package/qrcode (vendored as `qrcode.js`)
- https://github.com/smythp/reddit_links_dataset
- https://github.com/ada-url/url-dataset
- https://commoncrawl.org/ (URL index used as model training data)
- https://github.com/ansisg/hamr — the fork whose transformer + arithmetic-coding experiment inspired the neural scheme (reimplemented here with a ~200x smaller model and deterministic in-browser inference)
- [Hammersmith One](https://fonts.google.com/specimen/Hammersmith+One) by Sorkin Type Co, self-hosted under the [SIL Open Font License](fonts/OFL.txt)
