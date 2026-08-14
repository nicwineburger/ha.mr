# ha.mr
Compresses links and optimizes QR codes entirely in the browser, without a back-end database.

## How
1. Common parts of the link (e.g. protocol, `www.` prefix, `index.html`) are manually detected and reduced to individual bits. If present, the port is encoded as a raw numeric value.
2. Second-level and top-level domains are matched against a Huffman-coded dictionary of the most common websites and TLDs.
3. The rest of the link is split into parts, and each segment is either fitted to a predefined character set, or Huffman coded.
4. For links, the output is encoded in the full character set of a URL. (I've been informed that square brackets `[]` are not supposed to be a part of this set, but it's too late to change that now.)
5. For QR codes, the output uses the alphanumeric character set to remove overhead compared to other QR code generators.

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
- [Hammersmith One](https://fonts.google.com/specimen/Hammersmith+One) by Sorkin Type Co, self-hosted under the [SIL Open Font License](fonts/OFL.txt)
