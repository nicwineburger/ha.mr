"""
Learns a URL token vocabulary by BPE over separator-delimited chunks,
for use with GREEDY LONGEST-MATCH tokenization (not merge-order BPE).

Greedy longest-match is what ships: it's ~15 identical lines in Python
and JavaScript, trivially deterministic, and needs only the vocab
string list. BPE here is just the method for choosing good strings.

Chunks: URLs are split so each separator [/?&=.#] starts a new chunk
and stays attached to it ("en", ".wikipedia", ".org", "/wiki", ...).
Merges never cross chunk boundaries, which keeps the pair statistics
tractable and yields natural URL units (".com", "/index.php", "?v=").
"""
import re
import sys
from collections import Counter, defaultdict

def chunks_of(url):
    return [c for c in re.split(r"(?=[/?&=.#])", url) if c]

def learn_vocab(urls, vocab_size):
    # Base vocabulary: EOS placeholder handled by caller; all single chars
    base = [chr(c) for c in range(0x21, 0x7f)]

    words = Counter()
    for u in urls:
        for c in chunks_of(u):
            words[c] += 1

    # word -> list of current symbols
    seqs = {w: list(w) for w in words}
    pair_counts = Counter()
    pair_words = defaultdict(set)
    for w, seq in seqs.items():
        n = words[w]
        for a, b in zip(seq, seq[1:]):
            pair_counts[(a, b)] += n
            pair_words[(a, b)].add(w)

    merges = []
    target = vocab_size - len(base) - 1  # -1 for EOS
    while len(merges) < target and pair_counts:
        (a, b), count = pair_counts.most_common(1)[0]
        if count < 2:
            break
        merged = a + b
        merges.append(merged)
        for w in list(pair_words[(a, b)]):
            seq = seqs[w]
            n = words[w]
            i = 0
            out = []
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            # update pair statistics for this word
            for x, y in zip(seq, seq[1:]):
                pair_counts[(x, y)] -= n
                if pair_counts[(x, y)] <= 0: del pair_counts[(x, y)]
                pair_words[(x, y)].discard(w)
            for x, y in zip(out, out[1:]):
                pair_counts[(x, y)] += n
                pair_words[(x, y)].add(w)
            seqs[w] = out

    # Final vocab: single chars + merged strings.
    # Order doesn't matter for greedy longest-match; sort for determinism.
    return sorted(base + merges)

class GreedyTokenizer:
    """Greedy longest-match tokenizer. EOS = id 0; strings from id 1."""
    def __init__(self, vocab):
        self.vocab = vocab                    # list of strings
        self.ids = {s: i + 1 for i, s in enumerate(vocab)}
        self.max_len = max(len(s) for s in vocab)
        self.eos = 0
        self.size = len(vocab) + 1

    def encode(self, text):
        out = []
        i = 0
        while i < len(text):
            for l in range(min(self.max_len, len(text) - i), 0, -1):
                tid = self.ids.get(text[i:i+l])
                if tid is not None:
                    out.append(tid)
                    i += l
                    break
            else:
                raise ValueError(f"untokenizable char: {text[i]!r}")
        return out

    def decode(self, ids):
        return "".join(self.vocab[i - 1] for i in ids)

class CharTokenizer:
    """v1-style char tokenizer: EOS = 0, chars 0x21..0x7E -> 1..94."""
    def __init__(self):
        self.eos = 0
        self.size = 95
        self.max_len = 1
        self.vocab = [chr(c) for c in range(0x21, 0x7f)]
    def encode(self, text):
        return [ord(c) - 0x20 for c in text]
    def decode(self, ids):
        return "".join(chr(i + 0x20) for i in ids)

if __name__ == "__main__":
    infile, outfile, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    urls = []
    with open(infile) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u = line[2:] if line[1] == " " else line
            u = u.removeprefix("https://").removeprefix("http://")
            urls.append(u)
    # Learning on a sample is fine; frequencies are what matter
    sample = urls[:400000]
    vocab = learn_vocab(sample, size)
    with open(outfile, "w") as f:
        f.write("\n".join(vocab))
    print(f"learned {len(vocab)} tokens (+1 EOS = {len(vocab)+1})")
    tok = GreedyTokenizer(vocab)
    total_chars = sum(len(u) for u in urls[:20000])
    total_toks = sum(len(tok.encode(u)) for u in urls[:20000])
    print(f"compression: {total_chars/total_toks:.2f} chars/token")
