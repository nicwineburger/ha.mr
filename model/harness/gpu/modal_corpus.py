"""
Builds the GPU-campaign URL corpus on Modal (modal.com), scaled ~16x
from the CPU campaign: every 25th cluster.idx block instead of every
400th. Because 400 = 16 * 25, the shipped corpus's blocks (NR%400==3)
are a strict subset of this sampling (NR%25==3), and split membership
is still sha1(url)-hashed with the same bucket() and fractions as
../build-corpus.py - so no URL from the shared holdout can ever enter
this training set. That keeps every number comparable with the CPU
campaign in ../README.md.

Everything lands on the Modal Volume "hamr-gpu":
  blocks.txt                sampled (file, offset, length) index blocks
  raw/chunk-*.txt.gz        filtered URLs per fetch chunk
  corpus-{train,val,holdout}.txt   hash-split corpus (same format)
  holdout-eval.txt          first 4000 holdout URLs (shared eval set)
  vocab-{1024,2048}.txt     BPE-learned greedy-match vocabularies
  packed-{1024,2048}.npy    pre-packed uint16 training windows (ctx 96)
  packed-val-{1024,2048}.npy  val windows for during-training loss

Run stages (idempotent, each skips work already on the volume):
  modal run modal_corpus.py --stage sample
  modal run modal_corpus.py --stage fetch
  modal run modal_corpus.py --stage build
  modal run modal_corpus.py --stage pack
or --stage all for the whole pipeline.
"""
import os

import modal

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRAWL = "CC-MAIN-2025-26"
BASE = f"https://data.commoncrawl.org/cc-index/collections/{CRAWL}/indexes"
BLOCK_MODULUS = 25   # shipped corpus used 400; 400 % 25 == 0 -> superset
BLOCK_OFFSET = 3     # same offset as fetch-data.sh (awk NR % 400 == 3)
N_FETCH_CHUNKS = 96
CTX = 96
SEED = 1234

app = modal.App("hamr-corpus")
vol = modal.Volume.from_name("hamr-gpu", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy", "requests")
    # set at interpreter start so str-hash-dependent iteration orders
    # (set/dict) are reproducible across container runs
    .env({"PYTHONHASHSEED": "0"})
    .add_local_file(os.path.join(HARNESS, "vocab.py"), "/root/vocab.py")
)

VOL = "/vol"


def _url_ok(u):
    """Identical filter to ../build-corpus.py."""
    if not (10 <= len(u) <= 180):
        return False
    if not u.startswith(("http://", "https://")):
        return False
    if any(ord(c) < 0x21 or ord(c) > 0x7e for c in u):
        return False
    if u.lower().endswith("/robots.txt"):
        return False
    return True


@app.function(image=image, volumes={VOL: vol}, timeout=1800, memory=8192)
def sample_blocks():
    """Download cluster.idx and write the sampled block list."""
    import os
    import requests

    out = os.path.join(VOL, "blocks.txt")
    if os.path.exists(out):
        with open(out) as f:
            n = sum(1 for _ in f)
        print(f"blocks.txt exists: {n} blocks")
        return n
    print("downloading cluster.idx ...")
    r = requests.get(f"{BASE}/cluster.idx", stream=True, timeout=600)
    r.raise_for_status()
    n_lines = 0
    kept = []
    buf = b""
    for chunk in r.iter_content(1 << 22):
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            n_lines += 1  # 1-indexed like awk NR
            if n_lines % BLOCK_MODULUS == BLOCK_OFFSET:
                parts = line.decode().split("\t")
                kept.append(f"{parts[1]} {parts[2]} {parts[3]}")
    with open(out, "w") as f:
        f.write("\n".join(kept) + "\n")
    vol.commit()
    print(f"cluster.idx: {n_lines} lines -> {len(kept)} sampled blocks")
    return len(kept)


@app.function(image=image, volumes={VOL: vol}, timeout=3600, cpu=8,
              memory=16384, max_containers=32)
def fetch_chunk(chunk_id: int):
    """Fetch this chunk's share of index blocks, extract filtered URLs."""
    import gzip
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import requests

    out = os.path.join(VOL, "raw", f"chunk-{chunk_id:04d}.txt.gz")
    if os.path.exists(out):
        return (chunk_id, -1)
    with open(os.path.join(VOL, "blocks.txt")) as f:
        blocks = [l.split() for l in f.read().splitlines() if l]
    mine = blocks[chunk_id::N_FETCH_CHUNKS]

    sess = requests.Session()

    def fetch(b):
        fname, off, ln = b[0], int(b[1]), int(b[2])
        for attempt in range(4):
            try:
                r = sess.get(f"{BASE}/{fname}",
                             headers={"Range": f"bytes={off}-{off + ln - 1}"},
                             timeout=90)
                r.raise_for_status()
                return gzip.decompress(r.content).decode("utf-8", "ignore")
            except Exception:
                if attempt == 3:
                    return ""
        return ""

    urls = set()
    with ThreadPoolExecutor(24) as pool:
        for text in pool.map(fetch, mine):
            for line in text.splitlines():
                i = line.find("{")
                if i < 0:
                    continue
                try:
                    rec = json.loads(line[i:])
                except Exception:
                    continue
                if rec.get("status") != "200":
                    continue
                u = rec.get("url", "")
                if _url_ok(u):
                    urls.add(u)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wt") as f:
        f.write("\n".join(sorted(urls)))
    vol.commit()
    print(f"chunk {chunk_id}: {len(mine)} blocks -> {len(urls)} urls")
    return (chunk_id, len(urls))


@app.function(image=image, volumes={VOL: vol}, timeout=7200, cpu=16,
              memory=65536)
def build_corpus():
    """Dedupe, cap per host, hash-split, learn vocabs. Mirrors
    ../build-corpus.py exactly (bucket function, fractions, caps)."""
    import gzip
    import hashlib
    import os
    import random
    import subprocess
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/root")
    from vocab import learn_vocab, GreedyTokenizer

    if os.path.exists(os.path.join(VOL, "vocab-2048.txt")):
        print("corpus + vocabs already built")
        return

    random.seed(SEED)
    urls = set()
    raw = os.path.join(VOL, "raw")
    for fn in sorted(os.listdir(raw)):
        with gzip.open(os.path.join(raw, fn), "rt") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    urls.add(line)
    print(f"unique urls: {len(urls)}")

    byhost = defaultdict(list)
    for u in urls:
        byhost[u.split("/", 3)[2].lower()].append(u)
    print(f"hosts: {len(byhost)}")

    capped = []
    for host, us in byhost.items():
        us.sort()  # deterministic before shuffle
        random.shuffle(us)
        capped.extend(us[:30])
    del urls, byhost

    if not os.path.exists("/tmp/url-dataset"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ada-url/url-dataset",
                        "/tmp/url-dataset"], check=True)
    with open("/tmp/url-dataset/out.txt") as f:
        popular = [u.strip() for u in f if u.strip()]
    popular = [u for u in popular if _url_ok(u)]

    def bucket(u):
        return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 100

    train, val, hold = [], [], []
    popular_set = set(popular)
    for u in capped + popular:
        b = bucket(u)
        if b < 2:
            hold.append(u)
        elif b < 3:
            val.append(u)
        else:
            train.append((u, u in popular_set))

    random.shuffle(train)
    with open(os.path.join(VOL, "corpus-train.txt"), "w") as f:
        for u, pop in train:
            f.write(("P " if pop else ". ") + u + "\n")
    random.shuffle(hold)
    with open(os.path.join(VOL, "corpus-holdout.txt"), "w") as f:
        f.write("\n".join(hold))
    with open(os.path.join(VOL, "holdout-eval.txt"), "w") as f:
        f.write("\n".join(hold[:4000]))
    with open(os.path.join(VOL, "corpus-val.txt"), "w") as f:
        f.write("\n".join(val))
    print(f"train: {len(train)}, val: {len(val)}, holdout: {len(hold)}")
    print(f"train chars: {sum(len(u) for u, _ in train) / 1e6:.1f}M")
    vol.commit()

    # vocab learning, same recipe as ../vocab.py __main__: first 400K
    # train URLs (scheme stripped), BPE over separator chunks
    sample = []
    with open(os.path.join(VOL, "corpus-train.txt")) as f:
        for line in f:
            u = line.rstrip("\n")[2:]
            u = u.removeprefix("https://").removeprefix("http://")
            sample.append(u)
            if len(sample) >= 400000:
                break
    for size in (1024, 2048):
        v = learn_vocab(sample, size)
        with open(os.path.join(VOL, f"vocab-{size}.txt"), "w") as f:
            f.write("\n".join(v))
        tok = GreedyTokenizer(v)
        chars = sum(len(u) for u in sample[:20000])
        toks = sum(len(tok.encode(u)) for u in sample[:20000])
        print(f"vocab-{size}: {len(v)} tokens, {chars / toks:.2f} chars/token")
    vol.commit()


def _encode_worker_init(vocab_path):
    global _tok
    import sys
    sys.path.insert(0, "/root")
    from vocab import GreedyTokenizer
    with open(vocab_path) as f:
        _tok = GreedyTokenizer(f.read().splitlines())


def _encode_url(u):
    try:
        return _tok.encode(u)
    except ValueError:
        return None


@app.function(image=image, volumes={VOL: vol}, timeout=7200, cpu=16,
              memory=65536)
def pack_windows(vocab_size: int):
    """Tokenize the train + val corpora and pack EOS-separated ctx+1
    windows aligned to URL starts - the same packing as
    ../experiment.py pack(), streamed into a uint16 array.

    The 3x popular upweight and the seeded shuffle from
    experiment.load_train_urls() happen here, so training jobs just
    load the array. Window layout for a given (corpus, vocab, seed) is
    fixed across all training runs."""
    import os
    from multiprocessing import Pool

    import numpy as np

    out = os.path.join(VOL, f"packed-{vocab_size}.npy")
    if os.path.exists(out):
        print(f"{out} exists")
        return

    def load_urls(path, upweight):
        urls = []
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if upweight:
                    pop, u = line[0] == "P", line[2:]
                else:
                    pop, u = False, line
                u = u.removeprefix("https://").removeprefix("http://")
                urls.append(u)
                if pop:  # popular-site URLs upweighted 3x
                    urls.append(u)
                    urls.append(u)
        np.random.RandomState(SEED).shuffle(urls)
        return urls

    vocab_path = os.path.join(VOL, f"vocab-{vocab_size}.txt")

    def pack(urls, out_path, eos=0):
        buf = []
        # upper bound on windows: tokens never exceed chars + 2 EOS/url
        bound = (sum(len(u) for u in urls) + 2 * len(urls)) // (CTX + 1)
        rows = np.empty((bound + 1024, CTX + 1), dtype=np.uint16)
        n = 0
        with Pool(16, initializer=_encode_worker_init,
                  initargs=(vocab_path,)) as pool:
            for ids in pool.imap(_encode_url, urls, chunksize=4096):
                if ids is None:
                    continue
                if not buf:
                    buf = [eos]
                buf += ids + [eos]
                if len(buf) >= CTX + 1:
                    rows[n] = buf[:CTX + 1]
                    n += 1
                    buf = []
        rows = rows[:n]
        np.save(out_path, rows)
        print(f"{out_path}: {n} windows = {n * CTX / 1e6:.0f}M train tokens")

    pack(load_urls(os.path.join(VOL, "corpus-train.txt"), True), out)
    val_out = os.path.join(VOL, f"packed-val-{vocab_size}.npy")
    val_urls = load_urls(os.path.join(VOL, "corpus-val.txt"), False)[:200000]
    pack(val_urls, val_out)
    vol.commit()


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("sample", "all"):
        n = sample_blocks.remote()
        print(f"sampled blocks: {n}")
    if stage in ("fetch", "all"):
        results = list(fetch_chunk.map(range(N_FETCH_CHUNKS)))
        fresh = [r for r in results if r[1] >= 0]
        print(f"fetched {len(fresh)} chunks "
              f"({sum(r[1] for r in fresh)} urls pre-dedupe)")
    if stage in ("build", "all"):
        build_corpus.remote()
    if stage in ("pack", "all"):
        list(pack_windows.map([1024, 2048]))
