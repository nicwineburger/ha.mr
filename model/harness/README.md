# Model experiment harness

Tools for training and comparing URL-compression model variants under
identical, deterministic conditions. This is how the shipped model was
chosen; use it to evaluate any future variant against the same data.

## Determinism

- **Splits are content-hashed**: a URL's membership in train/val/holdout
  depends only on `sha1(url)`, so splits survive re-fetching, re-ordering,
  and corpus growth (`build-corpus.py`).
- **Seeds fix everything else**: model init, data shuffle, batch order.
- **Evaluation is shared**: every variant scores the same holdout URLs
  against the same classic-scheme baselines.

## Workflow

```sh
./fetch-data.sh        # corpus + vocab + classic baselines (~500MB download)
python3 experiment.py '{"name":"my-variant","tokenizer":"vocab-1024.txt",
  "dim":128,"layers":4,"heads":4,"mlp":384,"ctx":96,"train_tokens":25000000}'
python3 eval-matched.py   # coverage-aware comparison of all runs
python3 train-final.py    # trains the shipped config, exports ../url-model.bin
```

`experiment.py` reports bits/char and estimated payload symbols;
`eval-matched.py` reports the number that actually matters - the average
hybrid payload (`min(classic, neural)` per URL, classic where a URL
doesn't fit the variant) over the shared holdout, plus coverage and
neural win rate.

## Screening campaign results (2026-08)

Corpus: 1.68M train URLs (102M chars) sampled across the Common Crawl
`CC-MAIN-2025-26` URL index + ada-url/url-dataset (popular sites,
upweighted 3x); ~34K held out by hash. All variants trained with a
25M-token budget on 4 CPU cores; evaluated on the same 3,997 holdout
URLs (classic average: 39.85 symbols).

| variant | tokenizer | params | f16 size | bits/char | coverage | avg hybrid | vs classic |
|---|---|---|---|---|---|---|---|
| char-base | char | 685K | 1.4MB | 2.920 | 97.7% | 26.42 | -33.7% |
| bpe1k | greedy-1024 | 800K | 1.6MB | 2.923 | 99.9% | 25.38 | -36.3% |
| bpe2k | greedy-2048 | 931K | 1.9MB | 2.921 | 99.9% | 25.32 | -36.5% |
| **bpe1k-big** | greedy-1024 | **2.06M** | **4.1MB** | **2.731** | 99.9% | **23.83** | **-40.2%** |
| bpe1k-xl | greedy-1024 | 4.22M | 8.4MB | 2.655 | 99.9% | 23.20 | -41.8% |

Findings:
- **Tokenization is coverage, not quality**: at equal capacity, BPE-style
  vocabs matched char-level bits/char, but their ~2.1 chars/token
  stretches the 96-token context to ~200 characters of URL, lifting
  neural coverage from 97.7% to 99.9% - worth ~2.5pp of hybrid savings
  on its own (and it halves inference steps per URL).
- **Capacity is quality**: 800K -> 2.06M params bought 3.9pp; 2.06M ->
  4.22M bought only 1.6pp more while doubling download size and halving
  inference speed. The knee is at ~2M params.
- **Shipped config**: `bpe1k-big` (dim 192, 5 layers, 6 heads, mlp 576,
  ctx 96, vocab 1024), long-trained at 75M tokens (`train-final.py`).

Estimated-symbol figures track the real pipeline closely (the estimate
is model cross-entropy + 6 bits of format overhead, base-85-ized); final
numbers in the PR/README come from the real coder via
`../benchmark.mjs`, which also verifies round-trips.

Not pursued, and why:
- **Domain-conditioned splitting** (classic domain + neural path):
  the hybrid already picks the better whole-URL scheme per link, and
  neural wins 98% of the time on the holdout, so the headroom from
  splitting within a URL is small relative to its format complexity.
- **Larger vocabularies**: 2048 matched 1024 at this capacity; vocab
  rows compete with transformer weights for the parameter budget.
- **GPU-scale training**: out of scope for this environment, but the
  harness is the right starting point - the same configs train
  unchanged on CUDA, and the scaling curve suggests a GPU-trained
  ~2M-param model on 50x the data is the next real win.
