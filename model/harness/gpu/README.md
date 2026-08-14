# GPU training campaign (Modal)

Scales the [CPU screening campaign](../README.md) to a ~20M-URL corpus
and GPU training on [Modal](https://modal.com) serverless A10Gs. The
candidate model produced here is `../../url-model-next.bin` (payload
`link_version: 2`); integration and the versioned upgrade ship
separately.

## Corpus (16x the shipped corpus)

`modal_corpus.py` samples **every 25th** `cluster.idx` block of the
Common Crawl `CC-MAIN-2025-26` URL index (the shipped corpus used
every 400th; 400 ≡ 0 mod 25, so the old blocks are a strict subset),
fetches and filters them across 96 parallel containers, and rebuilds
the corpus with the **identical** filters, per-host cap, and
sha1-bucket split as `../build-corpus.py` - split membership depends
only on the URL string, so no shared-holdout URL can enter training.

- 38,474 index blocks → 90.5M unique URLs → 1.91M hosts
- capped 30/host: **19.8M train URLs (1.20B chars)**, 204K val, 409K holdout
- vocabularies relearned on this corpus: vocab-1024 (2.14 chars/token),
  vocab-2048 (2.44 chars/token)
- packed windows (ctx 96): 440M tokens (vocab-1024), 396M (vocab-2048)

Everything lives on the Modal Volume `hamr-gpu`; raw index blocks are
filtered in-memory and never stored.

## Training (`modal_train.py`)

The harness architecture and packing, unchanged, on A10G: batch 512
(vs 64 on CPU) with sqrt-scaled peak LR 8e-3 (vs 3e-3), per-epoch
window reshuffle, periodic val loss. ~350K tokens/s for the 2.06M
model (~15 min per 500M-token screening run, ~$0.30).

## Screening at 500M tokens

Shared eval: the first 4,000 URLs of the new hash-bucket holdout
(classic average **40.12** symbols over 3,999 scoreable URLs -
statistically identical to the CPU campaign's 39.85). Estimated
symbols from model cross-entropy + 6 bits overhead, base-85-ized, as
in `../experiment.py`; hybrid = coverage-aware `min(classic, est)` per
URL via `eval-hybrid.py`.

| variant | params | f16 size | bits/char | avg hybrid | vs classic |
|---|---|---|---|---|---|
| shipped v1 (75M tok, CPU) | 2.06M | 4.1MB | 2.851 | 24.70 | -38.4% |
| g-base1024 | 2.06M | 4.1MB | 2.364 | 20.81 | -48.1% |
| g-base2048 | 2.26M | 4.5MB | 2.370 | 20.80 | -48.2% |
| g-mid1024 | 3.26M | 6.5MB | 2.297 | 20.26 | -49.5% |
| g-big1024 | 4.22M | 8.4MB | 2.264 | 20.00 | -50.2% |

The shipped model's 2.851 bits/char here (vs 2.336 on the old holdout)
is not a regression: the old holdout was drawn from the same 2.4K
index blocks as its training data, so it shared hosts with training;
this 38K-block holdout samples the host space 16x more broadly. All
rows above share the same eval set, so they are directly comparable.

Findings:
- **Data is the big lever at fixed size**: same 2.06M architecture,
  16x data → 24.70 → 20.81 est. hybrid symbols (-9.7pp vs classic).
- **vocab-2048 still doesn't pay**: even relearned on 16x data with
  better chars/token (2.44), it ties vocab-1024 at +0.4MB - same
  conclusion as the CPU campaign.
- **The ~2M knee holds**: capacity still helps (+1.4pp at 3.26M,
  +2.1pp at 4.22M) but with clearly diminishing symbols/MB (0.23 for
  base→mid, 0.14 for mid→big), while download size grows 1.6-2x and
  pure-JS decode speed drops proportionally. Following the campaign
  rule - maximal compression at minimal model size - the final run
  keeps the shipped architecture: a drop-in upgrade with zero
  size/speed cost.

## Final run

`g-final1024`: the shipped config (dim 192, 5 layers, 6 heads, mlp
576, ctx 96, vocab 1024) trained for 3B tokens (~6.8 epochs of the
440M-token pack), `link_version: 2`, exported as
`../../url-model-next.bin` (hamr-url-model-v2, f16). Final numbers -
real-coder round-trip benchmark on held-out URLs vs the shipped model
- are in the results summary on this branch.

Raw result rows (config JSON + metrics + val curves) are in
`results.jsonl`. Note: rows written by concurrently-finishing Modal
runs can clobber each other on volume commit (last writer wins); the
checked-in file was reconstructed from run logs where that happened.

## Reproducing

```sh
pip install 'modal[api-proxy-support]'
modal run modal_corpus.py --stage all      # ~30 min, mostly CPU containers
modal run modal_train.py --action eval-shipped
modal run modal_train.py --action train-many --config '[...]'   # screening
modal run modal_train.py --action train --config '{... "train_tokens": 3000000000, "link_version": 2}'
modal run modal_train.py --action export --config '{...}'
modal volume get hamr-gpu url-model-next.bin ../../url-model-next.bin
node benchmark-model.mjs ../../url-model-next.bin data/holdout-eval.txt 1000
```

Total campaign cost: see results summary (~$10 of Modal credit:
corpus build + 5 GPU runs + final).
