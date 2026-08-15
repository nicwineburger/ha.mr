# GPU training campaign — results summary

Everything below was produced on Modal (serverless GPU) between
2026-08-14 14:20Z and 2026-08-15; methodology, corpus construction,
and per-stage details are in [README.md](README.md), raw rows with
configs and full val curves in [results.jsonl](results.jsonl).

## Deliverables (on this branch, not integrated)

- `model/url-model-next.bin` — **the recommended candidate**:
  the shipped architecture (dim 192, 5 layers, 6 heads, mlp 576,
  ctx 96, vocab-1024 relearned) trained 3B tokens on the 19.8M-URL
  corpus. `link_version: 2`, 4.13MB.
- `model/url-model-next-distilled.bin` — second candidate, same
  architecture distilled from a 71.7M-param teacher. Ties the plain
  run (see below); shipped for the coordinator's head-to-head.

Neither `url-model.bin` nor payload versioning was touched;
integration goes through the documented upgrade path in
`model/README.md`.

## Final numbers (real coder, 1,000 shared-holdout URLs, Node)

| model | size | avg hybrid | vs classic (40.5) | neural chosen |
|---|---|---|---|---|
| shipped v1 | 4.1MB | 24.7 | −39.0% | 95.2% |
| **url-model-next** | 4.1MB | **19.5** | **−51.8%** | 98.6% |
| url-model-next-distilled | 4.1MB | 19.5 | −51.9% | 98.7% |

All 1,000 round-trips verified through `compressHybrid` /
`decompressHybrid` for both candidates (0 failures). Same size, same
decode speed as shipped — payloads ~21% smaller end-to-end.

## Campaign table (shared 4,000-URL eval; est. symbols)

| run | params | tokens | bits/char | est hybrid | vs classic |
|---|---|---|---|---|---|
| shipped v1 (CPU, 75M tok) | 2.06M | 75M | 2.851 | 24.70 | −38.4% |
| g-base1024 | 2.06M | 500M | 2.364 | 20.81 | −48.1% |
| g-base2048 | 2.26M | 500M | 2.370 | 20.80 | −48.2% |
| g-mid1024 | 3.26M | 500M | 2.297 | 20.26 | −49.5% |
| g-big1024 | 4.22M | 500M | 2.264 | 20.00 | −50.2% |
| **g-final1024** → url-model-next | 2.06M | 3B | 2.2013 | 19.36 | −51.4% |
| g-distill1024 → …-distilled | 2.06M | 3B | 2.1985 | 19.32 | −51.5% |
| g-teacher768 (never ships) | 71.7M | 3B | 1.5693 | 14.20 | — |

The shipped model's 2.851 here (vs 2.336 in the CPU campaign) is a
distribution note, not a regression: the old holdout shared hosts
with its training blocks; this eval set samples the host space 16×
more broadly. Every row above uses the same eval set.

## Findings

1. **Data at fixed size was the campaign's win**: same 2.06M
   architecture, 16× corpus, 40× tokens → −38.4% → −51.4%.
2. **vocab-2048 still loses** even relearned on the big corpus:
   quality ties vocab-1024 at +0.4MB of download.
3. **The ~2M knee holds at GPU scale**: 4.22M params buys −2.1pp more
   (est.) but 2× download and ~2× slower pure-JS decode. Not worth it
   for a client-side model; revisit only if payload size outranks
   asset size someday.
4. **Distillation from a 35× teacher adds nothing at this scale**:
   the KD run led by ~0.02–0.03 nats mid-training, but the advantage
   vanished by the end of the schedule (val 3.164 vs 3.169; real-coder
   results identical). With 3B tokens of real URLs available, the
   2.06M student is capacity-limited, not signal-limited. The teacher
   itself (1.569 bits/char) shows what ~5.5MB-class future budgets
   could chase.
5. **Training-side numerics don't need browser constraints**: bf16
   autocast, TF32, and GPU eval all reproduced CPU-recipe quality;
   only inference determinism matters, and both exports load and
   round-trip in `neural.js` unmodified.

## Val-loss curves (nats/token, packed val subset)

| step | g-final1024 | g-distill1024 | g-teacher768 |
|---|---|---|---|
| 2,000 | 3.893 | 3.827 | 4.670 |
| 16,000 | 3.363 | 3.318 | 2.784 |
| 32,000 | 3.278 | 3.257 | 2.443 |
| 48,000 | 3.202 | 3.192 | 2.276 |
| 61,035 | 3.169 | 3.164 | 2.224 |

g-final1024 was still improving slowly at budget end (−0.033 nats
over the last quarter); a 2× budget might buy another ~0.2 est.
symbols. Teacher first attempt at peak LR 3e-3 stalled on a high
plateau and was restarted at 1e-3/3000-step warmup (~$1.4 written
off); it later survived a driver restart by resuming from its volume
checkpoint at step 48K.

## Modal cost (estimates from run durations; dashboard is authoritative)

| item | GPU/CPU time | est. cost |
|---|---|---|
| corpus fetch+build+pack (CPU containers) | ~26 core-h + 64GB-RAM hours | ~$5 |
| screening ×4 + shipped eval (A10G) | ~1.4h | ~$1.6 |
| g-final1024 (A10G) | 1.5h | ~$1.7 |
| g-teacher768 incl. failed LR attempt (A100-40GB) | ~4.8h | ~$10.1 |
| g-distill1024 (A100-40GB) | 1.8h | ~$3.8 |
| exports, evals, volume storage | — | ~$1 |
| **total** | | **~$23** |

Under the revised $50 cap (original $20 cap + teacher/distill
extension). No apps left running. The `hamr-gpu` volume (~9GB:
corpus, packed windows, vocabs, checkpoints) is retained for
follow-up training; delete it if no more runs are planned — storage
is pennies/month. The teacher's f16 weights are also archived on
this branch under `teacher/` (split for GitHub's file-size limit),
so only the fp32 optimizer-state checkpoint and the corpus would be
lost with the volume.
