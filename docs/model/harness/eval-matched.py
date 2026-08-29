"""
Coverage-aware comparison of screened variants: for every holdout URL,
each variant contributes min(classic_size, est_neural_size), with
classic alone where the URL doesn't fit the variant's context. This is
the payload size users actually see, and unlike raw per-variant
averages it's comparable across variants with different coverage.

Prereqs: results.jsonl + ckpt-<name>.pt from experiment.py runs, and
classic sizes: node classic-sizes.mjs data/corpus-holdout.txt data/classic.json
"""
import json
import math
import os

import torch
import torch.nn.functional as F

import experiment as ex

classic = json.load(open(os.path.join(ex.DATA, "classic.json")))["sizes"]
urls_all = [u.strip() for u in open(os.path.join(ex.DATA, "corpus-holdout.txt")) if u.strip()]
urls_all = [u for u in urls_all if u in classic][:4000]
classic_avg = sum(classic[u] for u in urls_all) / len(urls_all)
LOG2E = 1 / math.log(2)

seen = set()
for line in open(os.path.join(ex.HERE, "results.jsonl")):
    r = json.loads(line)
    cfg = r["config"]
    name = cfg["name"]
    if name in seen: continue
    seen.add(name)
    ckpt = os.path.join(ex.HERE, f"ckpt-{name}.pt")
    if not os.path.exists(ckpt):
        continue
    tok = ex.make_tokenizer(cfg["tokenizer"])
    model = ex.build_model(cfg, tok.size)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    est = {}
    with torch.no_grad():
        batch, metas = [], []
        def flush():
            if not batch: return
            maxlen = max(len(s) for s in batch)
            x = torch.zeros(len(batch), maxlen, dtype=torch.int64)
            for i, s in enumerate(batch):
                x[i, :len(s)] = torch.tensor(s)
            logp = F.log_softmax(model(x), dim=-1)
            for i, s in enumerate(batch):
                bits = 0.0
                for t in range(len(s) - 1):
                    bits -= logp[i, t, s[t + 1]].item() * LOG2E
                est[metas[i]] = ex.estimate_symbols(bits)
            batch.clear(); metas.clear()
        for u in urls_all:
            try:
                ids = tok.encode(ex.strip_scheme(u))
            except ValueError:
                continue
            seq = [tok.eos] + ids + [tok.eos]
            if len(seq) > cfg["ctx"]:
                continue
            batch.append(seq); metas.append(u)
            if len(batch) == 64: flush()
        flush()

    hybrid, covered, won = [], 0, 0
    for u in urls_all:
        c, n = classic[u], est.get(u)
        if n is not None:
            covered += 1
            if n < c: won += 1
            hybrid.append(min(c, n))
        else:
            hybrid.append(c)
    avg = sum(hybrid) / len(hybrid)
    print(json.dumps({
        "name": name,
        "params": r["params"],
        "bits_per_char": r["bits_per_char"],
        "coverage": round(covered / len(urls_all), 3),
        "neural_win_rate": round(won / max(1, covered), 3),
        "avg_hybrid_symbols": round(avg, 2),
        "vs_classic": f"{100 * (1 - avg / classic_avg):.1f}%"
    }))

print(f"\nclassic avg: {classic_avg:.2f} over {len(urls_all)} URLs")
