"""
Deterministic experiment harness for URL-compression model variants.

Any variant is a config: tokenizer (char or learned vocab file), model
architecture, context length, and training-token budget. Data splits
are hash-based (see build-corpus.py) so every experiment trains and
evaluates on exactly the same URLs. Seeds fix init and batch order.

Metrics (on the shared holdout):
- bits/char: total model cross-entropy over the holdout / total chars
- est. payload symbols: per-URL bits + format overhead, converted to
  base-85 symbol count - directly comparable to classic payload sizes
  (compute those once with: node classic-sizes.mjs)

CLI: python3 experiment.py '{"name": ..., "tokenizer": "char"|path,
  "dim": 128, "layers": 4, "heads": 4, "mlp": 384, "ctx": 96,
  "train_tokens": 25000000, "seed": 1234}'
Appends one JSON line to results.jsonl and saves ckpt-<name>.pt.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import GreedyTokenizer, CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def make_tokenizer(spec):
    if spec == "char":
        return CharTokenizer()
    with open(os.path.join(DATA, spec) if not os.path.isabs(spec) else spec) as f:
        return GreedyTokenizer(f.read().splitlines())


def strip_scheme(u):
    if u.startswith("https://"): return u[8:]
    if u.startswith("http://"): return u[7:]
    return u


def load_train_urls(seed):
    urls = []
    with open(os.path.join(DATA, "corpus-train.txt")) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            pop, u = line[0] == "P", strip_scheme(line[2:])
            urls.append(u)
            if pop:  # popular-site URLs upweighted 3x
                urls.append(u); urls.append(u)
    np.random.RandomState(seed).shuffle(urls)
    return urls


def pack(urls, tok, ctx, limit_tokens):
    """EOS-separated training windows aligned to URL starts."""
    rows, buf, total = [], [], 0
    for u in urls:
        try:
            ids = tok.encode(u)
        except ValueError:
            continue
        if not buf: buf = [tok.eos]
        buf += ids + [tok.eos]
        if len(buf) >= ctx + 1:
            rows.append(buf[:ctx + 1])
            total += ctx
            buf = []
            if limit_tokens and total >= limit_tokens * 1.05:
                break
    return np.array(rows, dtype=np.int64)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g


def build_model(cfg, vocab_size):
    DIM, HEADS, MLP = cfg["dim"], cfg["heads"], cfg["mlp"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(DIM)
            self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
            self.proj = nn.Linear(DIM, DIM, bias=False)
            self.norm2 = RMSNorm(DIM)
            self.up = nn.Linear(DIM, MLP, bias=False)
            self.down = nn.Linear(MLP, DIM, bias=False)
        def forward(self, x):
            B, T, C = x.shape
            h = self.norm1(x)
            q, k, v = self.qkv(h).split(DIM, dim=2)
            q = q.view(B, T, HEADS, C // HEADS).transpose(1, 2)
            k = k.view(B, T, HEADS, C // HEADS).transpose(1, 2)
            v = v.view(B, T, HEADS, C // HEADS).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.transpose(1, 2).contiguous().view(B, T, C))
            x = x + self.down(F.relu(self.up(self.norm2(x))))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, DIM)
            self.pos = nn.Embedding(cfg["ctx"], DIM)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg["layers"])])
            self.norm = RMSNorm(DIM)
        def forward(self, ids):
            B, T = ids.shape
            x = self.embed(ids) + self.pos(torch.arange(T)).unsqueeze(0)
            for b in self.blocks:
                x = b(x)
            return self.norm(x) @ self.embed.weight.T

    return Model()


def train(cfg, tok):
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("threads", 4))
    ctx = cfg["ctx"]
    data = pack(load_train_urls(seed), tok, ctx, cfg["train_tokens"])
    model = build_model(cfg, tok.size)
    n_params = sum(p.numel() for p in model.parameters())
    batch = cfg.get("batch", 64)
    lr_max = cfg.get("lr", 3e-3)
    total_steps = cfg["train_tokens"] // (batch * ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    warmup = min(200, total_steps // 20)
    print(f"[{cfg['name']}] vocab={tok.size} params={n_params/1e3:.0f}K "
          f"windows={len(data)} steps={total_steps}", flush=True)
    t0 = time.time()
    perm = np.random.RandomState(seed + 1).permutation(len(data))
    for step in range(total_steps):
        lr = lr_max * (step + 1) / warmup if step < warmup else \
            lr_max * 0.1 + 0.45 * lr_max * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g in opt.param_groups: g["lr"] = lr
        idx = perm[(step * batch) % max(1, len(data) - batch):][:batch]
        b = torch.from_numpy(data[idx])
        x, y = b[:, :-1], b[:, 1:]
        loss = F.cross_entropy(model(x).reshape(-1, tok.size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  step {step+1}/{total_steps} loss {loss.item():.4f} "
                  f"({(step+1)*batch*ctx/el/1e3:.0f}K tok/s)", flush=True)
    return model, n_params, time.time() - t0


def round_to_f16(model):
    """The browser computes with f16-rounded weights; evaluate the same."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.from_numpy(p.numpy().astype(np.float16).astype(np.float32)))


def holdout_bits(model, tok, ctx, path, limit=4000):
    """Per-URL (url, chars, model bits) over the holdout file."""
    model.eval()
    urls = [strip_scheme(u.strip()) for u in open(path) if u.strip()][:limit]
    results = []
    LOG2E = 1 / math.log(2)
    with torch.no_grad():
        batch, metas = [], []
        def flush():
            if not batch: return
            maxlen = max(len(s) for s in batch)
            x = torch.zeros(len(batch), maxlen, dtype=torch.int64)
            for r, s in enumerate(batch):
                x[r, :len(s)] = torch.tensor(s)
            logp = F.log_softmax(model(x), dim=-1)
            for r, s in enumerate(batch):
                bits = 0.0
                for t in range(len(s) - 1):
                    bits -= logp[r, t, s[t + 1]].item() * LOG2E
                results.append((metas[r], len(metas[r]), bits))
            batch.clear(); metas.clear()
        for u in urls:
            try:
                ids = tok.encode(u)
            except ValueError:
                continue
            seq = [tok.eos] + ids + [tok.eos]
            if len(seq) > ctx: continue
            batch.append(seq); metas.append(u)
            if len(batch) == 64: flush()
        flush()
    model.train()
    return results


LOG2_85 = math.log2(85)
OVERHEAD_BITS = 6  # sentinel + https flag + version marker + coder flush


def estimate_symbols(bits):
    return math.floor((bits + OVERHEAD_BITS) / LOG2_85) + 1


def main():
    cfg = json.loads(sys.argv[1])
    tok = make_tokenizer(cfg["tokenizer"])
    model, n_params, train_time = train(cfg, tok)
    round_to_f16(model)
    rows = holdout_bits(model, tok, cfg["ctx"],
                        os.path.join(DATA, "corpus-holdout.txt"))
    total_bits = sum(b for _, _, b in rows)
    total_chars = sum(c for _, c, _ in rows)
    symbols = [estimate_symbols(b) for _, _, b in rows]
    result = {
        "name": cfg["name"],
        "config": cfg,
        "params": n_params,
        "model_mb_f16": round(n_params * 2 / 1e6, 2),
        "train_minutes": round(train_time / 60, 1),
        "holdout_urls": len(rows),
        "bits_per_char": round(total_bits / total_chars, 4),
        "avg_est_symbols": round(sum(symbols) / len(symbols), 2),
    }
    print(json.dumps(result))
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    torch.save(model.state_dict(), os.path.join(HERE, f"ckpt-{cfg['name']}.pt"))


if __name__ == "__main__":
    main()
