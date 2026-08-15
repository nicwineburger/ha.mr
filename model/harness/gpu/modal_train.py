"""
GPU training for the URL-compression model on Modal (A10G).

Same architecture, packing, seeds, and evaluation as ../experiment.py
(the model code is copied verbatim apart from a device-aware
torch.arange), trained from the pre-packed windows that
modal_corpus.py wrote to the "hamr-gpu" Volume. Differences from the
CPU harness, all recorded in each result row:
  - batch 512 (vs 64) with sqrt-scaled peak LR (8e-3 vs 3e-3)
  - windows reshuffled every epoch (the CPU loop reused one
    permutation; screening budgets never completed an epoch so this
    only matters for multi-epoch final runs)
  - periodic val loss on a fixed slice of packed val windows

Usage:
  modal run modal_train.py --action train --config '{"name": ...}'
  modal run modal_train.py --action eval-shipped
  modal run modal_train.py --action export --config '{"name": ..., ...}'

Results append to /vol/results.jsonl; per-URL estimated payload
symbols for the shared 4000-URL eval set go to /vol/est-<name>.json
(combined locally with classic sizes for the hybrid comparison).
"""
import json
import os

import modal

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CTX = 96
EOS = 0

app = modal.App("hamr-train")
vol = modal.Volume.from_name("hamr-gpu", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.8.0")
    .env({"PYTHONHASHSEED": "0"})
    .add_local_file(os.path.join(HARNESS, "vocab.py"), "/root/vocab.py")
)

VOL = "/vol"


def make_tokenizer(spec):
    import sys
    sys.path.insert(0, "/root")
    from vocab import GreedyTokenizer
    with open(os.path.join(VOL, spec)) as f:
        return GreedyTokenizer(f.read().splitlines())


def strip_scheme(u):
    if u.startswith("https://"): return u[8:]
    if u.startswith("http://"): return u[7:]
    return u


def build_model(cfg, vocab_size):
    """Copied from ../experiment.py build_model; the only change is
    device-aware torch.arange so it runs on CUDA. Parameter names and
    shapes are identical, so checkpoints interchange freely.

    cfg["qat"] = "int8" | "int4" switches the block Linears to
    fake-quantized forward passes (symmetric per-output-channel
    scales, straight-through estimator), matching the v3 export and
    the neural.js dequantizing loader. Embedding/positions/norms stay
    f16 - they are small and the tied head is quality-sensitive."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DIM, HEADS, MLP = cfg["dim"], cfg["heads"], cfg["mlp"]

    qat = cfg.get("qat")
    if qat:
        qmax = {"int8": 127, "int4": 7}[qat]
        qmin = {"int8": -127, "int4": -8}[qat]

        def fake_quant(w):
            scale = w.detach().abs().amax(dim=1, keepdim=True) \
                .clamp(min=1e-6) / qmax
            wq = (w / scale).round().clamp(qmin, qmax) * scale
            return w + (wq - w).detach()

        class Lin(nn.Linear):
            def forward(self, x):
                return F.linear(x, fake_quant(self.weight))
    else:
        Lin = nn.Linear

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.g = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(DIM)
            self.qkv = Lin(DIM, 3 * DIM, bias=False)
            self.proj = Lin(DIM, DIM, bias=False)
            self.norm2 = RMSNorm(DIM)
            self.up = Lin(DIM, MLP, bias=False)
            self.down = Lin(MLP, DIM, bias=False)
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
            x = self.embed(ids) + \
                self.pos(torch.arange(T, device=ids.device)).unsqueeze(0)
            for b in self.blocks:
                x = b(x)
            return self.norm(x) @ self.embed.weight.T

    return Model()


def round_to_f16(model):
    """The browser computes with f16-rounded weights; evaluate the same."""
    import numpy as np
    import torch
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.from_numpy(
                p.cpu().numpy().astype(np.float16).astype(np.float32)))


def holdout_bits(model, tok, ctx, path, device, limit=4000):
    """Per-URL (url, chars, model bits): ../experiment.py holdout_bits
    with batched GPU evaluation. Keys are the raw file lines (scheme
    included) to match classic-sizes.mjs output."""
    import math
    import torch
    import torch.nn.functional as F

    model.eval()
    raws = [u.strip() for u in open(path) if u.strip()][:limit]
    results = []
    LOG2E = 1 / math.log(2)
    with torch.no_grad():
        batch, metas = [], []
        def flush():
            if not batch: return
            maxlen = max(len(s) for s in batch)
            x = torch.zeros(len(batch), maxlen, dtype=torch.int64, device=device)
            for r, s in enumerate(batch):
                x[r, :len(s)] = torch.tensor(s, device=device)
            logp = F.log_softmax(model(x), dim=-1).cpu()
            for r, s in enumerate(batch):
                bits = 0.0
                for t in range(len(s) - 1):
                    bits -= logp[r, t, s[t + 1]].item() * LOG2E
                results.append((metas[r], len(strip_scheme(metas[r])), bits))
            batch.clear(); metas.clear()
        for raw in raws:
            try:
                ids = tok.encode(strip_scheme(raw))
            except ValueError:
                continue
            seq = [tok.eos] + ids + [tok.eos]
            if len(seq) > ctx: continue
            batch.append(seq); metas.append(raw)
            if len(batch) == 256: flush()
        flush()
    model.train()
    return results


LOG2_85 = 6.409390936137702  # log2(85)
OVERHEAD_BITS = 6  # sentinel + https flag + version marker + coder flush


def estimate_symbols(bits):
    import math
    return math.floor((bits + OVERHEAD_BITS) / LOG2_85) + 1


def evaluate_and_record(model, tok, cfg, n_params, train_time, curve, device):
    """f16-round, eval the shared holdout subset, persist artifacts."""
    import torch

    round_to_f16(model)
    rows = holdout_bits(model, tok, cfg["ctx"],
                        os.path.join(VOL, "holdout-eval.txt"), device)
    total_bits = sum(b for _, _, b in rows)
    total_chars = sum(c for _, c, _ in rows)
    est = {u: estimate_symbols(b) for u, _, b in rows}
    symbols = list(est.values())
    result = {
        "name": cfg["name"],
        "config": cfg,
        "params": n_params,
        "model_mb_f16": round(n_params * 2 / 1e6, 2),
        "train_minutes": round(train_time / 60, 1),
        "holdout_urls": len(rows),
        "bits_per_char": round(total_bits / total_chars, 4),
        "avg_est_symbols": round(sum(symbols) / len(symbols), 2),
        "val_curve": curve,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "val_curve"}))
    with open(os.path.join(VOL, f"est-{cfg['name']}.json"), "w") as f:
        json.dump(est, f)
    with open(os.path.join(VOL, "results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    if n_params:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(VOL, f"ckpt-{cfg['name']}.pt"))
    vol.commit()
    return result


def _run_training(cfg):
    """The shared training loop. Extras over the base recipe, all
    opt-in via cfg and recorded with the result row:
      amp: bfloat16 autocast (teacher only; students train fp32/TF32
           because they must match the shipped recipe)
      ckpt_every: save a resumable checkpoint (model+opt+step+curve)
           to the volume every N steps, and resume from it on restart
           - lets big runs survive preemption (Modal retries)
      distill_from: teacher config dict; adds KL(student, teacher
           logits) to the loss (see cfg kd_alpha / kd_temp)
    """
    import contextlib
    import math
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    amp = cfg.get("amp", False)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp \
        else contextlib.nullcontext

    tok = make_tokenizer(cfg["tokenizer"])
    vsize = cfg["tokenizer"].split("-")[1].split(".")[0]
    packed = np.load(os.path.join(VOL, f"packed-{vsize}.npy"))
    data = torch.from_numpy(packed.astype(np.int16)).to(device)
    del packed
    val = np.load(os.path.join(VOL, f"packed-val-{vsize}.npy"))[:4096]
    val = torch.from_numpy(val.astype(np.int16)).to(device)

    teacher = None
    if cfg.get("distill_from"):
        tcfg = cfg["distill_from"]
        teacher = build_model(tcfg, tok.size).to(device)
        ck = torch.load(os.path.join(VOL, f"ckpt-live-{tcfg['name']}.pt"),
                        map_location=device)
        teacher.load_state_dict(ck["model"])
        teacher.eval()
        teacher.requires_grad_(False)
        print(f"teacher {tcfg['name']} loaded (step {ck['step']})", flush=True)

    model = build_model(cfg, tok.size).to(device)
    if cfg.get("init_from"):
        # Warm-start (e.g. QAT fine-tuning from a trained checkpoint).
        # Prefer the live checkpoint's unrounded weights; fall back to
        # the f16-rounded eval checkpoint.
        live = os.path.join(VOL, f"ckpt-live-{cfg['init_from']}.pt")
        if os.path.exists(live):
            model.load_state_dict(torch.load(live, map_location=device)["model"])
        else:
            model.load_state_dict(torch.load(
                os.path.join(VOL, f"ckpt-{cfg['init_from']}.pt"),
                map_location=device))
        print(f"initialized from {cfg['init_from']}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    ctx, batch = cfg["ctx"], cfg.get("batch", 512)
    lr_max = cfg.get("lr", 8e-3)
    total_steps = cfg["train_tokens"] // (batch * ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    warmup = cfg.get("warmup", min(500, total_steps // 20))
    val_every = cfg.get("val_every", 1000)
    ckpt_every = cfg.get("ckpt_every", 0)
    kd_alpha = cfg.get("kd_alpha", 0.5)
    kd_temp = cfg.get("kd_temp", 1.0)
    live_ckpt = os.path.join(VOL, f"ckpt-live-{cfg['name']}.pt")

    start_step, curve, elapsed = 0, [], 0.0
    if ckpt_every and os.path.exists(live_ckpt):
        ck = torch.load(live_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, curve, elapsed = ck["step"], ck["curve"], ck["elapsed"]
        print(f"resumed from step {start_step}", flush=True)

    print(f"[{cfg['name']}] vocab={tok.size} params={n_params/1e3:.0f}K "
          f"windows={len(data)} steps={total_steps} batch={batch}", flush=True)

    def val_loss():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad(), autocast():
            for i in range(0, len(val), 1024):
                b = val[i:i + 1024].long()
                x, y = b[:, :-1], b[:, 1:]
                loss = F.cross_entropy(model(x).reshape(-1, tok.size),
                                       y.reshape(-1), reduction="sum")
                tot += loss.item(); cnt += y.numel()
        model.train()
        return tot / cnt

    def save_live(step):
        tmp = live_ckpt + ".tmp"
        torch.save({"step": step,
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(),
                    "curve": curve,
                    "elapsed": elapsed + time.time() - t0}, tmp)
        os.replace(tmp, live_ckpt)
        vol.commit()

    t0 = time.time()
    steps_per_epoch = len(data) // batch
    perm, perm_epoch = None, -1
    for step in range(start_step, total_steps):
        lr = lr_max * (step + 1) / warmup if step < warmup else \
            lr_max * 0.1 + 0.45 * lr_max * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g in opt.param_groups: g["lr"] = lr
        epoch, pos = divmod(step, steps_per_epoch)
        if epoch != perm_epoch:
            perm = torch.from_numpy(
                np.random.RandomState(seed + 1 + epoch)
                .permutation(len(data))).to(device)
            perm_epoch = epoch
        idx = perm[pos * batch:(pos + 1) * batch]
        b = data[idx].long()
        x, y = b[:, :-1], b[:, 1:]
        with autocast():
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tok.size), y.reshape(-1))
            if teacher is not None:
                with torch.no_grad(), \
                        torch.autocast("cuda", dtype=torch.bfloat16):
                    tlogits = teacher(x).float()
                kl = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.log_softmax(tlogits / kd_temp, dim=-1),
                    log_target=True, reduction="batchmean") \
                    * (kd_temp ** 2) / logits.shape[1]
                loss = (1 - kd_alpha) * loss + kd_alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % val_every == 0 or step + 1 == total_steps:
            vl = val_loss()
            el = elapsed + time.time() - t0
            curve.append([step + 1, round(vl, 5)])
            print(f"  step {step+1}/{total_steps} train {loss.item():.4f} "
                  f"val {vl:.4f} ({(step+1-start_step)*batch*ctx/(time.time()-t0)/1e3:.0f}K tok/s)",
                  flush=True)
        if ckpt_every and ((step + 1) % ckpt_every == 0
                           or step + 1 == total_steps):
            save_live(step + 1)
    return evaluate_and_record(model, tok, cfg, n_params,
                               elapsed + time.time() - t0, curve, device)


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=8 * 3600,
              cpu=8, memory=32768)
def train_gpu(cfg: dict):
    return _run_training(cfg)


@app.function(image=image, volumes={VOL: vol}, gpu="A100-40GB",
              timeout=12 * 3600, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train_big(cfg: dict):
    """Teacher-scale and distillation runs: A100, bf16 autocast,
    resumable checkpoints (pass amp/ckpt_every in cfg)."""
    return _run_training(cfg)


@app.function(image=image, volumes={VOL: vol}, gpu="A10G", timeout=1800,
              cpu=8, memory=16384)
def eval_shipped(bin_name: str = "url-model.bin"):
    """Re-evaluate the shipped CPU-trained model on the new shared
    eval subset, so the campaign has an apples-to-apples baseline row."""
    import struct

    import numpy as np
    import torch

    with open(os.path.join(VOL, bin_name), "rb") as f:
        raw = f.read()
    (hlen,) = struct.unpack_from("<I", raw, 0)
    header = json.loads(raw[4:4 + hlen].decode())
    cfg = {"name": f"shipped-reeval", "tokenizer": "url-model.bin",
           "dim": header["dim"], "layers": header["layers"],
           "heads": header["heads"], "mlp": header["mlpDim"],
           "ctx": header["maxLen"], "link_version": header.get("linkVersion")}

    import sys
    sys.path.insert(0, "/root")
    from vocab import GreedyTokenizer
    tok = GreedyTokenizer(header["tokens"])
    assert tok.size == header["vocab"], (tok.size, header["vocab"])

    model = build_model(cfg, tok.size)
    sd = model.state_dict()
    offset = 4 + hlen
    name_map = {"embed": "embed.weight", "pos": "pos.weight",
                "norm": "norm.g"}
    for t in header["tensors"]:
        n = int(np.prod(t["shape"]))
        dtype = t.get("dtype", "f16")
        if dtype in ("int8", "int4"):
            # v3: per-row f16 scales, then row-major int values
            rows = t["shape"][0]
            cols = n // rows
            scales = np.frombuffer(raw, dtype=np.float16, count=rows,
                                   offset=offset).astype(np.float32)
            offset += rows * 2
            if dtype == "int8":
                q = np.frombuffer(raw, dtype=np.int8, count=n,
                                  offset=offset).astype(np.float32)
                offset += n
            else:
                packed = np.frombuffer(raw, dtype=np.uint8, count=n // 2,
                                       offset=offset)
                offset += n // 2
                q = np.empty(n, dtype=np.float32)
                q[0::2] = (packed & 0x0f).astype(np.float32) - 8
                q[1::2] = (packed >> 4).astype(np.float32) - 8
            arr = q.reshape(rows, cols) * scales[:, None]
            arr = arr.reshape(-1)
        else:
            arr = np.frombuffer(raw, dtype=np.float16, count=n,
                                offset=offset).astype(np.float32)
            offset += n * 2
        key = t["name"]
        if key in name_map:
            sd_key = name_map[key]
        else:
            b, part = key.split(".")
            i = b[1:]
            sd_key = f"blocks.{i}." + {
                "norm1": "norm1.g", "qkv": "qkv.weight",
                "proj": "proj.weight", "norm2": "norm2.g",
                "up": "up.weight", "down": "down.weight"}[part]
        sd[sd_key] = torch.from_numpy(arr.reshape(t["shape"]).copy())
    model.load_state_dict(sd)
    model = model.to("cuda")
    return evaluate_and_record(model, tok, cfg, 0, 0.0, [], "cuda")


@app.function(image=image, volumes={VOL: vol}, timeout=1800, cpu=8,
              memory=16384)
def export_model(cfg: dict, out_name: str):
    """Export a trained checkpoint. Plain models export in
    hamr-url-model-v2 format (identical to ../train-final.py export()).
    With cfg["quant"] = "int8" | "int4" the block matmul weights are
    quantized per output channel (f16 scale per row, computed and
    applied exactly as the neural.js v3 loader dequantizes) and the
    format becomes hamr-url-model-v3."""
    import struct

    import numpy as np
    import torch

    quant = cfg.get("quant")
    tok = make_tokenizer(cfg["tokenizer"])
    model = build_model(cfg, tok.size)
    model.load_state_dict(torch.load(
        os.path.join(VOL, f"ckpt-{cfg['name']}.pt"), map_location="cpu"))
    round_to_f16(model)

    tensors, blobs = [], []
    def add(name, t):
        a = t.detach().numpy().astype(np.float16)
        tensors.append({"name": name, "shape": list(a.shape)})
        blobs.append(a.tobytes())

    def add_q(name, t):
        if not quant:
            return add(name, t)
        w = t.detach().numpy().astype(np.float32)
        qmax = {"int8": 127, "int4": 7}[quant]
        qmin = {"int8": -127, "int4": -8}[quant]
        # f16 scales, so quantization here and dequantization in the
        # JS loader use the exact same numbers
        scales = (np.abs(w).max(axis=1) / qmax).clip(min=1e-6) \
            .astype(np.float16)
        q = np.round(w / scales.astype(np.float32)[:, None]) \
            .clip(qmin, qmax).astype(np.int8)
        if quant == "int4":
            u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(-1)
            packed = (u[0::2] | (u[1::2] << 4)).astype(np.uint8)
            payload = packed.tobytes()
        else:
            payload = q.tobytes()
        tensors.append({"name": name, "shape": list(w.shape),
                        "dtype": quant})
        blobs.append(scales.tobytes() + payload)

    add("embed", model.embed.weight)
    add("pos", model.pos.weight)
    for i, b in enumerate(model.blocks):
        add(f"b{i}.norm1", b.norm1.g)
        add_q(f"b{i}.qkv", b.qkv.weight)
        add_q(f"b{i}.proj", b.proj.weight)
        add(f"b{i}.norm2", b.norm2.g)
        add_q(f"b{i}.up", b.up.weight)
        add_q(f"b{i}.down", b.down.weight)
    add("norm", model.norm.g)
    header = {
        "format": "hamr-url-model-v3" if quant else "hamr-url-model-v2",
        "vocab": tok.size, "dim": cfg["dim"], "layers": cfg["layers"],
        "heads": cfg["heads"], "mlpDim": cfg["mlp"], "maxLen": cfg["ctx"],
        "linkVersion": cfg["link_version"],
        "tensors": tensors
    }
    if tok.max_len > 1:
        header["tokens"] = tok.vocab
    hb = json.dumps(header).encode()
    path = os.path.join(VOL, out_name)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    vol.commit()
    print(f"exported {path}: {os.path.getsize(path)/1e6:.2f}MB")
    return os.path.getsize(path)


@app.local_entrypoint()
def main(action: str = "train", config: str = ""):
    if action == "train":
        cfg = json.loads(config)
        train_gpu.remote(cfg)
    elif action == "train-big":
        # teacher-scale or distillation run on A100 (amp, ckpt/resume)
        train_big.remote(json.loads(config))
    elif action == "train-many":
        for r in train_gpu.map(json.loads(config)):
            print("done:", r["name"], r["bits_per_char"])
    elif action == "eval-shipped":
        eval_shipped.remote()
    elif action == "export":
        cfg = json.loads(config)
        export_model.remote(cfg, cfg.get("out", "url-model-next.bin"))
    else:
        raise SystemExit(f"unknown action {action}")
