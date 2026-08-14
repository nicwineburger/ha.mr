"""
Trains the tiny character-level URL model used for neural link compression.

Design constraints (so the browser can reproduce inference bit-exactly
in plain JavaScript):
- ReLU MLP (no GELU/SiLU - those need exp/tanh)
- RMSNorm (only +,*,/,sqrt - all IEEE correctly-rounded in JS)
- Learned absolute position embeddings (no RoPE - sin/cos are not
  correctly-rounded in JS)
- Softmax is fine: the JS side uses a deterministic exp approximation
"""
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)
random.seed(1234)
torch.set_num_threads(4)

# --- Vocabulary: printable ASCII 0x21..0x7E -> ids 1..94, EOS = 0 ---
EOS = 0
VOCAB = 95
MAX_LEN = 128

DIM = 128
LAYERS = 4
HEADS = 4
MLP_DIM = 384

def tokenize(s):
    return [ord(c) - 0x20 for c in s]

def strip_scheme(u):
    if u.startswith("https://"): return u[8:]
    if u.startswith("http://"): return u[7:]
    return u

# --- Load corpus ---
here = os.path.dirname(os.path.abspath(__file__))
urls = []
with open(os.path.join(here, "data", "cc-urls.txt")) as f:
    urls += [strip_scheme(u.strip()) for u in f if u.strip()]
popular = []
with open(os.path.join(here, "data", "url-dataset", "out.txt")) as f:
    popular += [strip_scheme(u.strip()) for u in f if u.strip()]
# Keep only in-vocab URLs
def ok(u):
    return 0 < len(u) <= 170 and all(0x21 <= ord(c) <= 0x7e for c in u)
urls = [u for u in urls if ok(u)]
popular = [u for u in popular if ok(u)]
random.shuffle(urls)
random.shuffle(popular)

# Hold out benchmark/validation sets before any training exposure
holdout = urls[:1500] + popular[:500]
urls = urls[1500:]
popular = popular[500:]
with open(os.path.join(here, "data", "holdout-urls.txt"), "w") as f:
    f.write("\n".join(holdout))

# Popular sites are upweighted: they're what people actually shorten
train_urls = urls + popular * 3
random.shuffle(train_urls)
val_urls = holdout[:1000]

print(f"train urls: {len(train_urls)}, holdout: {len(holdout)}")

# --- Pack into training windows aligned to URL starts ---
# Each window: EOS url EOS url ... (truncated at MAX_LEN+1 for shift)
def pack(url_list):
    rows = []
    buf = []
    for u in url_list:
        if not buf:
            buf = [EOS]
        buf += tokenize(u) + [EOS]
        while len(buf) >= MAX_LEN + 1:
            rows.append(buf[:MAX_LEN + 1])
            buf = []
            break  # windows always start at a URL boundary
    return np.array(rows, dtype=np.int64)

data = pack(train_urls)
val_data = pack(val_urls)
print(f"train windows: {len(data)}, val windows: {len(val_data)}")

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
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.norm2 = RMSNorm(DIM)
        self.up = nn.Linear(DIM, MLP_DIM, bias=False)
        self.down = nn.Linear(MLP_DIM, DIM, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(DIM, dim=2)
        q = q.view(B, T, HEADS, C // HEADS).transpose(1, 2)
        k = k.view(B, T, HEADS, C // HEADS).transpose(1, 2)
        v = v.view(B, T, HEADS, C // HEADS).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.down(F.relu(self.up(self.norm2(x))))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(MAX_LEN, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = RMSNorm(DIM)
        # Output head is tied to the embedding
    def forward(self, ids):
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T)).unsqueeze(0)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x @ self.embed.weight.T

model = Model()
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e3:.0f}K (~{n_params*2/1e6:.2f}MB f16)")

BATCH = 48
EPOCHS = float(os.environ.get("EPOCHS", "2"))
steps_per_epoch = len(data) // BATCH
total_steps = int(steps_per_epoch * EPOCHS)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
warmup = min(200, total_steps // 20)

def lr_at(step):
    if step < warmup:
        return 3e-3 * (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    return 3e-4 + 0.5 * (3e-3 - 3e-4) * (1 + math.cos(math.pi * p))

def evaluate():
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(val_data), 96 * 40), 96):
            b = torch.from_numpy(val_data[i:i+96])
            x, y = b[:, :-1], b[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            tot += loss.item() * x.numel()
            n += x.numel()
    model.train()
    return tot / n

print(f"training {total_steps} steps ({EPOCHS} epochs)")
t0 = time.time()
step = 0
perm = np.random.permutation(len(data))
while step < total_steps:
    i = (step * BATCH) % (len(data) - BATCH)
    if i + BATCH > len(data):
        perm = np.random.permutation(len(data))
    idx = perm[i:i+BATCH]
    b = torch.from_numpy(data[idx])
    x, y = b[:, :-1], b[:, 1:]
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 200 == 0 or step == total_steps:
        el = time.time() - t0
        toks = step * BATCH * MAX_LEN
        print(f"step {step}/{total_steps} loss {loss.item():.4f} "
              f"({toks/el/1e3:.0f}K tok/s, {el/60:.1f} min)", flush=True)

val_loss = evaluate()
print(f"val loss: {val_loss:.4f} nats/char = {val_loss/math.log(2):.3f} bits/char")

# --- Export: JSON header + little-endian f16 tensor data ---
# The JS side computes with these exact f16-rounded values, so round
# them here and report the f16 validation loss too.
def export(path):
    tensors = []
    blobs = []
    def add(name, t):
        a = t.detach().numpy().astype(np.float16)
        tensors.append({"name": name, "shape": list(a.shape)})
        blobs.append(a.tobytes())
    add("embed", model.embed.weight)
    add("pos", model.pos.weight)
    for i, b in enumerate(model.blocks):
        add(f"b{i}.norm1", b.norm1.g)
        add(f"b{i}.qkv", b.qkv.weight)
        add(f"b{i}.proj", b.proj.weight)
        add(f"b{i}.norm2", b.norm2.g)
        add(f"b{i}.up", b.up.weight)
        add(f"b{i}.down", b.down.weight)
    add("norm", model.norm.g)
    header = json.dumps({
        "format": "hamr-url-model-v1",
        "vocab": VOCAB, "dim": DIM, "layers": LAYERS, "heads": HEADS,
        "mlpDim": MLP_DIM, "maxLen": MAX_LEN,
        "tensors": tensors
    }).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    print(f"exported {path}: {os.path.getsize(path)/1e6:.2f}MB")

# f16 round-trip the weights in-place, then re-evaluate
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.from_numpy(p.numpy().astype(np.float16).astype(np.float32)))
val_f16 = evaluate()
print(f"val loss after f16 rounding: {val_f16/math.log(2):.3f} bits/char")

export(os.path.join(here, "url-model.bin"))
torch.save(model.state_dict(), os.path.join(here, "url-model.pt"))
