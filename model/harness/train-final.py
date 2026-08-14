"""
Trains the shipped model (the winning config from the screening
campaign, see README.md) and exports it as ../url-model.bin in
hamr-url-model-v2 format: uint32 header length, JSON header with
dimensions + token manifest, then float16 tensors in manifest order.

Usage: python3 train-final.py [out.bin]
"""
import json
import os
import struct
import sys

import numpy as np
import torch

import experiment as ex

SHIPPED_CONFIG = {
    "name": "shipped",
    "tokenizer": "vocab-1024.txt",
    "dim": 192, "layers": 5, "heads": 6, "mlp": 576, "ctx": 96,
    "train_tokens": 75000000,
    "seed": 1234,
    # The payload version this model's links carry. BUMP THIS for any
    # retrained model, archive the previous url-model.bin as
    # url-model-v<old>.bin, and keep both deployed - that's what keeps
    # previously issued neural links decodable.
    "link_version": 1
}


def export(model, tok, cfg, path):
    tensors, blobs = [], []
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
    header = {
        "format": "hamr-url-model-v2",
        "vocab": tok.size, "dim": cfg["dim"], "layers": cfg["layers"],
        "heads": cfg["heads"], "mlpDim": cfg["mlp"], "maxLen": cfg["ctx"],
        "linkVersion": cfg.get("link_version", 1),
        "tensors": tensors
    }
    if tok.max_len > 1:
        header["tokens"] = tok.vocab
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    print(f"exported {path}: {os.path.getsize(path)/1e6:.2f}MB")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ex.HERE, "..", "url-model.bin")
    cfg = SHIPPED_CONFIG
    tok = ex.make_tokenizer(cfg["tokenizer"])
    model, n_params, train_time = ex.train(cfg, tok)
    ex.round_to_f16(model)
    rows = ex.holdout_bits(model, tok, cfg["ctx"],
                           os.path.join(ex.DATA, "corpus-holdout.txt"))
    bits = sum(b for _, _, b in rows)
    chars = sum(c for _, c, _ in rows)
    print(f"holdout: {bits/chars:.4f} bits/char over {len(rows)} URLs (f16)")
    export(model, tok, cfg, out)


if __name__ == "__main__":
    main()
