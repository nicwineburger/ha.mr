"""
Builds the experiment corpus with deterministic hash-based splits.
- train/val/holdout membership depends only on the URL string (sha1),
  so splits are stable no matter how or in what order data was fetched
- per-host cap keeps single sites from dominating
Outputs: corpus-train.txt, corpus-holdout.txt (2%), corpus-val.txt (1%)
"""
import gzip, json, os, random, hashlib
from collections import defaultdict

random.seed(1234)
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

urls = set()
for fn in sorted(os.listdir(os.path.join(here, "ccblocks"))):
    try:
        with gzip.open(os.path.join(here, "ccblocks", fn), "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                i = line.find("{")
                if i < 0: continue
                try: rec = json.loads(line[i:])
                except Exception: continue
                if rec.get("status") != "200": continue
                u = rec.get("url", "")
                if not (10 <= len(u) <= 180): continue
                if not u.startswith(("http://", "https://")): continue
                if any(ord(c) < 0x21 or ord(c) > 0x7e for c in u): continue
                if u.lower().endswith("/robots.txt"): continue
                urls.add(u)
    except Exception:
        pass

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

# popular-site URLs (upweighted later by the trainer)
with open(os.path.join(here, "url-dataset/out.txt")) as f:
    popular = [u.strip() for u in f if u.strip()]
popular = [u for u in popular if 10 <= len(u) <= 180
           and all(0x21 <= ord(c) <= 0x7e for c in u)
           and u.startswith(("http://", "https://"))]

def bucket(u):
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 100

train, val, hold = [], [], []
popular_set = set(popular)
for u in capped + popular:
    b = bucket(u)
    if b < 2: hold.append(u)
    elif b < 3: val.append(u)
    else: train.append((u, u in popular_set))

# Keep popular flag: trainer upweights popular-site URLs 3x
random.shuffle(train)
with open(os.path.join(here, "corpus-train.txt"), "w") as f:
    for u, pop in train:
        f.write(("P " if pop else ". ") + u + "\n")
random.shuffle(hold)
with open(os.path.join(here, "corpus-holdout.txt"), "w") as f:
    f.write("\n".join(hold))
with open(os.path.join(here, "corpus-val.txt"), "w") as f:
    f.write("\n".join(val))
print(f"train: {len(train)}, val: {len(val)}, holdout: {len(hold)}")
print(f"train chars: {sum(len(u) for u, _ in train)/1e6:.1f}M")
