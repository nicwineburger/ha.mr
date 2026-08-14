"""
Extracts and curates URLs from the Common Crawl index blocks fetched
by fetch-data.sh. Produces data/cc-urls.txt.
"""
import gzip
import json
import os
import random
from collections import defaultdict

random.seed(1234)
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

urls = []
for fn in sorted(os.listdir(os.path.join(here, "ccblocks"))):
    try:
        with gzip.open(os.path.join(here, "ccblocks", fn), "rt",
                       encoding="utf-8", errors="ignore") as f:
            for line in f:
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
                if not (10 <= len(u) <= 180):
                    continue
                if not u.startswith(("http://", "https://")):
                    continue
                # Keep printable ASCII only - URL.href output is ASCII
                if any(ord(c) < 0x21 or ord(c) > 0x7e for c in u):
                    continue
                if u.lower().endswith("/robots.txt"):
                    continue
                urls.append(u)
    except Exception:
        pass

print(f"raw urls: {len(urls)}")

# Cap per-host so single sites don't dominate the distribution
byhost = defaultdict(list)
for u in urls:
    byhost[u.split("/", 3)[2].lower()].append(u)
print(f"distinct hosts: {len(byhost)}")

capped = []
for host, us in byhost.items():
    random.shuffle(us)
    capped.extend(us[:30])
random.shuffle(capped)
print(f"after per-host cap: {len(capped)}")

with open(os.path.join(here, "cc-urls.txt"), "w") as f:
    f.write("\n".join(capped))
