"""
Coverage-aware hybrid comparison for the GPU campaign - the same
metric as ../eval-matched.py, computed locally from artifacts:
  - classic sizes: node ../classic-sizes.mjs holdout-eval.txt classic-eval.json
  - per-URL neural estimates: est-<name>.json (from modal_train.py)

Usage: python3 eval-hybrid.py <holdout-eval.txt> <classic-eval.json> <est-*.json...>
"""
import json
import sys

holdout, classic_path, *est_paths = sys.argv[1:]
classic = json.load(open(classic_path))["sizes"]
urls = [u.strip() for u in open(holdout) if u.strip()]
urls = [u for u in urls if u in classic]
classic_avg = sum(classic[u] for u in urls) / len(urls)

for path in est_paths:
    est = json.load(open(path))
    hybrid, covered, won = [], 0, 0
    for u in urls:
        c, n = classic[u], est.get(u)
        if n is not None:
            covered += 1
            if n < c: won += 1
            hybrid.append(min(c, n))
        else:
            hybrid.append(c)
    avg = sum(hybrid) / len(hybrid)
    print(json.dumps({
        "est": path.split("/")[-1],
        "coverage": round(covered / len(urls), 3),
        "neural_win_rate": round(won / max(1, covered), 3),
        "avg_hybrid_symbols": round(avg, 2),
        "vs_classic": f"{100 * (1 - avg / classic_avg):.1f}%",
    }))

print(f"\nclassic avg: {classic_avg:.2f} over {len(urls)} URLs")
