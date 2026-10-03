#!/usr/bin/env python3
"""Pack every run into docs/data.json for the map viewer: P(Land) per point (0-255), the answer key and the scores."""
import csv, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import load_truth, load_run, score, RUNS, ROOT, FORMATS

truth = load_truth()
out = {"truth": "".join("1" if truth[k] else "0" for k in sorted(truth, key=lambda k: (-k[0], k[1]))), "runs": []}
for r in RUNS:
    pts, meta = load_run(r)
    s = score(pts, truth)
    grid = 2 if len(pts) == 16200 else 10
    out["runs"].append({"id": r["id"], "label": r["label"], "format": r["format"], "stage": r["stage"], "grid": grid,
                        "asked": FORMATS[r["format"]], "minutes": round(meta.get("seconds_run", 0) / 60, 1),
                        "p": "".join(f"{min(255, round(p * 255)):02x}" for _, p in pts), **{k: round(v, 4) for k, v in s.items()}})
with open(os.path.join(ROOT, "docs", "data.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("docs/data.json:", len(out["runs"]), "runs")
