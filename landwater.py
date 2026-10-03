#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["llama-cpp-python==0.3.35", "numpy>=1.26", "jinja2>=3.1"]
# ///
"""Ask an open model "Land or Water?" for every point of the grid, on your CPU, and score its map.

    uv run landwater.py --list
    uv run landwater.py --model qwen3-0.6b --quick          # 648 points, a few minutes on a laptop
    uv run landwater.py --model gemma-4-26b-a4b-it           # the full 16,200-point grid with the best prompt

The model file is downloaded from Hugging Face at the pinned commit and checked against its sha256.
"""
import argparse, json, os, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "eval")
MODELS = {m["id"]: m for m in json.load(open(os.path.join(EVAL, "models.json")))["models"]}
# what each model needs beyond the defaults (the exact settings of the published runs; see runs/*/*.run.json)
EXTRA = {"olmo-3-7b-base": ["--completion"],
         "gpt-oss-120b": ["--template-var", "reasoning_effort=low", "--template-date", "2026-10-03",
                          "--assistant-prefix", "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"],
         "gemma-4-26b-a4b-it": ["--template-date", "2026-10-03"]}


def fetch(m, cache):
    path = os.path.join(cache, m["file"])
    if os.path.exists(path) and os.path.getsize(path) == m["bytes"]:
        return path
    url = f"https://huggingface.co/{m['gguf_repo']}/resolve/{m['gguf_commit']}/{m['file']}"
    print(f"downloading {m['file']} ({m['bytes'] / 1e9:.1f} GB) at commit {m['gguf_commit'][:10]}")
    subprocess.run([sys.executable, os.path.join(EVAL, "pdl.py"), url, path, m["sha256"]], check=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a model id from --list")
    ap.add_argument("--format", default="ref", choices=["ref", "deg", "nsew", "signed"],
                    help="how the coordinate is written (default ref: \"41° N, 73° W\", the best one)")
    ap.add_argument("--quick", action="store_true", help="the 10-degree grid: 648 points instead of 16,200")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cache", default=os.path.join(ROOT, "models"), help="where model files go")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runnable = {k: m for k, m in MODELS.items() if m["file"].endswith(".gguf") and "-of-" not in m["file"]}
    if a.list or not a.model:
        for k, m in runnable.items():
            print(f"  {k:24s} {m['bytes'] / 1e9:6.1f} GB  {m['license']:14s} {m['modality'].split(';')[0]}")
        return
    m = runnable.get(a.model) or sys.exit(f"unknown model {a.model!r}: see --list")
    os.makedirs(a.cache, exist_ok=True)
    gguf = fetch(m, a.cache)
    name = f"{a.model}.{a.format}" + (".quick" if a.quick else "")
    out = os.path.join(ROOT, "runs", "yours")
    cmd = [sys.executable, os.path.join(EVAL, "run_eval.py"), "--model", gguf, "--name", name, "--out", out,
           "--format", a.format, "--grid", "10" if a.quick else "2", "--threads", str(a.threads)] + EXTRA.get(a.model, [])
    if m.get("template") and "--completion" not in cmd:
        cmd += ["--template", os.path.join(EVAL, m["template"].split(" ")[0])]
    subprocess.run(cmd, check=True, cwd=EVAL)
    sys.path.insert(0, EVAL)
    from score import load_truth, score
    import csv
    with open(os.path.join(out, name + ".csv")) as f:
        pts = [((int(r["lat"]), int(r["lon"])), float(r["p_land_vs_water"])) for r in csv.DictReader(f)]
    s = score(pts, load_truth())
    print(f"\n{m['label']}, asked as {a.format}, {s['points']:,} points:\n"
          f"  area-weighted {s['area_weighted']:.1%}  (always-Water {s['floor_area_weighted']:.1%})\n"
          f"  AUC {s['auc']:.3f}  (0.50 = a coin)\n  says Land at {s['says_land']:,} points; land at {s['is_land']:,}\n"
          f"  raw output: runs/yours/{name}.csv")


if __name__ == "__main__":
    main()
