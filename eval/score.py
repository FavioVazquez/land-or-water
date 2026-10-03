#!/usr/bin/env python3
"""Score every run against the answer key and check every number the write-up quotes.

    python eval/score.py           # the results table (also written to RESULTS.md)
    python eval/score.py --check   # recompute each quoted number from the CSVs; exit 1 if one is off

Standard library only. P(Land) for a point is P("Land") / (P("Land") + P("Water")) at the first answer token.
"""
import csv, hashlib, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK = os.path.join(ROOT, "data", "landmask_ne10m_land_v5.1.1.csv")

#   id, label, csv under runs/, how the coordinate was written, stage
RUNS = [dict(id=i, label=l, file=f, format=fm, stage=st) for i, l, f, fm, st in [
    ("gemma-4-26b-a4b.ref",          "Gemma 4 26B-A4B",       "full/gemma-4-26b-a4b.ref.csv",            "ref",    "full"),
    ("olmo-3-7b-instruct.signed",    "Olmo 3 7B Instruct",    "full/olmo-3-7b-instruct-signed.csv",      "signed", "full"),
    ("qwen3-0.6b.signed",            "Qwen3 0.6B",            "full/qwen3-0.6b-signed.csv",              "signed", "full"),
    ("gemma-4-26b-a4b.ref.pilot",    "Gemma 4 26B-A4B",       "pilot/gemma-4-26b-a4b.ref.csv",           "ref",    "pilot"),
    ("qwen3.5-9b.ref",               "Qwen3.5 9B",            "pilot/qwen3.5-9b.ref.csv",                "ref",    "pilot"),
    ("gpt-oss-120b.ref",             "gpt-oss-120b",          "pilot/gpt-oss-120b.ref.csv",              "ref",    "pilot"),
    ("qwen2.5-7b-instruct.ref",      "Qwen2.5 7B Instruct",   "pilot/qwen2.5-7b-instruct.ref.csv",       "ref",    "pilot"),
    ("olmo-3.1-32b-instruct.deg",    "Olmo 3.1 32B Instruct", "pilot/olmo-3.1-32b-instruct.deg.csv",     "deg",    "pilot"),
    ("qwen3-32b.deg",                "Qwen3 32B",             "pilot/qwen3-32b.deg.csv",                 "deg",    "pilot"),
    ("qwen3-14b.deg",                "Qwen3 14B",             "pilot/qwen3-14b.deg.csv",                 "deg",    "pilot"),
    ("qwen3-4b.deg",                 "Qwen3 4B",              "pilot/qwen3-4b.deg.csv",                  "deg",    "pilot"),
    ("qwen3-4b.nsew",                "Qwen3 4B",              "pilot/qwen3-4b.nsew.csv",                 "nsew",   "pilot"),
    ("qwen3-4b.signed",              "Qwen3 4B",              "pilot/qwen3-4b.signed.csv",               "signed", "pilot"),
    ("olmo-3-7b-instruct.deg",       "Olmo 3 7B Instruct",    "pilot/olmo-3-7b-instruct.deg.csv",        "deg",    "pilot"),
    ("olmo-3-7b-instruct.nsew",      "Olmo 3 7B Instruct",    "pilot/olmo-3-7b-instruct.nsew.csv",       "nsew",   "pilot"),
    ("olmo-3-7b-base.deg",           "Olmo 3 7B base",        "pilot/olmo-3-7b-base.deg.csv",            "deg",    "pilot"),
    ("olmo-3-7b-base.nsew",          "Olmo 3 7B base",        "pilot/olmo-3-7b-base.nsew.csv",           "nsew",   "pilot"),
    ("olmo-3-7b-base.signed",        "Olmo 3 7B base",        "pilot/olmo-3-7b-base.signed.csv",         "signed", "pilot"),
    ("gemma-4-26b-a4b.ref.nobos",    "Gemma 4 26B-A4B (BOS bug)", "excluded/gemma-4-26b-a4b.ref.nobos.csv", "ref", "excluded"),
]]
FORMATS = {"signed": "Latitude: 41, Longitude: -73", "nsew": "Latitude: 41 N, Longitude: 73 W",
           "deg": "Coordinates: 41°N 73°W", "ref": "41° N, 73° W"}

# Every number the article and the README quote: (run id, metric, value as printed, decimals)
CLAIMS = [
    ("gemma-4-26b-a4b.ref", "area_weighted", 0.687, 3), ("gemma-4-26b-a4b.ref", "floor_area_weighted", 0.711, 3),
    ("gemma-4-26b-a4b.ref", "auc", 0.755, 3), ("gemma-4-26b-a4b.ref", "accuracy", 0.634, 3),
    ("gemma-4-26b-a4b.ref", "says_land", 8303, 0), ("gemma-4-26b-a4b.ref", "is_land", 5379, 0),
    ("gemma-4-26b-a4b.ref", "floor_accuracy", 0.668, 3),
    ("gemma-4-26b-a4b.ref.pilot", "auc", 0.76, 2), ("gemma-4-26b-a4b.ref.nobos", "auc", 0.67, 2),
    ("gpt-oss-120b.ref", "area_weighted", 0.757, 3), ("gpt-oss-120b.ref", "says_land_share", 0.057, 3),
    ("qwen3.5-9b.ref", "auc", 0.71, 2), ("olmo-3.1-32b-instruct.deg", "auc", 0.71, 2),
    ("qwen3-4b.deg", "auc", 0.61, 2), ("qwen3-14b.deg", "auc", 0.58, 2), ("qwen3-32b.deg", "auc", 0.60, 2),
    ("olmo-3-7b-instruct.signed", "area_weighted", 0.713, 3), ("qwen3-0.6b.signed", "says_water", 16199, 0),
]


def load_truth():
    with open(MASK) as f:
        return {(int(r["lat"]), int(r["lon"])): r["land"] == "1" for r in csv.DictReader(f)}


def load_run(r):
    path = os.path.join(ROOT, "runs", r["file"])
    with open(path) as f:
        pts = [((int(x["lat"]), int(x["lon"])), float(x["p_land_vs_water"])) for x in csv.DictReader(f)]
    meta = json.load(open(path[:-4] + ".run.json"))
    return pts, meta


def auc(ps, ys):
    """Chance that a random land point gets a higher P(Land) than a random water point (ties count half)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks, i = [0.0] * len(ps), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ps[order[j + 1]] == ps[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(ys)
    neg = len(ys) - pos
    return (sum(r for r, y in zip(ranks, ys) if y) - pos * (pos + 1) / 2) / (pos * neg)


def score(pts, truth):
    ys = [truth[k] for k, _ in pts]
    ps = [p for _, p in pts]
    said = [p > 0.5 for p in ps]
    w = [math.cos(math.radians(k[0])) for k, _ in pts]
    right = [s == y for s, y in zip(said, ys)]
    W = sum(w)
    return {"accuracy": sum(right) / len(pts), "floor_accuracy": sum(not y for y in ys) / len(pts),
            "area_weighted": sum(wi for wi, r in zip(w, right) if r) / W,
            "floor_area_weighted": sum(wi for wi, y in zip(w, ys) if not y) / W,
            "auc": auc(ps, ys), "says_land": sum(said), "says_water": len(said) - sum(said),
            "says_land_share": sum(said) / len(said), "is_land": sum(ys), "points": len(pts)}


def provenance(r):
    """What pins a run: the model file (Hugging Face commit + sha256; the tokenizer ships inside it), the chat template
    (sha256 recorded at run time, re-hashed here from eval/templates/), the answer token ids and the llama.cpp bindings."""
    meta = load_run(r)[1]
    models = {m["file"].split(" ")[0]: m for m in json.load(open(os.path.join(ROOT, "eval", "models.json")))["models"]}
    m = models[meta["model_file"]]
    tpl = meta.get("template_file")
    tpl_ok = None
    if tpl:
        with open(os.path.join(ROOT, "eval", "templates", os.path.basename(tpl)), "rb") as f:
            tpl_ok = hashlib.sha256(f.read()).hexdigest() == meta["template_sha256"]
    ids = meta["answer_token_ids"]
    key = ("Land", "Water") if tpl else (" Land", " Water")
    return {"file": meta["model_file"], "commit": m["gguf_commit"], "sha256": m["sha256"].split(" ")[0],
            "template": os.path.basename(tpl) if tpl else "none (base model, plain completion)",
            "template_sha256": meta.get("template_sha256") or "", "template_ok": tpl_ok,
            "token_ids": f"{ids[key[0]]} / {ids[key[1]]}", "llama_cpp": meta["llama_cpp_python"]}


def main():
    truth = load_truth()
    scores = {r["id"]: score(load_run(r)[0], truth) for r in RUNS}
    if "--check" in sys.argv:
        bad = 0
        for rid, metric, want, nd in CLAIMS:
            got = scores[rid][metric]
            ok = round(got, nd) == round(want, nd) if nd else got == want
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'}  {rid:28s} {metric:20s} quoted {want}  computed {got:.4f}")
        print(f"\n{len(CLAIMS) - bad} of {len(CLAIMS)} quoted numbers match the data")
        for r in RUNS:
            pv = provenance(r)
            if pv["template_ok"] is False:
                bad += 1
                print(f"BAD  {r['id']}: eval/templates/{pv['template']} does not hash to the template the run used")
        print("every chat template in eval/templates/ hashes to the one its runs recorded" if not bad else "")
        sys.exit(1 if bad else 0)
    lines = ["| model | asked as | points | area-weighted (always-Water) | accuracy (always-Water) | AUC | says Land |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in RUNS:
        s = scores[r["id"]]
        lines.append(f"| {r['label']} | `{FORMATS[r['format']]}` | {s['points']:,} | {s['area_weighted']:.1%} ({s['floor_area_weighted']:.1%}) "
                     f"| {s['accuracy']:.1%} ({s['floor_accuracy']:.1%}) | {s['auc']:.3f} | {s['says_land_share']:.1%} |")
    table = "\n".join(lines)
    prov = ["| run | model file (Hugging Face commit, sha256) | chat template (sha256) | token ids Land / Water | llama-cpp-python |",
            "|---|---|---|---|---|"]
    for r in RUNS:
        pv = provenance(r)
        prov.append(f"| `{r['id']}` | `{pv['file']}` ({pv['commit'][:10]}, `{pv['sha256'][:16]}`) "
                    + (f"| `{pv['template']}` (`{pv['template_sha256'][:16]}`) " if pv["template_sha256"] else f"| {pv['template']} ")
                    + f"| {pv['token_ids']} | {pv['llama_cpp']} |")
    print(table)
    with open(os.path.join(ROOT, "RESULTS.md"), "w") as f:
        f.write("# Results\n\nGenerated by `python eval/score.py` from the CSVs in `runs/`. In brackets: the score of answering "
                "\"Water\" everywhere on the same points. AUC: 0.50 is a coin, 1.00 a perfect map.\n\n" + table + "\n\n"
                "## What pins each run\n\nThe tokenizer ships inside the GGUF, so the model file's Hugging Face commit and sha256 pin "
                "it. The chat template's sha256 was recorded when the run started; `python eval/score.py --check` re-hashes the copy "
                "in `eval/templates/` against it. Full hashes are in `eval/models.json` and each `runs/*/*.run.json`.\n\n"
                + "\n".join(prov) + "\n")


if __name__ == "__main__":
    main()
