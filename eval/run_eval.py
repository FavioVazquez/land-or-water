#!/usr/bin/env python3
"""Land or Water? eval.

For every point of a 2-degree grid (180 x 90 = 16,200 points) ask the model, in text, whether the
coordinate is on land or water, and read the full next-token distribution at the first answer
position (no sampling: the greedy answer is the argmax). We store P("Land"), P("Water") and the
argmax token for every point.

Every point is evaluated on its own: empty KV cache, the full prompt, one llama_decode call. No
prefix caching and no batching across points: on this CPU build those change the Q8_0 numerics
enough to move P(Land) by a few hundredths, so the canonical single-prompt path is used for all.
"""
import argparse
import csv
import hashlib
import ctypes
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import llama_cpp as L

# ---- The prompt: written once, used unchanged for every model -------------------------------
# One user message and no system message of our own. It is rendered with each model's official chat
# template (templates/*.jinja, copied from the Qwen/Qwen3-* and allenai/Olmo-3-7B-Instruct repos),
# exactly as transformers' apply_chat_template would (tools=None, add_generation_prompt=True,
# enable_thinking=False). For Qwen3 that closes the thinking block before the answer; Olmo 3 Instruct
# has no thinking block and its template adds its own default system line. Either way the next token
# after the rendered prompt is the answer itself.
# Question formats. "signed" is the original prompt (the first full runs). After those runs read the minus
# sign rather than the place, a 10-degree pilot compared it with hemisphere-letter formats (see results/pilot/).
# Everything except the coordinate description and the coordinate line is the same in every format.
FORMATS = {
    "signed": ("Land or Water? The coordinate below is in decimal degrees "
               "(positive latitude is north, positive longitude is east). "
               "Answer with exactly one word: Land or Water.\n"
               "Latitude: {lat}, Longitude: {lon}"),
    "nsew": ("Land or Water? The coordinate below is in degrees with hemisphere letters "
             "(N or S for latitude, E or W for longitude). "
             "Answer with exactly one word: Land or Water.\n"
             "Latitude: {alat} {ns}, Longitude: {alon} {ew}"),
    "deg": ("Land or Water? The coordinate below is in degrees with hemisphere letters "
            "(N or S for latitude, E or W for longitude). "
            "Answer with exactly one word: Land or Water.\n"
            "Coordinates: {alat}\u00b0{ns} {alon}\u00b0{ew}"),
    # the prompt of the reference write-up (outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth, Aug 2025):
    # instructions first, then only the coordinate, "x\u00b0 S, y\u00b0 W"
    "ref": ("If this location is over land, say 'Land'. If this location is over water, say 'Water'. "
            "Do not say anything else. {alat}\u00b0 {ns}, {alon}\u00b0 {ew}"),
}
USER = FORMATS["signed"]
COMPLETION_SUFFIX = "\nAnswer:"  # base models (no chat template): the answer is the next token, " Land" or " Water"
COMPLETION = False                # set in main() from --completion
TEMPLATE = None  # set in main() from --template
SPECIAL = {"bos_token": "", "eos_token": ""}  # filled from the model's vocabulary in main(), as transformers passes them
TEMPLATE_VARS = {}       # extra template variables from --template-var (e.g. reasoning_effort=low for gpt-oss)
ASSISTANT_PREFIX = ""    # text after the generation prompt (gpt-oss: "<|channel|>final<|message|>" skips the analysis channel)
TEMPLATE_DATE = time.strftime("%Y-%m-%d", time.gmtime())   # what strftime_now() returns inside templates (gpt-oss prints it)


def compile_template(source):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise RuntimeError(msg)

    def strftime_now(fmt):
        # transformers gives templates the current date; a fixed date (--template-date, default the run's start date)
        # keeps every prompt of a run identical and the run reproducible
        return time.strftime(fmt, time.strptime(TEMPLATE_DATE, "%Y-%m-%d"))

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(source)


def user_text(lat, lon):
    return USER.format(lat=lat, lon=lon, alat=abs(lat), alon=abs(lon),
                       ns="N" if lat > 0 else "S", ew="E" if lon > 0 else "W")


def render(template, user_text):
    # transformers' apply_chat_template passes the tokenizer's special tokens to the template; Gemma's template starts
    # with {{ bos_token }}. Before 2026-10-03 14:40 UTC they were not passed (only the Gemma 4 run was affected:
    # the Qwen and Olmo templates do not use bos_token, and eos_token only appears after assistant turns).
    return template.render(messages=[{"role": "user", "content": user_text}], tools=None,
                           add_generation_prompt=True, enable_thinking=False, **SPECIAL, **TEMPLATE_VARS) + ASSISTANT_PREFIX


# ---- The grid: cell centres, north to south, west to east -------------------------------------
# 2-degree grid (16,200 points): lat 89..-89, lon -179..179. 10-degree pilot grid (648 points): lat 85..-85,
# lon -175..175; every pilot point is also a point of the 2-degree grid. No point lies on 0 (no sign/letter ambiguity).
def grid(step):
    return ([90 - step // 2 - step * i for i in range(180 // step)],
            [-180 + step // 2 + step * j for j in range(360 // step)])


LATS, LONS = grid(2)


def prompt_for(lat, lon):
    if COMPLETION:
        return user_text(lat, lon) + COMPLETION_SUFFIX
    return render(TEMPLATE, user_text(lat, lon))


def tokenize(vocab, text, special=True, add_special=False):
    b = text.encode("utf-8")
    buf = (L.llama_token * (len(b) + 16))()
    n = L.llama_tokenize(vocab, b, len(b), buf, len(buf), add_special, special)
    if n < 0:
        raise RuntimeError("tokenize failed")
    return list(buf[:n])


def piece(vocab, tok):
    buf = ctypes.create_string_buffer(256)
    n = L.llama_token_to_piece(vocab, tok, buf, 256, 0, True)
    return buf.raw[:n].decode("utf-8", errors="replace")


def meta(model, key):
    buf = ctypes.create_string_buffer(65536)
    n = L.llama_model_meta_val_str(model, key.encode(), buf, 65536)
    return buf.value.decode("utf-8", errors="replace") if n >= 0 else None


def log_softmax(lg):
    lg = lg.astype(np.float64)
    m = lg.max()
    return lg - (m + np.log(np.exp(lg - m).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .gguf (first shard for split files)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--template", help="official chat_template.jinja of the model (chat models)")
    ap.add_argument("--completion", action="store_true", help="base model: plain text + '\\nAnswer:', answer tokens ' Land'/' Water'")
    ap.add_argument("--format", default="signed", choices=sorted(FORMATS), help="how the coordinate is written")
    ap.add_argument("--grid", type=int, default=2, choices=[2, 10], help="grid step in degrees")
    ap.add_argument("--template-var", action="append", default=[], metavar="KEY=VALUE",
                    help="extra chat-template variable, e.g. reasoning_effort=low (repeatable)")
    ap.add_argument("--assistant-prefix", default="",
                    help="text appended after the generation prompt, e.g. '<|channel|>final<|message|>' for gpt-oss")
    ap.add_argument("--template-date", default=None, help="date returned by strftime_now() in templates (YYYY-MM-DD)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N points (smoke test)")
    ap.add_argument("--verify", type=int, default=40, help="re-run N random points; results must be identical")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global TEMPLATE, USER, COMPLETION, LATS, LONS, ASSISTANT_PREFIX, TEMPLATE_DATE
    USER = FORMATS[args.format]
    TEMPLATE_VARS.update(kv.split("=", 1) for kv in args.template_var)
    ASSISTANT_PREFIX = args.assistant_prefix
    TEMPLATE_DATE = args.template_date or TEMPLATE_DATE
    COMPLETION = args.completion
    LATS, LONS = grid(args.grid)
    if not COMPLETION:
        if not args.template:
            sys.exit("--template is required unless --completion")
        TEMPLATE = compile_template(open(args.template).read())

    L.llama_backend_init()
    mp = L.llama_model_default_params()
    mp.n_gpu_layers = 0
    t_load0 = time.time()
    model = L.llama_model_load_from_file(args.model.encode(), mp)
    if not model:
        sys.exit("model load failed")
    vocab = L.llama_model_get_vocab(model)
    n_vocab = L.llama_vocab_n_tokens(vocab)
    for key, tok in (("bos_token", L.llama_vocab_bos(vocab)), ("eos_token", L.llama_vocab_eos(vocab))):
        SPECIAL[key] = piece(vocab, tok) if tok >= 0 else ""

    points = [(i, j, lat, lon) for i, lat in enumerate(LATS) for j, lon in enumerate(LONS)]
    if args.limit:
        points = points[: args.limit]
    # chat models: the template already holds every special token (as apply_chat_template(tokenize=True) does);
    # base models: plain text, with the BOS token only if the model's tokenizer asks for one
    toks = [tokenize(vocab, prompt_for(lat, lon), add_special=COMPLETION) for (_, _, lat, lon) in points]
    embedded_same = None
    if not COMPLETION:
        embedded = compile_template(meta(model, "tokenizer.chat_template"))
        embedded_same = render(embedded, user_text(41, -73)) == prompt_for(41, -73)
    Lmax = max(len(t) for t in toks)

    sp = " " if COMPLETION else ""     # after "Answer:" the answer word starts with a space
    land_ids = tokenize(vocab, sp + "Land", special=False)
    water_ids = tokenize(vocab, sp + "Water", special=False)
    if len(land_ids) != 1 or len(water_ids) != 1:
        sys.exit(f"answer words are not single tokens: Land={land_ids} Water={water_ids}")
    LAND, WATER = land_ids[0], water_ids[0]
    variants = {"land": "p_land_lower", "water": "p_water_lower", " Land": "p_land_space", " Water": "p_water_space",
                "LAND": "p_land_upper", "WATER": "p_water_upper"}
    if COMPLETION:
        variants = {" land": "p_land_lower_space", " water": "p_water_lower_space", "Land": "p_land_nospace",
                    "Water": "p_water_nospace", " LAND": "p_land_upper_space", " WATER": "p_water_upper_space"}
    extra = {w: tokenize(vocab, w, special=False) for w in variants}
    extra = {w: t[0] for w, t in extra.items() if len(t) == 1}

    cp = L.llama_context_default_params()
    cp.n_ctx = 256
    cp.n_batch = 256
    cp.n_ubatch = 256
    cp.n_seq_max = 1
    cp.n_threads = args.threads
    cp.n_threads_batch = args.threads
    ctx = L.llama_init_from_model(model, cp)
    if not ctx:
        sys.exit("context init failed")
    mem = L.llama_get_memory(ctx)
    batch = L.llama_batch_init(256, 0, 1)
    t_load = time.time() - t_load0

    def answer(t):
        """Evaluate one full prompt alone (empty cache, one decode); return the next-token log-probs."""
        L.llama_memory_clear(mem, True)
        for k, tok in enumerate(t):
            batch.token[k] = tok
            batch.pos[k] = k
            batch.n_seq_id[k] = 1
            batch.seq_id[k][0] = 0
            batch.logits[k] = 1 if k == len(t) - 1 else 0
        batch.n_tokens = len(t)
        if L.llama_decode(ctx, batch) != 0:
            sys.exit("decode failed")
        return log_softmax(np.ctypeslib.as_array(L.llama_get_logits_ith(ctx, len(t) - 1), shape=(n_vocab,)))

    desc = ctypes.create_string_buffer(256)
    L.llama_model_desc(model, desc, 256)
    fields = ["row", "col", "lat", "lon", "p_land", "p_water", "p_land_vs_water", "mass_land_water",
              "logp_land", "logp_water", "argmax_token", "argmax_prob"] + [variants[w] for w in extra]
    csv_path = os.path.join(args.out, f"{args.name}.csv")
    f = open(csv_path, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(fields)
    results = {}
    n_tok = 0
    loads = [os.getloadavg()[0]]
    t0 = time.time()
    for idx, (i, j, lat, lon) in enumerate(points):
        lp = answer(toks[idx])
        n_tok += len(toks[idx])
        am = int(np.argmax(lp))
        pl, pw = float(np.exp(lp[LAND])), float(np.exp(lp[WATER]))
        row = [i, j, lat, lon, f"{pl:.6e}", f"{pw:.6e}",
               f"{1.0 / (1.0 + np.exp(lp[WATER] - lp[LAND])):.6f}", f"{pl + pw:.6f}",
               f"{lp[LAND]:.5f}", f"{lp[WATER]:.5f}", piece(vocab, am), f"{float(np.exp(lp[am])):.6f}"]
        row += [f"{float(np.exp(lp[t])):.6e}" for t in extra.values()]
        wr.writerow(row)
        results[idx] = (float(lp[LAND]), float(lp[WATER]))
        done = idx + 1
        if done % 900 == 0 or done == len(points):
            f.flush()
            loads.append(os.getloadavg()[0])
            el = time.time() - t0
            print(f"[{args.name}] {done}/{len(points)} points  {el:.0f}s  {n_tok / el:.0f} tok/s  "
                  f"eta {el / done * (len(points) - done):.0f}s", flush=True)
    f.close()
    t_run = time.time() - t0

    # repeat check: a random sample of points evaluated again must give bit-identical log-probs
    rng = np.random.default_rng(0)
    vidx = sorted(rng.choice(len(points), size=min(args.verify, len(points)), replace=False).tolist())
    diffs = []
    for idx in vidx:
        lp = answer(toks[idx])
        diffs.append(max(abs(lp[LAND] - results[idx][0]), abs(lp[WATER] - results[idx][1])))

    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None

    info = {
        "name": args.name,
        "model_file": os.path.basename(args.model),
        "model_desc": desc.value.decode(),
        "general.name": meta(model, "general.name"),
        "general.file_type": meta(model, "general.file_type"),
        "n_params": L.llama_model_n_params(model),
        "n_vocab": n_vocab,
        "chat_template": meta(model, "tokenizer.chat_template"),
        "prompt_example": prompt_for(41, -73),
        "user_template": USER, "format": args.format, "grid_step_deg": args.grid,
        "mode": "completion (base model, no chat template)" if COMPLETION else "chat template",
        "completion_suffix": COMPLETION_SUFFIX if COMPLETION else None,
        "answer_tokens": [sp + "Land", sp + "Water"],
        "first_prompt_tokens": [piece(vocab, t) for t in toks[0][:3]],
        "add_bos_token_meta": meta(model, "tokenizer.ggml.add_bos_token"),
        "system_message": None, "enable_thinking": None if COMPLETION else False,
        "template_special_tokens": dict(SPECIAL),
        "template_vars": dict(TEMPLATE_VARS), "assistant_prefix": ASSISTANT_PREFIX, "template_date": TEMPLATE_DATE,
        "template_file": os.path.basename(args.template) if args.template and not COMPLETION else None,
        "template_sha256": hashlib.sha256(open(args.template, "rb").read()).hexdigest() if args.template and not COMPLETION else None,
        "embedded_template_renders_same_prompt": embedded_same,
        "max_prompt_tokens": Lmax, "min_prompt_tokens": min(len(t) for t in toks),
        "method": "each point evaluated alone: empty KV cache, full prompt, one llama_decode, logits of the last prompt token, log-softmax over the whole vocabulary",
        "answer_token_ids": {sp + "Land": LAND, sp + "Water": WATER, **extra},
        "points": len(points), "lats": [LATS[0], LATS[-1], args.grid], "lons": [LONS[0], LONS[-1], args.grid],
        "tokens_evaluated": n_tok,
        "seconds_load": round(t_load, 2), "seconds_run": round(t_run, 2),
        "threads": args.threads, "n_ctx": cp.n_ctx, "n_batch": cp.n_batch, "n_ubatch": cp.n_ubatch,
        "seconds_per_point": round(t_run / len(points), 4),
        "loadavg_1min_during_run": {"min": round(min(loads), 2), "mean": round(sum(loads) / len(loads), 2),
                                    "max": round(max(loads), 2), "samples": len(loads)},
        "repeat_check_points": len(vidx), "repeat_check_max_abs_logp_diff": float(max(diffs)) if diffs else None,
        "llama_cpp_python": L.__version__,
        "system_info": L.llama_print_system_info().decode(),
        "cpu": cmd("lscpu | grep 'Model name' | sed 's/.*: *//'"),
        "cpus": os.cpu_count(),
        "mem_total": cmd("grep MemTotal /proc/meminfo"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
    }
    with open(os.path.join(args.out, f"{args.name}.run.json"), "w") as jf:
        json.dump(info, jf, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k not in ("chat_template", "system_info")}, indent=2))
    L.llama_batch_free(batch)
    L.llama_free(ctx)
    L.llama_model_free(model)


if __name__ == "__main__":
    main()
