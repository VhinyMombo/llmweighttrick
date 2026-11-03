"""Run MATH-500 against a local Ollama model and grade with math-verify.

Usage:
  python bench_math.py --mode nothink            # all 500
  python bench_math.py --mode think --limit 20   # quick sample
Results are appended to results/<model>_<mode>.jsonl, so re-running resumes.
"""
import argparse, json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from math_verify import parse, verify

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Qwen3's recommended sampling settings per mode
SAMPLING = {
    "think":   dict(temperature=0.6, top_p=0.95, top_k=20),
    "nothink": dict(temperature=0.7, top_p=0.8,  top_k=20),
}


def ask(model, problem, mode, num_predict):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(problem=problem)}],
        "think": mode == "think",
        "stream": False,
        "options": {**SAMPLING[mode], "num_ctx": num_predict + 1024, "num_predict": num_predict},
    }, timeout=3600)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("eval_count", 0)


def grade(gold, answer):
    g = parse("$" + gold + "$")
    p = parse(answer)
    return bool(p) and verify(g, p), (str(p[-1]) if p else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--mode", choices=["think", "nothink"], default="nothink")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=None)
    a = ap.parse_args()
    num_predict = a.num_predict or (16384 if a.mode == "think" else 4096)

    here = os.path.dirname(os.path.abspath(__file__))
    rows = [json.loads(l) for l in open(os.path.join(here, "data/math500/test.jsonl"))][: a.limit]
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    out = os.path.join(here, "results", f"{a.model.replace(':', '_')}_{a.mode}.jsonl")

    done = {}
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l); done[x["unique_id"]] = x
    todo = [r for r in rows if r["unique_id"] not in done]
    print(f"{len(rows)} problems, {len(done)} already done, {len(todo)} to run -> {out}", flush=True)

    lock = threading.Lock()
    t0 = time.time()

    def work(r):
        content, ntok = ask(a.model, r["problem"], a.mode, num_predict)
        return r, content, ntok

    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        futs = [ex.submit(work, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r, content, ntok = fu.result()
            ok, pred = grade(r["answer"], content)  # main thread: math-verify uses signal.alarm
            res = dict(unique_id=r["unique_id"], level=r["level"], subject=r["subject"],
                       gold=r["answer"], pred=pred, correct=ok, tokens=ntok, response=content)
            with lock:
                f.write(json.dumps(res) + "\n"); f.flush()
                done[res["unique_id"]] = res
                n = sum(x["correct"] for x in done.values())
                print(f"[{i}/{len(todo)}] {'OK ' if res['correct'] else 'BAD'} "
                      f"acc={n}/{len(done)}={n/len(done):.1%} tok={res['tokens']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # summary
    res = [done[r["unique_id"]] for r in rows if r["unique_id"] in done]
    n = sum(x["correct"] for x in res)
    print(f"\n== {a.model} [{a.mode}] MATH-500: {n}/{len(res)} = {n/len(res):.1%}")
    for key in ("level", "subject"):
        groups = {}
        for x in res:
            groups.setdefault(x[key], []).append(x["correct"])
        print(f"\nby {key}:")
        for k in sorted(groups):
            g = groups[k]
            print(f"  {str(k):28s} {sum(g):3d}/{len(g):3d} = {sum(g)/len(g):.1%}")


if __name__ == "__main__":
    main()
