"""Run a model on the 100 practice problems (index % 5 == 0 of MATH-500) via Ollama and grade.

Same prompt / sampling as bench_math.py (non-thinking mode), plus a fixed seed.
Results -> weight_edit/runs/<name>.jsonl (resumable). Prints a failure-mode summary.
Budget: 100 problems = 2 forward passes (64 + 36 sequences); --limit N for fewer.
"""
import argparse, json, os, re, time, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from math_verify import parse, verify

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)


def ask(model, problem, num_predict, seed):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(problem=problem)}],
        "think": False, "stream": False,
        "options": {**SAMPLING, "seed": seed, "num_ctx": num_predict + 1024, "num_predict": num_predict},
    }, timeout=3600)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("eval_count", 0), d.get("done_reason")


def grade(gold, answer):
    g = parse("$" + gold + "$")
    p = parse(answer)
    return bool(p) and verify(g, p), (str(p[-1]) if p else None)


def loopiness(text, n=30):
    """fraction of the last 1500 chars covered by a repeated 30-char chunk (crude loop detector)."""
    tail = text[-1500:]
    if len(tail) < 2 * n:
        return 0.0
    chunks = [tail[i:i + n] for i in range(0, len(tail) - n, 10)]
    c = Counter(chunks)
    return max(c.values()) / max(1, len(chunks))


def classify(res, num_predict):
    if res["correct"]:
        return "correct"
    if res["tokens"] >= num_predict or res.get("done_reason") == "length":
        return "truncated" + ("_loop" if res["loop"] > 0.2 else "")
    if "\\boxed" not in res["response"]:
        return "no_boxed"
    if res["pred"] is None:
        return "unparsable"
    return "wrong_answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated unique_ids to run (subset)")
    a = ap.parse_args()

    rows = json.load(open(os.path.join(HERE, "practice.json")))
    if a.ids:
        keep = set(a.ids.split(","))
        rows = [r for r in rows if r["unique_id"] in keep]
    rows = rows[: a.limit]
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"{a.name}.jsonl")
    done = {}
    if os.path.exists(out):
        for l in open(out):
            x = json.loads(l); done[x["unique_id"]] = x
    todo = [r for r in rows if r["unique_id"] not in done]
    print(f"{a.model}: {len(rows)} problems, {len(done)} done, {len(todo)} to run -> {out}", flush=True)
    lock = threading.Lock(); t0 = time.time()

    def work(r):
        content, ntok, why = ask(a.model, r["problem"], a.num_predict, a.seed)
        return r, content, ntok, why

    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        futs = [ex.submit(work, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r, content, ntok, why = fu.result()
            ok, pred = grade(r["answer"], content)
            res = dict(unique_id=r["unique_id"], level=r["level"], subject=r["subject"], gold=r["answer"],
                       pred=pred, correct=ok, tokens=ntok, done_reason=why, loop=loopiness(content),
                       response=content)
            res["mode"] = classify(res, a.num_predict)
            with lock:
                f.write(json.dumps(res) + "\n"); f.flush(); done[res["unique_id"]] = res
                n = sum(x["correct"] for x in done.values())
                print(f"[{i}/{len(todo)}] {res['mode']:14s} acc={n}/{len(done)} tok={ntok} "
                      f"gold={r['answer'][:20]!r} pred={str(pred)[:20]!r} {time.time()-t0:.0f}s", flush=True)

    res = [done[r["unique_id"]] for r in rows if r["unique_id"] in done]
    n = sum(x["correct"] for x in res)
    print(f"\n== {a.model} [{a.name}] practice: {n}/{len(res)} = {n/len(res):.1%}  (median tokens {sorted(x['tokens'] for x in res)[len(res)//2]})")
    print("modes:", dict(Counter(x["mode"] for x in res)))
    for key in ("level", "subject"):
        g = {}
        for x in res: g.setdefault(x[key], []).append(x["correct"])
        print(f"by {key}: " + ", ".join(f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))))


if __name__ == "__main__":
    main()
