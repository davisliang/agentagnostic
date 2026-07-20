"""Optimize a workflow for routerllm's ifeval task.

Reuses the notebook's cells directly (no reimplementation) but swaps in ifeval's
own constraint checker as the grader, so a score here means what it means in
routerllm's summary.json.

  selection : 100 train examples (notebook splits 60 dev / 40 internal test)
  reporting : router's 46-example holdout14 subset, untouched during selection

Usage: run_ifeval.py [--rounds N] [--smoke]
"""
import argparse, json, pathlib, statistics, sys, time

HERE = pathlib.Path(__file__).parent
REPO = pathlib.Path("/Users/davis/Documents/code/agentagnostic")
sys.path.insert(0, str(HERE))
import grader  # noqa: E402  (adds lm-eval to sys.path as a side effect)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--smoke", action="store_true", help="tiny run to prove the wiring")
ap.add_argument("--out", default=str(HERE / "ifeval_result.json"))
args = ap.parse_args()

# ---- load the notebook's machinery ----------------------------------------
cells = json.loads((REPO / "workflow_optimizer_v0.ipynb").read_text())["cells"]
src = lambda i: "".join(cells[i]["source"])
ns = {"__name__": "nb"}
for i in (2, 6, 9, 15, 20, 18):        # models, call_model, grading, pareto, runtime, agent
    exec(src(i), ns)
# the analyzer's model classes only — not the analyzer itself, which would go
# infer a task config we already know.
c12 = src(12)
exec(c12[:c12.index("def generate_json(")], ns)

# ---- the task ---------------------------------------------------------------
train = [json.loads(l) for l in open(HERE / "ifeval_train.jsonl")]
test = [json.loads(l) for l in open(HERE / "ifeval_test.jsonl")]
for r in train + test:
    r["answer"] = ""                    # unused: the constraint checker is the grader

if args.smoke:
    train, args.rounds = train[:8], 1

DESCRIPTION = (
    "Follow the instructions in the prompt exactly. Each prompt states one or more "
    "verifiable formatting constraints — for example: wrap the entire response in "
    "double quotation marks, write in all lowercase, use exactly N bullet points, "
    "avoid commas entirely, repeat the prompt before answering, end with a specific "
    "phrase, or include a postscript. An answer is correct ONLY if it satisfies EVERY "
    "constraint in the prompt; content quality is not judged at all, only compliance. "
    "The constraints are stated in the prompt itself, so a response can be checked "
    "against them before being returned."
)

ns["analysis"] = ns["TaskConfig"](
    description=DESCRIPTION, check_type="exact", judge_rubric="",
    answer_examples=['"The fire consumed my hatred."', "the answer in all lowercase"],
)
ns["CHECK_TYPE"] = "custom"
ns["JUDGE_RUBRIC"] = ""
ns["GRADER_PATH"] = str(HERE / "grader.py")

split = max(1, len(train) * 3 // 5)
ns["DATA_DEV"], ns["DATA_TEST"] = train[:split], train[split:]

# Tighter than the notebook default: caps a pathological candidate at cents, not
# dollars, per example. Still allows a generate -> verify -> revise loop.
_Runtime = ns["Runtime"]
ns["Runtime"] = lambda m, **kw: _Runtime(m, max_calls=6, token_budget=20_000)

BASE_MODEL = ns["MODELS"][0]           # haiku; a workflow may escalate itself

def evaluate(program, data):
    return ns["evaluate_program"](program, data, BASE_MODEL, grade=grader.grade)

# ---- optimize ----------------------------------------------------------------
t0, archive = time.time(), []
for rnd in range(1, args.rounds + 1):
    print(f"\n{'='*72}\n=== design round {rnd}/{args.rounds}\n{'='*72}", flush=True)
    context = ns["summarize_archive"](archive)
    for program in ns["run_design_round"](rnd, context):
        if any(program["code"] == seen["code"] for seen in archive):
            continue
        scored = evaluate(program, ns["DATA_DEV"])
        program["dev_accuracy"] = scored["accuracy"]
        program["dev_cost"] = scored["cost_per_query"]
        program["dev_records"] = scored["records"]
        archive.append(program)
        print(f"  + {program['name']:26} dev {program['dev_accuracy']:.3f}  "
              f"${program['dev_cost']:.5f}/q", flush=True)

if not archive:
    sys.exit("no candidates survived — nothing to report")

# ---- rank on the held-out slice of train, then the router's 46 -------------
front = ns["pareto_front"]([{"name": p["name"], "accuracy": p["dev_accuracy"],
                             "cost_per_query": p["dev_cost"]} for p in archive])
by_name = {p["name"]: p for p in archive}
finalists = [by_name[r["name"]] for r in front]
print(f"\n{len(archive)} candidates -> {len(finalists)} on the dev frontier", flush=True)

results = []
for p in finalists:
    internal = evaluate(p, ns["DATA_TEST"])
    held = evaluate(p, test)
    results.append({
        "name": p["name"], "description": p.get("description", ""), "code": p["code"],
        "dev_accuracy": p["dev_accuracy"], "dev_cost": p["dev_cost"],
        "internal_accuracy": internal["accuracy"], "internal_cost": internal["cost_per_query"],
        "test_accuracy": held["accuracy"], "test_cost": held["cost_per_query"],
        "test_n": len(test),
        "test_answers": [{"prompt_hash": t["prompt_hash"], "answer": r["answer"],
                          "score": r["score"], "error": r["error"]}
                         for t, r in zip(test, held["records"])],
    })
    print(f"  {p['name']:26} dev {p['dev_accuracy']:.3f} | internal {internal['accuracy']:.3f} "
          f"| TEST {held['accuracy']:.3f}  ${held['cost_per_query']:.5f}/q", flush=True)

spend = sum(r["dev_cost"] * len(ns["DATA_DEV"]) for r in
            [{"dev_cost": p["dev_cost"]} for p in archive]) \
      + sum(r["internal_cost"] * len(ns["DATA_TEST"]) + r["test_cost"] * len(test) for r in results)

pathlib.Path(args.out).write_text(json.dumps({
    "task": "ifeval", "rounds": args.rounds,
    "n_train": len(train), "n_dev": len(ns["DATA_DEV"]),
    "n_internal": len(ns["DATA_TEST"]), "n_test": len(test),
    "baselines_test": {"haiku": 0.848, "opus": 0.891, "router": 0.848, "oracle": 0.957},
    "workflow_api_spend_usd": round(spend, 4),
    "wall_clock_s": round(time.time() - t0, 1),
    "results": results,
}, indent=1))
print(f"\nworkflow API spend: ${spend:.2f}   wall clock: {(time.time()-t0)/60:.1f} min")
print(f"written: {args.out}")
