"""Build the `ml_papers_<as_of>` benchmark: problem -> solution over cited, post-cutoff ML papers.

Each example is a problem/solution pair extracted from the abstract of a machine
learning paper that (a) was submitted to arXiv AFTER the training-data cutoff of
the newest models in the catalog — Jan 2026 for claude-sonnet-5 and
claude-opus-4-8, per Anthropic's model docs — so no model can have memorized the
reference solutions, and (b) has MORE THAN 10 citations already (per Semantic
Scholar), so the pairs come from work the field actually built on. Diversity is
enforced by round-robining candidates across arXiv primary categories before
extraction, so no single subfield dominates the set.

The task: given the problem, propose a solution; an LLM judge scores the
proposal against the approach the paper actually took.

Only pairs that CLEARLY make sense are kept: the extractor must state a
self-contained problem that does not leak the approach, and a specific technical
solution — abstracts for surveys, dataset releases, position papers, or vague
"novel framework" claims are discarded. Extraction runs through the **Claude
Agent SDK** (the local Claude Code login), NOT the metered API — so building the
dataset bills the subscription, not ANTHROPIC_API_KEY. A re-run does not
reproduce the same pairs bit-for-bit; the frozen benchmark directory is the
artifact, and `sources.json` records every kept paper's id, title, date, and
citation count.

The benchmark's NAME carries the build date (`ml_papers_<as_of>`), so the name
itself says when the data goes stale and should be recycled.

    uv run python scripts/build_ml_papers.py                 # subscription-billed extraction
    uv run python scripts/build_ml_papers.py --dry-run       # fetch + filter only, no model calls
"""
import argparse
import asyncio
import datetime
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Papers submitted after this date post-date the training data of every model in
# the catalog: claude-sonnet-5 and claude-opus-4-8 both have a Jan 2026 training
# cutoff (claude-haiku-4-5's is older still), per the Anthropic model overview.
MODEL_CUTOFF = "2026-02-01"

ML_CATEGORIES = {"cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"}
ATOM = {"atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"}

# The description the design agent sees (config/task/*.yaml -> the design prompt).
# It states the task and the answer shape; it does not say the problems come from
# recent papers, so a workflow can't be told to go look the answers up.
TASK_DESCRIPTION = (
    "You are given a research problem in machine learning. Propose a concrete "
    "technical approach that solves it: the key idea and mechanism, in a few "
    "sentences. Answer with the proposed approach only — no preamble.")

# Human-only blurb for the UI benchmark picker; never reaches the model.
BENCH_DESCRIPTION = (
    "Problem -> solution over cited (>10 citations) arXiv ML papers submitted "
    "after the Jan 2026 training cutoff of the newest catalog models. A judge "
    "scores a proposed solution against the approach the paper actually took; "
    "post-cutoff, so the reference solutions cannot have been memorized.")

# What the judge scores against. The reference is A solution, not THE only one —
# the rubric says so, and grading_note in benchmark.yaml states the consequence.
JUDGE_RUBRIC = (
    "The reference answer is the approach a recent research paper actually took "
    "for this problem. Score whether the candidate independently arrives at the "
    "reference approach's core technical idea: 90-100 = same key mechanism or "
    "idea, even if worded differently; 60-80 = overlaps on the central idea but "
    "misses or replaces key components; 30-50 = a plausible generic approach "
    "that does not capture what makes the reference work; 0-20 = vague, "
    "restates the problem, or technically incoherent. Judge substance, not "
    "style or length.")

ANSWER_EXAMPLES = [
    "Train a small router on embedding features to send easy queries to the "
    "cheap model and only escalate uncertain ones, using agreement between two "
    "cheap samples as the uncertainty signal.",
    "Replace the dense attention over the full context with a two-stage scheme: "
    "a lightweight scorer selects the top-k relevant blocks, and full attention "
    "runs only within the selected blocks.",
]

EXTRACT_PROMPT = """For each machine-learning paper below, extract a problem/solution pair for a reasoning benchmark, where a model will be shown the problem and asked to propose a solution.

Rules:
- "problem": the research problem in 1-3 sentences, fully self-contained — understandable with no access to the paper, no phrases like "this paper" or "the authors", and it must NOT hint at or reveal the paper's approach.
- "solution": the paper's actual approach in 2-4 sentences — the key idea and mechanism, specific enough that a correct independent proposal could be recognized against it.
- "clear": true ONLY if the abstract states both a specific problem and a specific technical approach. Set false for surveys, position papers, benchmark or dataset releases, evaluation-only studies, abstracts whose problem only makes sense inside a niche subfield, or approaches too vague to state (e.g. "a novel framework"). If clear is false, still fill problem and solution with your best attempt.

Reply with ONLY a JSON array — no markdown fences, no commentary — with one object per paper:
[{"arxiv_id": "...", "clear": true/false, "problem": "...", "solution": "..."}]

{papers}"""


def http_get(url: str, data: bytes = None, headers: dict = None, tries: int = 6) -> bytes:
    """Fetch with backoff — both arXiv and Semantic Scholar 429 bursts.

    Args:
        url: The URL to fetch.
        data: Optional POST body.
        headers: Optional request headers.
        tries: Attempts before giving up.

    Returns:
        The response body.
    """
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))
        except (TimeoutError, urllib.error.URLError):
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def fetch_cited_papers(window: str, min_citations: int, log=print) -> list[dict]:
    """Find papers in the window with more than `min_citations` citations.

    Citation counts come from Semantic Scholar — arXiv has none. The bulk search
    is paginated by continuation token; the text query is broad ML vocabulary
    (the authoritative ML filter is arXiv categories, applied later).

    Args:
        window: "YYYY-MM-DD:YYYY-MM-DD" publication-date range.
        min_citations: Keep papers cited STRICTLY more than this.
        log: Where progress lines go.

    Returns:
        Dicts with "arxiv_id" (bare, no version) and "citations", most cited first.
    """
    base = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
            + urllib.parse.urlencode({
                "query": "learning | model | neural | agent | language | transformer "
                         "| diffusion | reinforcement | vision | retrieval",
                "publicationDateOrYear": window,
                "minCitationCount": str(min_citations + 1),
                "fieldsOfStudy": "Computer Science",
                "fields": "citationCount,externalIds"}))
    papers, token = [], None
    while True:
        data = json.loads(http_get(base + (f"&token={token}" if token else "")))
        for paper in data.get("data", []):
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            if arxiv_id:
                papers.append({"arxiv_id": arxiv_id.split("v")[0],
                               "citations": paper.get("citationCount") or 0})
        log(f"  {len(papers)} arXiv-backed papers with >{min_citations} citations "
            f"(of {data.get('total')} candidates)...")
        token = data.get("token")
        if not token:
            break
        time.sleep(2)
    papers.sort(key=lambda p: -p["citations"])
    return papers


def fetch_arxiv_details(papers: list[dict], since: str = MODEL_CUTOFF,
                        log=print) -> list[dict]:
    """Look the papers up on arXiv for title, abstract, and categories.

    This is also the authoritative ML filter — a paper whose categories don't
    intersect ML_CATEGORIES (power electronics can clear a citation bar too) is
    dropped here — AND the authoritative cutoff filter: Semantic Scholar can
    date a paper by its venue publication even when its arXiv preprint is
    older, so the post-cutoff guarantee is enforced against arXiv's
    `published` field, which is the FIRST version's date even for papers
    revised since.

    Args:
        papers: `fetch_cited_papers` output.
        since: Drop papers whose v1 submission predates this date.
        log: Where progress lines go.

    Returns:
        Dicts with arxiv_id, citations, title, abstract, primary_category,
        published, url — ML papers first submitted after the cutoff only.
    """
    by_id = {p["arxiv_id"]: p for p in papers}
    detailed = []
    ids = list(by_id)
    for at in range(0, len(ids), 100):
        url = ("https://export.arxiv.org/api/query?id_list="
               + ",".join(ids[at:at + 100]) + "&max_results=100")
        feed = ET.fromstring(http_get(url))
        for entry in feed.findall("atom:entry", ATOM):
            arxiv_id = entry.findtext("atom:id", "", ATOM).rsplit("/", 1)[-1].split("v")[0]
            if arxiv_id not in by_id:
                continue
            categories = {c.get("term") for c in entry.findall("atom:category", ATOM)}
            primary = entry.find("arxiv:primary_category", ATOM)
            primary = primary.get("term") if primary is not None else ""
            if not (categories & ML_CATEGORIES):
                continue                       # cited, recent — but not ML
            if entry.findtext("atom:published", "", ATOM)[:10] < since:
                continue                       # v1 predates the cutoff — memorizable
            detailed.append({
                "arxiv_id": arxiv_id,
                "citations": by_id[arxiv_id]["citations"],
                "title": " ".join(entry.findtext("atom:title", "", ATOM).split()),
                "abstract": " ".join(entry.findtext("atom:summary", "", ATOM).split()),
                "primary_category": primary,
                "published": entry.findtext("atom:published", "", ATOM)[:10],
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            })
        log(f"  {len(detailed)} ML papers detailed from {min(at + 100, len(ids))} looked up...")
        time.sleep(3)                          # arXiv asks for a pause between requests
    return detailed


def interleave_by_category(papers: list[dict]) -> list[dict]:
    """Order candidates so extraction draws evenly across subfields.

    Within a category the most-cited come first; across categories a round-robin
    ensures that stopping at `--target` keeps the set diverse instead of
    whatever one hot subfield submitted that month.

    Args:
        papers: `fetch_arxiv_details` output.

    Returns:
        The same papers, round-robin ordered by primary category.
    """
    groups: dict[str, list[dict]] = {}
    for paper in papers:                       # already most-cited first
        groups.setdefault(paper["primary_category"], []).append(paper)
    ordered, rank = [], 0
    while any(groups.values()):
        for category in sorted(groups):
            if rank < len(groups[category]):
                ordered.append(groups[category][rank])
        rank += 1
        if rank > max(len(g) for g in groups.values()):
            break
    return ordered


async def _ask_claude(prompt: str, model: str) -> str:
    """One tool-less Claude Agent SDK call, returning the reply text.

    This is the same auth path the design agent uses — the local Claude Code
    login — so extraction bills the subscription, not an API key.

    Args:
        prompt: The prompt to send.
        model: The model to run.

    Returns:
        The assistant's text, all blocks joined.
    """
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    options = ClaudeAgentOptions(model=model, allowed_tools=[],
                                 permission_mode="bypassPermissions")
    parts = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "".join(parts)


def _parse_pairs(reply: str) -> list[dict]:
    """Read the extraction reply's JSON array, tolerating stray fences.

    Args:
        reply: The model's reply text.

    Returns:
        The parsed list of pair dicts.

    Raises:
        ValueError: No JSON array could be parsed out of the reply.
    """
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in reply")
    parsed = json.loads(reply[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("reply is not a list")
    return parsed


async def extract_pairs(papers: list[dict], target: int, model: str,
                        log=print) -> list[dict]:
    """Extract problem/solution pairs, keeping only the clear ones.

    Papers go through in batches of 8 abstracts per model call, three calls in
    flight at a time, in the diversity order — so stopping at `target` keeps the
    category balance. A batch whose reply doesn't parse is retried once, then
    dropped (logged, never silent).

    Args:
        papers: Candidates in extraction order.
        target: Stop once this many pairs are kept.
        model: Model the SDK runs for extraction.
        log: Where progress lines go.

    Returns:
        Kept pairs: provenance fields plus "problem" and "solution".
    """
    def render(batch: list[dict]) -> str:
        blocks = [f"Paper (arxiv_id: {p['arxiv_id']})\nTitle: {p['title']}\n"
                  f"Abstract: {p['abstract']}" for p in batch]
        return EXTRACT_PROMPT.replace("{papers}", "\n\n".join(blocks))

    async def run_batch(batch: list[dict]) -> list[dict]:
        by_id = {p["arxiv_id"]: p for p in batch}
        for attempt in (1, 2):
            try:
                rows = _parse_pairs(await _ask_claude(render(batch), model))
                break
            except Exception as error:
                if attempt == 2:
                    log(f"  (dropping a batch of {len(batch)}: {type(error).__name__})")
                    return []
        kept = []
        for row in rows:
            paper = by_id.get(str(row.get("arxiv_id", "")).split("v")[0])
            if not (paper and row.get("clear")
                    and str(row.get("problem", "")).strip()
                    and str(row.get("solution", "")).strip()):
                continue
            kept.append({**{k: paper[k] for k in (
                            "arxiv_id", "title", "published", "url",
                            "citations", "primary_category")},
                         "problem": str(row["problem"]).strip(),
                         "solution": str(row["solution"]).strip()})
        return kept

    kept, seen = [], set()
    batches = [papers[at:at + 8] for at in range(0, len(papers), 8)]
    for at in range(0, len(batches), 3):       # three model calls in flight
        for pairs in await asyncio.gather(*(run_batch(b) for b in batches[at:at + 3])):
            for pair in pairs:
                key = re.sub(r"[^a-z0-9]+", " ", pair["problem"].lower()).strip()
                if key and key not in seen:
                    seen.add(key)
                    kept.append(pair)
        done = min((at + 3) * 8, len(papers))
        log(f"  {len(kept)} clear pairs from {done} abstracts")
        if len(kept) >= target:
            break
    return kept[:target]


def freeze(pairs: list[dict], window: str, min_citations: int) -> str:
    """Write the benchmark directory and task config for the kept pairs.

    Args:
        pairs: `extract_pairs` output.
        window: The submission window, recorded for provenance.
        min_citations: The citation bar, recorded for provenance.

    Returns:
        The stamped benchmark name.
    """
    as_of = datetime.date.today().isoformat()
    name = f"ml_papers_{as_of.replace('-', '')}"   # the name says when it goes stale

    bench_dir = ROOT / "benchmarks" / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "sources.json").write_text(json.dumps(
        {"as_of": as_of, "window": window, "min_citations": min_citations,
         "cutoff_basis": "training data cutoff of claude-sonnet-5 / claude-opus-4-8 "
                         "(Jan 2026, per the Anthropic model overview)",
         "items": pairs}, indent=1))
    (bench_dir / "data.jsonl").write_text("".join(
        json.dumps({"question": p["problem"], "answer": p["solution"]}) + "\n"
        for p in pairs))
    (bench_dir / "benchmark.yaml").write_text(
        f"# {name} — problem->solution over cited post-cutoff ML papers,\n"
        f"# built by scripts/build_ml_papers.py\n"
        f"# Papers submitted {window} (after the Jan 2026 model training cutoff)\n"
        f"# with >{min_citations} citations each (Semantic Scholar).\n"
        f"name: {name}\n"
        f"description: >-\n  {BENCH_DESCRIPTION}\n"
        "source_dataset: arxiv\n"
        f"as_of: {as_of}\n"
        f"window: {window!r}\n"
        f"examples: {len(pairs)}\n"
        "grading_supported: true\n"
        "check_type: llm_judge\n"
        "grading_note: >-\n"
        "  Judge-graded against the paper's actual approach, so 'accuracy' is mean\n"
        "  similarity to the reference solution — an alternative valid solution\n"
        "  scores low by design.\n")

    examples_yaml = "\n".join(f"    - {json.dumps(e)}" for e in ANSWER_EXAMPLES)
    rubric = JUDGE_RUBRIC.replace("\n", " ")
    (ROOT / "config" / "task" / f"{name}.yaml").write_text(
        "# Problem -> solution over cited (>10 citations) post-cutoff arXiv ML papers.\n"
        "# The papers post-date the Jan 2026 training cutoff of the newest catalog\n"
        "# models, so the reference solutions cannot be recalled — but they COULD be\n"
        "# searched, which is why the task is closed-book: it measures whether a\n"
        "# workflow can reason its way to an approach, not whether it can find the PDF.\n"
        f"# Timestamped: papers submitted {window}.\n"
        "task:\n"
        f"  name: {name}\n"
        f"  description: >-\n    {TASK_DESCRIPTION}\n"
        "  check_type: llm_judge\n"
        f"  dataset: benchmarks/{name}/data.jsonl\n"
        f"  judge_rubric: >-\n    {rubric}\n"
        "  answer_examples:\n"
        f"{examples_yaml}\n"
        "runtime:\n"
        "  tools: []          # closed-book: the papers are online, looking them up is cheating\n"
        "data:\n"
        f"  n_examples: {len(pairs)}     # use all of them ({len(pairs)} kept pairs)\n")
    return name


def main() -> None:
    """Fetch, filter, extract, and freeze — or stop after the filter with --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=100,
                        help="stop once this many clear pairs are kept (default 100)")
    parser.add_argument("--min", type=int, default=40, dest="minimum",
                        help="fail if fewer than this many pairs survive (default 40)")
    parser.add_argument("--min-citations", type=int, default=10,
                        help="keep papers cited strictly more than this (default 10)")
    parser.add_argument("--since", default=MODEL_CUTOFF,
                        help=f"window start, defaults to the model cutoff ({MODEL_CUTOFF})")
    parser.add_argument("--model", default="claude-sonnet-5",
                        help="model the SDK runs for extraction (default claude-sonnet-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and filter only; no model calls, nothing written")
    args = parser.parse_args()

    window = f"{args.since}:{datetime.date.today().isoformat()}"
    print(f"papers submitted {window} with >{args.min_citations} citations...")
    cited = fetch_cited_papers(window, args.min_citations)
    papers = interleave_by_category(fetch_arxiv_details(cited, since=args.since))
    spread = {}
    for paper in papers:
        spread[paper["primary_category"]] = spread.get(paper["primary_category"], 0) + 1
    print(f"{len(papers)} ML candidates across categories: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(spread.items())))
    if args.dry_run:
        for paper in papers[:10]:
            print(f"  {paper['citations']:4d} cites  {paper['primary_category']:9s} "
                  f"{paper['published']}  {paper['title'][:70]}")
        return

    print(f"extracting with {args.model} via the Claude Agent SDK "
          f"(subscription auth; stops at {args.target} kept)...")
    pairs = asyncio.run(extract_pairs(papers, args.target, args.model))
    if len(pairs) < args.minimum:
        raise SystemExit(f"only {len(pairs)} clear pairs survived, below the "
                         f"--min {args.minimum} floor — lower --min-citations or "
                         f"widen --since")
    name = freeze(pairs, window, args.min_citations)
    kept_spread = {}
    for pair in pairs:
        kept_spread[pair["primary_category"]] = kept_spread.get(pair["primary_category"], 0) + 1
    print(f"kept {len(pairs)} pairs: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(kept_spread.items())))
    print(f"wrote benchmarks/{name}/ and config/task/{name}.yaml")


if __name__ == "__main__":
    main()
