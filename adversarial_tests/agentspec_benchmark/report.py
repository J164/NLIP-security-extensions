"""Aggregate benchmark result rows into Markdown and CSV tables.

    python -m adversarial_tests.agentspec_benchmark.report results/<run>/results.jsonl
    python -m adversarial_tests.agentspec_benchmark.report results/<run>/results.jsonl --paper

Each input row is one trial (case x model x defense x repetition), written by the harness:

    {"case_id", "kind", "family", "model", "defense", "rep",
     "status": "ok" | "error" | "unavailable",
     "attack_goal": bool | null, "user_task": bool | null,
     "se_fired": bool, "se_rule": str | null, "expected_rule": str | null,
     "turn_latencies_s": [float], "se_ms": [float],
     "llm_calls": int, "input_tokens": int, "output_tokens": int, "error": str | null}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from .cases import load_cases

Z95 = 1.959964


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def load(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    return rows


def rate(rows: list[dict], key: str) -> dict:
    """Share of ``ok`` rows where ``row[key]`` is true, with its Wilson interval."""
    scored = [row for row in rows if row["status"] == "ok" and row.get(key) is not None]
    hits = sum(bool(row[key]) for row in scored)
    low, high = wilson(hits, len(scored))
    return {"hits": hits, "n": len(scored), "rate": hits / len(scored) if scored else math.nan,
            "low": low, "high": high}


def summarize(rows: list[dict]) -> list[dict]:
    """One summary record per (kind, family, model, defense)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["family"], row["model"], row["defense"])].append(row)
    summary = []
    for (kind, family, model, defense), group in sorted(groups.items()):
        statuses = {s: sum(r["status"] == s for r in group) for s in ("ok", "error", "unavailable")}
        ok = [r for r in group if r["status"] == "ok"]
        record = {
            "kind": kind, "family": family, "model": model, "defense": defense,
            "trials": len(group), **statuses,
            "user_task": rate(group, "user_task"),
            "se_fired": rate(group, "se_fired"),
            "turn_p50_s": percentile([t for r in ok for t in r.get("turn_latencies_s", [])], 0.5),
            "turn_p95_s": percentile([t for r in ok for t in r.get("turn_latencies_s", [])], 0.95),
            "se_p50_ms": percentile([t for r in ok for t in r.get("se_ms", [])], 0.5),
            "se_p95_ms": percentile([t for r in ok for t in r.get("se_ms", [])], 0.95),
            "llm_calls": _mean([r.get("llm_calls", 0) for r in ok]),
            "input_tokens": _mean([r.get("input_tokens", 0) for r in ok]),
            "output_tokens": _mean([r.get("output_tokens", 0) for r in ok]),
        }
        if kind == "attack":
            record["attack_goal"] = rate(group, "attack_goal")
            matched = [r for r in ok if r.get("expected_rule")]
            record["rule_match"] = rate(
                [{**r, "match": bool(r.get("se_fired")) and r.get("se_rule") == r["expected_rule"]}
                 for r in matched],
                "match",
            )
        summary.append(record)
    return summary


def by_mechanism(rows: list[dict]) -> list[dict]:
    """Attack rows grouped by (mechanism, model, defense); rows without a mechanism are skipped."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "attack" and row.get("mechanism"):
            groups[(row["mechanism"], row["model"], row["defense"])].append(row)
    return [
        {"mechanism": m, "model": model, "defense": defense, "cases": len({r["case_id"] for r in g}),
         "attack_goal": rate(g, "attack_goal"), "user_task": rate(g, "user_task")}
        for (m, model, defense), g in sorted(groups.items())
    ]


def mechanism_markdown(records: list[dict]) -> str:
    if not records:
        return ""
    out = ["## Attack cases by mechanism\n",
           "| mechanism | model | defense | cases | ASR | utility under attack |", "|---|---|---|---|---|---|"]
    for r in records:
        out.append(f"| {r['mechanism']} | {r['model']} | {r['defense']} | {r['cases']} | "
                   f"{_pct(r['attack_goal'])} | {_pct(r['user_task'])} |")
    return "\n".join(out) + "\n\n"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _pct(cell: dict | None) -> str:
    if not cell or cell["n"] == 0:
        return "—"
    return f"{cell['rate']:.0%} [{cell['low']:.0%}–{cell['high']:.0%}] (n={cell['n']})"


def _num(value: float, fmt: str = ".2f") -> str:
    return "—" if value is None or math.isnan(value) else format(value, fmt)


def markdown(summary: list[dict]) -> str:
    out = []
    for kind, title in (("attack", "Attack cases"), ("benign", "Benign cases")):
        records = [r for r in summary if r["kind"] == kind]
        if not records:
            continue
        out.append(f"## {title}\n")
        if kind == "attack":
            header = ["family", "model", "defense", "ASR", "utility under attack",
                      "detected", "rule match", "error", "unavailable"]
        else:
            header = ["family", "model", "defense", "utility", "false positive",
                      "error", "unavailable"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for r in records:
            if r["ok"] == 0 and r["unavailable"]:
                cells = ["unavailable"] * (len(header) - 5)
            elif kind == "attack":
                cells = [_pct(r["attack_goal"]), _pct(r["user_task"]), _pct(r["se_fired"]),
                         _pct(r["rule_match"])]
            else:
                cells = [_pct(r["user_task"]), _pct(r["se_fired"])]
            out.append("| " + " | ".join(
                [r["family"], r["model"], r["defense"], *cells, str(r["error"]), str(r["unavailable"])]
            ) + " |")
        out.append("")
    out.append("## Cost\n")
    header = ["kind", "family", "model", "defense", "turn p50 s", "turn p95 s",
              "SE p50 ms", "SE p95 ms", "LLM calls", "input tok", "output tok"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in summary:
        out.append("| " + " | ".join([
            r["kind"], r["family"], r["model"], r["defense"],
            _num(r["turn_p50_s"]), _num(r["turn_p95_s"]), _num(r["se_p50_ms"]), _num(r["se_p95_ms"]),
            _num(r["llm_calls"], ".1f"), _num(r["input_tokens"], ".0f"), _num(r["output_tokens"], ".0f"),
        ]) + " |")
    return "\n".join(out) + "\n"


def write_csv(summary: list[dict], path: Path) -> None:
    flat = []
    for r in summary:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for key in ("attack_goal", "user_task", "se_fired", "rule_match"):
            if isinstance(r.get(key), dict):
                for part in ("hits", "n", "rate", "low", "high"):
                    row[f"{key}_{part}"] = r[key][part]
        flat.append(row)
    fields = sorted({k for row in flat for k in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="results.jsonl file(s)")
    parser.add_argument("--paper", action="store_true",
                        help="Also write tables to results/summary/ (committed for the paper).")
    parser.add_argument("--out", type=Path, help="Directory for summary.md / summary.csv.")
    args = parser.parse_args()

    rows = load(args.results)
    known = {c["id"]: c.get("mechanism") for c in load_cases()}
    for row in rows:  # runs from before the field existed carry no mechanism
        row.setdefault("mechanism", known.get(row["case_id"]))
    summary = summarize(rows)
    text = markdown(summary)
    text = text.replace("## Cost", mechanism_markdown(by_mechanism(rows)) + "## Cost", 1)
    print(text)
    targets = [args.out or args.results[0].parent]
    if args.paper:
        targets.append(Path(__file__).parent / "results" / "summary")
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.md").write_text(text)
        write_csv(summary, target / "summary.csv")


if __name__ == "__main__":
    main()
