"""Render an eval report as Markdown, and gate it against committed thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

RETRIEVAL_COLS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def to_markdown(report: dict) -> str:
    d, a = report["dataset"], report["answers"]
    lines = [
        "# Evaluation report",
        "",
        f"Generated {report['generated_at']} - {report['corpus']['papers']} papers, "
        f"{report['corpus']['chunks']} chunks, {d['items']} golden questions "
        f"({d['answerable']} answerable, {d['unanswerable']} unanswerable). "
        f"Generator: `{report['model']}`. Run time {report['seconds']}s.",
        "",
        "## 1. Retrieval ablation",
        "",
        "Share of answerable questions whose answer-bearing passage is retrieved. "
        "Reranker weights were tuned on the dev half only; **test** is held out.",
        "",
        "| Mode | Split | " + " | ".join(RETRIEVAL_COLS) + " |",
        "|---|---|" + "---:|" * len(RETRIEVAL_COLS),
    ]
    for mode, res in report["retrieval"].items():
        for split in ("test", "dev"):
            cells = " | ".join(f"{res[split][c]:.3f}" for c in RETRIEVAL_COLS)
            lines.append(f"| `{mode}` | {split} | {cells} |")
    lines += [
        "",
        "## 2. Answer quality",
        "",
        "| Split | n | Answer accuracy | Wrong answers | False refusals | Correct refusals |"
        " Citation precision | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("test", "dev", "metrics"):
        m = a[split]
        lines.append(
            f"| {'all' if split == 'metrics' else split} | {m['n']} | {_pct(m['answer_accuracy'])} | "
            f"{_pct(m['wrong_answer_rate'])} | {_pct(m['false_abstain_rate'])} | "
            f"{_pct(m['refusal_accuracy'])} | {_pct(m['citation_precision'])} | "
            f"{m['p50_ms']} ms | {m['p95_ms']} ms |")
    lines += [
        "",
        "## 3. Resilience under injected faults",
        "",
        "Every answerable question, three seeded rounds per cell. *Availability* = the user "
        "got an answer or a deliberate refusal rather than an error. *Poisoned* = a "
        "fabricated or uncited claim reached the user.",
        "",
        "| Fault profile | Variant | Availability | Correct | Poisoned served | Fallback rate | Attempts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile, variants in report["resilience"].items():
        for name, m in variants.items():
            lines.append(
                f"| {profile} | {name} | {_pct(m['availability'])} | {_pct(m['correct'])} | "
                f"{_pct(m['poisoned_served'])} | {_pct(m['fallback_rate'])} | {m['mean_attempts']} |")
    misses = [r for r in a["rows"] if r["type"] != "unanswerable" and not r["correct"]]
    lines += ["", "## Misses", "", "Answerable questions the system did not answer correctly:", ""]
    lines += [f"- `{r['id']}` ({r['split']}) - {r['status']}" for r in misses] or ["- none"]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "report.md").write_text(to_markdown(report))


def _lookup(report: dict, dotted: str):
    node = report
    for part in dotted.split("/"):
        node = node[part]
    return node


def gate(report: dict, thresholds_path: Path) -> list[str]:
    """Compare a report with committed thresholds. Returns the list of violations."""
    spec = yaml.safe_load(thresholds_path.read_text())
    failures = []
    for path, floor in (spec.get("min") or {}).items():
        value = _lookup(report, path)
        if value < floor:
            failures.append(f"{path} = {value} is below the minimum {floor}")
    for path, ceiling in (spec.get("max") or {}).items():
        value = _lookup(report, path)
        if value > ceiling:
            failures.append(f"{path} = {value} is above the maximum {ceiling}")
    return failures
