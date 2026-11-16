"""Command line entry point: fetch -> ingest -> ask / serve / eval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from plumbline.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="plumbline - a cited-answer RAG system with the unglamorous parts done properly.")
console = Console()


@app.command()
def fetch() -> None:
    """Download the corpus PDFs listed in corpus/manifest.yaml from arXiv."""
    from plumbline.ingest.fetch import fetch_corpus, load_manifest

    settings = get_settings()
    papers = load_manifest(settings.manifest)
    styles = {"downloaded": "green", "cached": "dim", "failed": "red"}
    log = fetch_corpus(
        papers, settings.raw_dir,
        on_event=lambda kind, p, detail: console.print(
            f"[{styles[kind]}]{kind:>10}[/] {p.id}  {p.short:<12} {detail}"),
    )
    failed = [pid for pid, rec in log.items() if "error" in rec]
    if failed:
        console.print(f"[red]{len(failed)} paper(s) failed - re-run to retry.[/]")
        raise typer.Exit(1)


@app.command()
def ingest(force: bool = typer.Option(False, help="Re-parse even unchanged files.")) -> None:
    """Parse, quality-check, chunk and index the corpus."""
    from plumbline.ingest.pipeline import run_ingest

    report = run_ingest(
        get_settings(), force=force,
        on_event=lambda kind, p, detail: console.print(f"{kind:>12} {p.short:<12} {detail}"),
    )
    table = Table(title="Ingestion", show_header=False)
    table.add_row("chunks before quality gates", str(report["chunks_before_gates"]))
    table.add_row("chunks indexed", str(report["chunks_indexed"]))
    table.add_row("rejected", json.dumps(report["rejected"]))
    table.add_row("kinds", json.dumps(report["chunk_kinds"]))
    table.add_row("dense index", json.dumps(report["index"]))
    table.add_row("seconds", str(report["seconds"]))
    console.print(table)


@app.command()
def ask(
    question: str,
    mode: str = typer.Option("hybrid+rerank", help="bm25 | dense | hybrid | hybrid+rerank"),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
) -> None:
    """Answer one question from the corpus, with citations."""
    from plumbline.pipeline import AskOptions, RagPipeline

    result = asyncio.run(RagPipeline.load().answer(question, AskOptions(mode=mode)))
    if as_json:
        result.pop("events", None)
        console.print_json(json.dumps(result))
        return
    colour = {"answered": "green", "degraded": "yellow", "abstained": "cyan", "failed": "red"}
    console.print(f"[bold {colour[result['status']]}]{result['status']}[/]"
                  f"  via {result['served_by'] or '-'}  in {result['trace']['total_ms']} ms")
    if result["reason"]:
        console.print(f"[dim]{result['reason']}[/]")
    for claim in result["claims"]:
        cites = ", ".join(f"{c['short']} p.{c['page']}" for c in claim["citations"])
        console.print(f"\n{claim['text']}\n  [dim]-> {cites}[/]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API and the live dashboard."""
    import uvicorn

    uvicorn.run("plumbline.api.app:app", host=host, port=port, log_level="info")


@app.command("eval")
def evaluate(
    out: Path = typer.Option(Path("reports/latest"), help="Where to write report.json / report.md."),
    gate: bool = typer.Option(False, help="Exit 1 if evals/thresholds.yaml is violated."),
    rounds: int = typer.Option(3, help="Seeded rounds per resilience cell."),
) -> None:
    """Run the retrieval, answer-quality and resilience evaluations."""
    from plumbline.evals.report import gate as check_gate
    from plumbline.evals.report import to_markdown, write_report
    from plumbline.evals.runner import run_all

    report = asyncio.run(run_all(resilience_rounds=rounds))
    write_report(report, out)
    console.print(to_markdown(report))
    console.print(f"[dim]written to {out}/[/]")
    if gate:
        failures = check_gate(report, Path("evals/thresholds.yaml"))
        if failures:
            for f in failures:
                console.print(f"[red]GATE FAILED[/] {f}")
            raise typer.Exit(1)
        console.print("[green]quality gate passed[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
