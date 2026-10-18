"""The markdown report: the artifact a reader looks at first."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from legalrag.eval.runner import EvalRun

COLUMNS = ("recall@5", "recall@10", "recall@20", "precision@5", "mrr", "ndcg@10")


def _metric(values: Mapping[int, float], k: int) -> str:
    value = values.get(k)
    # A k that was never computed must not read as a zero score.
    return "-" if value is None else f"{value:.2f}"


def render_report(runs: Sequence[EvalRun]) -> str:
    if not runs:
        return ""

    first = runs[0]
    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(runs)} configuration(s) over {len(first.results)} question(s).",
        "",
        "| config | " + " | ".join(COLUMNS) + " |",
        "| --- |" + " --- |" * len(COLUMNS),
    ]

    for run in runs:
        metrics = run.metrics
        lines.append(
            "| {config} | {r5} | {r10} | {r20} | {p5} | {mrr} | {ndcg10} |".format(
                config=run.config,
                r5=_metric(metrics.recall, 5),
                r10=_metric(metrics.recall, 10),
                r20=_metric(metrics.recall, 20),
                p5=_metric(metrics.precision, 5),
                mrr=f"{metrics.mrr:.2f}",
                ndcg10=_metric(metrics.ndcg, 10),
            )
        )

    return "\n".join(lines) + "\n"


def write_report(markdown: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8", newline="\n")
