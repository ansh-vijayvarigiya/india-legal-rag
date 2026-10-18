from legalrag.eval.metrics import RetrievalMetrics
from legalrag.eval.report import render_report, write_report
from legalrag.eval.runner import EvalRun


def _run(config, recall5):
    metrics = RetrievalMetrics(
        recall={5: recall5, 10: recall5},
        precision={5: recall5 / 2, 10: recall5 / 4},
        mrr=recall5,
        ndcg={5: recall5, 10: recall5},
    )
    return EvalRun(config=config, ks=(5, 10), metrics=metrics, results=())


def test_render_report_lists_each_configuration():
    markdown = render_report([_run("dense-fixed", 0.62), _run("dense-structured", 0.71)])
    assert "dense-fixed" in markdown
    assert "dense-structured" in markdown
    assert "0.62" in markdown
    assert "0.71" in markdown
    assert markdown.count("|") > 8


def test_render_report_handles_no_runs():
    assert "dense" not in render_report([])


def test_render_report_marks_an_uncomputed_k():
    markdown = render_report([_run("dense-fixed", 0.62)])
    # the fixture computes k=5 and k=10 only, so recall@20 must not read as a zero score
    assert "| dense-fixed | 0.62 | 0.62 | - | 0.31 | 0.62 | 0.62 |" in markdown


def test_write_report_uses_utf8_and_lf(tmp_path):
    path = tmp_path / "results.md"
    write_report("# heading\nsecond line\n", path)
    assert path.read_bytes() == b"# heading\nsecond line\n"
