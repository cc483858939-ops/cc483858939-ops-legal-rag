from __future__ import annotations

from legal_rag.observability import TraceCollector, read_trace_array
from legal_rag.schema import RetrievalHit


def test_trace_collector_keeps_default_recent_ten(tmp_path) -> None:
    trace_path = tmp_path / "evidence_traces.json"

    for index in range(11):
        trace = TraceCollector(path=trace_path)
        stage = trace.start_stage("request", {"query": f"q-{index}"})
        trace.end_stage(stage, {"ok": True})
        trace.write_rolling_json()

    traces = read_trace_array(trace_path)

    assert len(traces) == 10
    assert traces[0]["stages"][0]["input"]["query"] == "q-1"
    assert traces[-1]["stages"][0]["input"]["query"] == "q-10"


def test_trace_collector_retention_count_is_configurable(tmp_path) -> None:
    trace_path = tmp_path / "evidence_traces.json"

    for index in range(4):
        trace = TraceCollector(path=trace_path, retention_count=3)
        stage = trace.start_stage("request", {"query": f"q-{index}"})
        trace.end_stage(stage, {"ok": True})
        trace.write_rolling_json()

    traces = read_trace_array(trace_path)

    assert len(traces) == 3
    assert [item["stages"][0]["input"]["query"] for item in traces] == ["q-1", "q-2", "q-3"]


def test_trace_stage_records_latency_and_error(tmp_path) -> None:
    trace = TraceCollector(path=tmp_path / "trace.json")
    stage = trace.start_stage("query_rewrite", {"query": "notice"})

    trace.end_stage(stage, error=ValueError("bad rewrite"))

    recorded = trace.trace["stages"][0]
    assert recorded["duration_ms"] >= 0
    assert recorded["errors"][0]["stage"] == "query_rewrite"
    assert trace.trace["errors"][0]["message"] == "bad rewrite"


def test_trace_hit_summary_omits_full_text_by_default(tmp_path) -> None:
    hit = RetrievalHit(
        chunk_id="1",
        source_id="s",
        doc_type="statute",
        title="Notice",
        citation="5 U.S.C. § 553",
        jurisdiction="US",
        text="x" * 1000,
        fusion_score=0.1,
    )
    trace = TraceCollector(path=tmp_path / "trace.json", include_text=False, text_chars=20)

    summary = trace.hit_summary(hit, rank=1)

    assert "text" not in summary
    assert summary["snippet"] == "x" * 20
