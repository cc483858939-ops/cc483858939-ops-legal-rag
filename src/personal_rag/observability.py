from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from personal_rag.schema import RetrievalHit
from personal_rag.text import normalize_whitespace

_TRACE_WRITE_LOCK = threading.Lock()


class TraceCollector:
    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str | Path = "runtime/traces/evidence_traces.json",
        retention_count: int = 30,
        include_text: bool = False,
        text_chars: int = 300,
        max_items_per_stage: int = 20,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self.retention_count = max(1, int(retention_count))
        self.include_text = include_text
        self.text_chars = max(0, int(text_chars))
        self.max_items_per_stage = max(1, int(max_items_per_stage))
        self.trace_id = uuid4().hex
        self._started_perf = time.perf_counter()
        self._finished = False
        self.trace: dict = {
            "trace_id": self.trace_id,
            "started_at": utc_now(),
            "ended_at": None,
            "duration_ms": None,
            "settings": {
                "trace_retention_count": self.retention_count,
                "trace_include_text": self.include_text,
                "trace_text_chars": self.text_chars,
                "trace_max_items_per_stage": self.max_items_per_stage,
            },
            "stages": [],
            "errors": [],
        }

    @classmethod
    def from_settings(cls, settings) -> TraceCollector:
        return cls(
            enabled=bool(getattr(settings, "trace_enabled", True)),
            path=getattr(settings, "trace_path", "runtime/traces/evidence_traces.json"),
            retention_count=int(getattr(settings, "trace_retention_count", 30)),
            include_text=bool(getattr(settings, "trace_include_text", False)),
            text_chars=int(getattr(settings, "trace_text_chars", 300)),
            max_items_per_stage=int(getattr(settings, "trace_max_items_per_stage", 20)),
        )

    @property
    def public_path(self) -> str:
        return str(self.path)

    def response_metadata(self, *, persisted: bool | None = None) -> dict:
        return {
            "trace_enabled": self.enabled,
            "trace_id": self.trace_id if self.enabled else None,
            "trace_path": self.public_path if self.enabled else None,
            "trace_retention_count": self.retention_count,
            "trace_persisted": bool(persisted) if self.enabled else False,
        }

    def add_settings(self, values: dict) -> None:
        self.trace["settings"].update(jsonable(values))

    def start_stage(self, name: str, input: dict | None = None) -> int:
        stage = {
            "name": name,
            "started_at": utc_now(),
            "ended_at": None,
            "duration_ms": None,
            "input": jsonable(input or {}),
            "output": {},
            "events": [],
            "errors": [],
            "_started_perf": time.perf_counter(),
        }
        self.trace["stages"].append(stage)
        return len(self.trace["stages"]) - 1

    def end_stage(
        self,
        stage_index: int,
        output: dict | None = None,
        error: Exception | str | None = None,
    ) -> None:
        stage = self.trace["stages"][stage_index]
        stage["ended_at"] = utc_now()
        stage["duration_ms"] = elapsed_ms(stage.pop("_started_perf"))
        if output:
            stage["output"] = jsonable(output)
        if error:
            self.add_error(stage["name"], error, stage_index=stage_index)

    def add_event(self, stage_index: int, name: str, payload: dict | None = None) -> None:
        self.trace["stages"][stage_index]["events"].append(
            {
                "name": name,
                "at": utc_now(),
                "payload": jsonable(payload or {}),
            }
        )

    def add_error(
        self,
        stage_name: str,
        error: Exception | str,
        *,
        stage_index: int | None = None,
    ) -> None:
        payload = {
            "stage": stage_name,
            "type": type(error).__name__ if isinstance(error, Exception) else "Error",
            "message": str(error)[:500],
            "at": utc_now(),
        }
        self.trace["errors"].append(payload)
        if stage_index is not None:
            self.trace["stages"][stage_index]["errors"].append(payload)

    def hit_summary(self, hit: RetrievalHit, *, rank: int) -> dict:
        data = {
            "rank": rank,
            "chunk_id": hit.chunk_id,
            "source_id": hit.source_id,
            "title": hit.title,
            "source_ref": hit.source_ref,
            "doc_type": hit.doc_type,
            "scores": {
                "dense": hit.dense_score,
                "bm25": hit.bm25_score,
                "fusion": hit.fusion_score,
                "rerank": hit.rerank_score,
                "final": hit.final_score,
            },
            "rank_explanation": hit.rank_explanation,
        }
        text = normalize_whitespace(hit.text)
        if self.include_text:
            data["text"] = text[: self.text_chars] if self.text_chars else text
        else:
            data["snippet"] = text[: self.text_chars]
        return data

    def hits_summary(self, hits: list[RetrievalHit]) -> list[dict]:
        return [
            self.hit_summary(hit, rank=index + 1)
            for index, hit in enumerate(hits[: self.max_items_per_stage])
        ]

    def finish(self) -> None:
        if self._finished:
            return
        self.trace["ended_at"] = utc_now()
        self.trace["duration_ms"] = elapsed_ms(self._started_perf)
        self._finished = True

    def write_rolling_json(self) -> bool:
        if not self.enabled:
            return False
        self.finish()
        try:
            with _TRACE_WRITE_LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                existing = read_trace_array(self.path)
                existing.append(strip_private_keys(self.trace))
                existing = existing[-self.retention_count :]
                self.path.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
        except Exception as exc:  # noqa: BLE001 - tracing must not break retrieval.
            self.add_error("trace_write", exc)
            return False


def read_trace_array(path: str | Path) -> list[dict]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def find_trace(path: str | Path, trace_id: str) -> dict | None:
    for trace in read_trace_array(path):
        if trace.get("trace_id") == trace_id:
            return trace
    return None


def summarize_traces(path: str | Path) -> list[dict]:
    summaries: list[dict] = []
    for trace in read_trace_array(path):
        stages = trace.get("stages", [])
        request_stage = next(
            (stage for stage in stages if stage.get("name") == "request"),
            {},
        )
        summaries.append(
            {
                "trace_id": trace.get("trace_id"),
                "started_at": trace.get("started_at"),
                "duration_ms": trace.get("duration_ms"),
                "query": request_stage.get("input", {}).get("query"),
                "stage_count": len(stages),
                "error_count": len(trace.get("errors", [])),
            }
        )
    return summaries


def jsonable(value):
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def strip_private_keys(value):
    if isinstance(value, dict):
        return {
            key: strip_private_keys(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [strip_private_keys(item) for item in value]
    return value


def elapsed_ms(started_perf: float) -> float:
    return round((time.perf_counter() - started_perf) * 1000, 3)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
