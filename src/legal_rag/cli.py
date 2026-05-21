from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from legal_rag.answer import ExtractiveAnswerer
from legal_rag.config import ROOT_DIR, load_settings
from legal_rag.eval import (
    EvalRunner,
    RetrievalEvalRunner,
    load_eval_set,
    load_retrieval_eval_set,
)
from legal_rag.factory import build_retriever, build_rewriter, build_store
from legal_rag.ingest import ingest_manifest
from legal_rag.observability import find_trace, read_trace_array, summarize_traces

app = typer.Typer(help="Legal RAG CLI")
trace_app = typer.Typer(help="Inspect local evidence traces")
app.add_typer(trace_app, name="trace")
DEFAULT_MANIFEST = ROOT_DIR / "configs" / "legal_sources.yml"
DEFAULT_ANSWER_EVAL_SET = ROOT_DIR / "data" / "eval" / "legal_rag_eval.jsonl"
DEFAULT_RETRIEVAL_EVAL_SET = ROOT_DIR / "data" / "eval" / "eval_queries.yml"


@app.command()
def ingest(
    manifest: Annotated[Path, typer.Option(help="Source manifest")] = DEFAULT_MANIFEST,
    offline: Annotated[bool, typer.Option(help="Use deterministic in-memory store")] = False,
) -> None:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    chunks = ingest_manifest(manifest, store)
    typer.echo(f"Ingested {len(chunks)} chunks from {manifest}")


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument()],
    manifest: Annotated[Path, typer.Option(help="Used only with --offline")] = DEFAULT_MANIFEST,
    top_k: Annotated[int, typer.Option()] = 8,
    mode: Annotated[str, typer.Option(help="hybrid|dense|bm25")] = "hybrid",
    debug: Annotated[bool, typer.Option()] = False,
    offline: Annotated[bool, typer.Option(help="Use deterministic in-memory store")] = False,
) -> None:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    if offline:
        ingest_manifest(manifest, store)
    retriever = build_retriever(settings, store)
    hits = retriever.retrieve(query, top_k=top_k, mode=mode)
    if debug:
        typer.echo(json.dumps([hit.model_dump(mode="json") for hit in hits], indent=2))
    else:
        for hit in hits:
            typer.echo(f"{hit.final_score:.4f}\t{hit.citation}\t{hit.title}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    manifest: Annotated[Path, typer.Option(help="Used only with --offline")] = DEFAULT_MANIFEST,
    top_k: Annotated[int, typer.Option()] = 8,
    offline: Annotated[bool, typer.Option(help="Use deterministic in-memory store")] = False,
) -> None:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    if offline:
        ingest_manifest(manifest, store)
    retriever = build_retriever(settings, store)
    hits = retriever.retrieve(question, top_k=top_k)
    answer = ExtractiveAnswerer().answer(question, hits)
    typer.echo(answer.answer)


@app.command(name="eval")
def eval_command(
    eval_set: Annotated[Path, typer.Option()] = DEFAULT_ANSWER_EVAL_SET,
    manifest: Annotated[Path, typer.Option()] = DEFAULT_MANIFEST,
    top_k: Annotated[int, typer.Option()] = 10,
    offline: Annotated[bool, typer.Option(help="Default true so seed eval is reproducible")] = True,
) -> None:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    if offline:
        ingest_manifest(manifest, store)
    retriever = build_retriever(settings, store)
    result = EvalRunner(retriever, ExtractiveAnswerer()).run(load_eval_set(eval_set), top_k=top_k)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command(name="eval-retrieval")
def eval_retrieval_command(
    eval_set: Annotated[Path, typer.Option()] = DEFAULT_RETRIEVAL_EVAL_SET,
    manifest: Annotated[Path, typer.Option()] = DEFAULT_MANIFEST,
    top_k: Annotated[int, typer.Option()] = 8,
    mode: Annotated[str, typer.Option(help="hybrid|dense|bm25")] = "hybrid",
    offline: Annotated[bool, typer.Option(help="Use deterministic in-memory store")] = False,
) -> None:
    settings = load_settings()
    store = build_store(settings, offline=offline)
    if offline:
        ingest_manifest(manifest, store)
    retriever = build_retriever(settings, store)
    rewriter = build_rewriter(settings)
    result = RetrievalEvalRunner(
        retriever,
        rewriter,
        embedder=getattr(store, "embedder", None),
        min_similarity=settings.query_extension_min_similarity,
        inferred_metadata_filters_enabled=settings.inferred_metadata_filters_enabled,
        query_rewrite_filter_min_confidence=settings.query_rewrite_filter_min_confidence,
    ).run(load_retrieval_eval_set(eval_set), top_k=top_k, mode=mode)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@trace_app.command(name="list")
def trace_list_command() -> None:
    settings = load_settings()
    typer.echo(json.dumps(summarize_traces(settings.trace_path), indent=2, ensure_ascii=False))


@trace_app.command(name="show")
def trace_show_command(
    trace_id: Annotated[str | None, typer.Option(help="Trace id to show")] = None,
) -> None:
    settings = load_settings()
    if trace_id:
        trace = find_trace(settings.trace_path, trace_id)
    else:
        traces = read_trace_array(settings.trace_path)
        trace = traces[-1] if traces else None
    if trace is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
