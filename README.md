# Personal Notes RAG Project

面向个人知识库的本地 RAG 项目。默认流程是：

1. 读取个人知识库 manifest
2. 将本地笔记切成 chunks
3. 写入 Qdrant
4. 使用 dense + BM25 混合检索
5. 通过 relevance gate 判断是否有可用证据
6. 将完整相关 chunks 拼入 prompt
7. LLM 基于个人知识库证据回答；无相关证据时按通识 fallback

默认模型配置为 Ollama OpenAI-compatible API 下的 `qwen3.5:9b`。

## Docker

构建：

```powershell
docker compose build test api cli
```

启动 Qdrant 和 API：

```powershell
docker compose up -d qdrant api
```

导入默认个人测试语料：

```powershell
docker compose run --rm cli personal-notes-rag ingest --manifest configs/personal_test_sources.yml
```

默认 API 地址：

```text
http://localhost:8088
```

## Configuration

关键默认值：

```text
CORPUS_MANIFEST=/app/configs/personal_test_sources.yml
QDRANT_COLLECTION=personal_kb_rag
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
SPARSE_BACKEND=bm25_jieba
INTENT_ROUTER_MODEL=qwen3.5:9b
QUERY_REWRITE_MODEL=qwen3.5:9b
TRACE_RETENTION_COUNT=30
```

个人知识库 manifest 使用通用来源字段：

```yaml
sources:
  - source_id: personal-test-note
    doc_type: note
    title: Personal Test Note
    source_ref: Personal Test Note v1, Section personal-test
    date: "2026-05-19"
    section: personal-test
    path: ../data/corpus/personal_test/my_note.txt
    metadata:
      domain: personal_kb
      topic: personal_test
      language: zh
```

## Evaluation

Docker 内跑单测：

```powershell
docker compose run --rm test pytest tests/test_llm_answer.py tests/test_api_cli_contracts.py tests/test_query_intent.py tests/test_relevance.py tests/test_observability.py -q
```

Docker 内跑个人中文全链路评测：

```powershell
docker compose run --rm cli python scripts/eval_personal_zh_retrieval.py --eval-set /app/data/eval/personal_zh_bm25_20.yml --api-url http://api:8080 --endpoint answer --mode hybrid --top-k 8 --llm-base-url http://ollama:11434/v1 --llm-model qwen3.5:9b --llm-timeout-seconds 180
```

为了避免污染 trace，评测时可以设置：

```powershell
$env:TRACE_ENABLED="false"
```

## Trace

API 默认把最近 30 次请求 trace 写入：

```text
runtime/traces/evidence_traces.json
```

CLI 查看：

```powershell
docker compose run --rm cli personal-notes-rag trace list
docker compose run --rm cli personal-notes-rag trace show
```

## Notes

- 项目不保留旧领域语料、旧 fixture、旧领域 metadata 字段或旧 CLI alias。
- API hit 使用 `source_ref` 表示来源。
- answer 使用 `sources` 表示实际返回给用户的 chunk 级来源。
- 当前回答链路不使用 support spans、claim checks 或正则答案改写。
