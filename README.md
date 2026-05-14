# legal-rag

工程化法律场景 RAG 项目，默认链路是：

1. ingest 美国法律文本并生成 citation metadata
2. 写入 Qdrant named vectors：`dense` 和 `bm25`
3. 查询时并行 dense embedding 召回与 BM25 sparse 召回
4. 使用 weighted RRF 融合
5. 如果配置 `RERANKER_MODEL`，执行 cross-encoder rerank；否则跳过
6. 仅基于证据生成 legal information answer，证据不足时拒答

## 默认模型

- Dense embedding：`BAAI/bge-large-en-v1.5` via FastEmbed/ONNX
- Sparse BM25：`Qdrant/bm25`
- 可选 reranker：`BAAI/bge-reranker-large`，默认不安装、不启用

备选：

- FastEmbed embedding：`BAAI/bge-base-en-v1.5`、`sentence-transformers/all-MiniLM-L6-v2`、`snowflake/snowflake-arctic-embed-l`、`intfloat/multilingual-e5-large`
- SentenceTransformers embedding：`BAAI/bge-m3`、`intfloat/e5-large-v2`、`Snowflake/snowflake-arctic-embed-l-v2.0`，需要 `EMBEDDING_BACKEND=sentence-transformers` 和 `INSTALL_MODEL_EXTRAS=true`
- Reranker：`cross-encoder/ms-marco-MiniLM-L6-v2`、`BAAI/bge-reranker-large`、`mixedbread-ai/mxbai-rerank-large-v1`，需要 `INSTALL_MODEL_EXTRAS=true`
- BM25：`Pyserini/Lucene`、`Elasticsearch/OpenSearch`、`rank-bm25`
- Query rewrite：默认关闭；可用 Ollama + `gemma4:e2b` 在检索前把中文/口语问题扩展成英文案名、法条、缩写等检索词。

## 本地运行

```powershell
copy .env.example .env
docker compose build
docker compose up -d qdrant
docker compose run --rm cli legal-rag ingest --manifest configs/legal_sources.yml --offline
docker compose run --rm cli legal-rag retrieve "What does 5 U.S.C. 553 require for notice?" --offline --debug
docker compose run --rm cli legal-rag ask "What does 17 U.S.C. 107 say about fair use?" --offline
docker compose run --rm test
```

API：

```powershell
docker compose up api
```

Endpoints：

- `GET /health`
- `POST /retrieve`
- `POST /evidence`
- `POST /query`

## Gemma 4 查询改写

本项目可以在 `/evidence` 检索前增加一层本地模型 query rewrite，但仍不生成法律回答。
模型只输出结构化检索词，后端继续返回 evidence pack。

推荐这台 16GB RAM + RTX 4060 Laptop 8GB VRAM 机器先用 `gemma4:e2b`。`gemma4:e4b`
也可能跑得动，但模型包更大、延迟更高；query rewrite 这种短任务优先选 E2B。

不需要在 Windows 宿主机安装 Ollama。使用 Docker profile 启动可选 Ollama 服务：

```powershell
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull gemma4:e2b
```

启用 API 的 query rewrite：

```powershell
$env:QUERY_REWRITE_BACKEND="ollama"
$env:QUERY_REWRITE_MODEL="gemma4:e2b"
docker compose --profile llm up -d qdrant ollama api
```

如果 Ollama 没启动、模型没拉取或改写超时，`/evidence` 会自动退回原始 query 检索，
并在返回 JSON 的 `query_rewrite.error` 中说明原因。

## 测试

单元测试默认不依赖外部 LLM、Hugging Face 下载或 Qdrant 服务：

```powershell
docker compose run --rm test
```

默认 `api` 和 `cli` 服务使用轻量 FastEmbed 路径，不安装 `sentence-transformers/torch`。
当前轻量镜像大小约为 `529MB`，Qdrant 镜像约为 `271MB`。首次真实 ingest 会把
FastEmbed/Hugging Face 模型权重下载到 Docker volume `legal-rag_model_cache`。

如果要启用 cross-encoder reranker 或 SentenceTransformers embedding backend，再显式构建重型镜像：

```powershell
$env:INSTALL_MODEL_EXTRAS="true"
$env:RERANKER_MODEL="BAAI/bge-reranker-large"
docker compose build api
docker compose up -d qdrant api
```

注意：重型 extras 会安装 PyTorch；默认 PyPI 解析可能拉入 GPU/CUDA wheel，镜像可达到数 GB。

测试覆盖：

- GovInfo / CourtListener fixture loader
- citation metadata 和 chunking
- dense + BM25 双路召回
- weighted RRF 融合
- reranker 配置为空时跳过、配置时执行
- seed eval 指标计算与 no-answer refusal

## 数据说明

`configs/legal_sources.yml` 固定首版美国法律语料来源，`data/eval/legal_rag_eval.jsonl`
提供 60 条 seed eval。仓库内 fixture 是小型公开法律摘录，用于离线测试；真实 ingest 可以按
manifest 从 GovInfo、CourtListener 或本地文件扩展。

本项目输出 legal information，不提供法律建议。所有回答都必须基于检索证据并带 citation。
