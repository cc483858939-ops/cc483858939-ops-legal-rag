# legal-rag

工程化法律场景 RAG 项目，默认链路是：

1. ingest 美国法律文本并生成 citation metadata
2. 写入 Qdrant named vectors：`dense` 和 `bm25`
3. 查询时并行 dense embedding 召回与 BM25 sparse 召回
4. 使用 weighted RRF 融合
5. 如果配置 `RERANKER_MODEL`，执行 cross-encoder rerank；否则跳过
6. 仅基于证据生成 legal information answer，证据不足时拒答

## 默认模型

- Dense embedding：`BAAI/bge-small-zh-v1.5` via FastEmbed/ONNX，默认优先低内存中文检索
- Sparse BM25：默认 `bm25-jieba` 应用层中文 BM25；兼容模式可显式启用 `Qdrant/bm25`
- 可选 reranker：默认不安装、不启用；如需重排再设置 `RERANKER_MODEL`

备选：

- FastEmbed embedding：`BAAI/bge-small-zh-v1.5`、`BAAI/bge-base-en-v1.5`、`BAAI/bge-large-en-v1.5`、`sentence-transformers/all-MiniLM-L6-v2`、`snowflake/snowflake-arctic-embed-l`、`intfloat/multilingual-e5-large`
- SentenceTransformers embedding：`BAAI/bge-base-zh-v1.5`、`BAAI/bge-m3`、`intfloat/e5-large-v2`、`Snowflake/snowflake-arctic-embed-l-v2.0`，需要 `EMBEDDING_BACKEND=sentence-transformers` 和 `INSTALL_MODEL_EXTRAS=true`
- Reranker：`cross-encoder/ms-marco-MiniLM-L6-v2`、`BAAI/bge-reranker-large`、`mixedbread-ai/mxbai-rerank-large-v1`，需要 `INSTALL_MODEL_EXTRAS=true`
- BM25：默认 `bm25-jieba`；备选 `Pyserini/Lucene`、`Elasticsearch/OpenSearch`、`rank-bm25`
- Query rewrite：默认关闭；可用 Ollama + `qwen3.5:9b` 在检索前把中文/口语问题扩展成英文案名、法条、缩写等检索词。

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

客服知识库快速测试：

```powershell
$env:STORE_BACKEND="memory"
$env:CORPUS_MANIFEST="/app/configs/support_sources.yml"
$env:RERANKER_MODEL="none"
docker compose up -d --force-recreate api
```

此模式会在 API 启动时把 `configs/support_sources.yml` 加载进内存 store，走同一条
`query rewrite -> metadata filter -> hybrid retrieval -> fusion -> optional rerank -> trace`
链路，但不写入 Qdrant。可以用这些问题测试：

- `退款多久到账`
- `订单发货后还能改地址吗`
- `两步验证备份码丢了怎么办`
- `续费后没用可以退款吗`
- `设备进水损坏保修吗`
- `什么时候会转人工客服`

Endpoints：

- `GET /health`
- `POST /retrieve`
- `POST /evidence`
- `POST /answer`
- `POST /query`

`/answer` 是可选 LLM 回答层：先复用 `/evidence` 的 RAG 检索链路，再把证据片段交给
OpenAI-compatible chat completions API 生成回答。前端可以选择 Ollama、DeepSeek、
OpenAI-compatible、OpenRouter、SiliconFlow、LM Studio 或自定义网关。API key 只随本次
`/answer` 请求发送给后端，不写入 trace；前端默认只保存 URL、model、开关等非密钥配置。
`/evidence` 和 `/answer` 在 `TRACE_ENABLED=true` 时都会写入 trace，默认只保留最近 30 个问题。

查看 trace：

```powershell
docker compose run --rm cli legal-rag trace list
docker compose run --rm cli legal-rag trace show
docker compose run --rm cli legal-rag trace show --trace-id <trace_id>
```

trace 关闭时，API 响应会返回 `trace_enabled=false`、`trace_persisted=false`，并且
`trace_id`/`trace_path` 为 `null`，避免展示一个实际不可查询的 trace。

## Qwen 3.5 查询改写

本项目可以在 `/evidence` 检索前增加一层本地模型 query rewrite，但仍不生成法律回答。
模型只输出结构化检索词，后端继续返回 evidence pack。

推荐这台 16GB RAM + RTX 4060 Laptop 8GB VRAM 机器先用 `qwen3.5:9b`。`qwen3.5:4b`
也可能跑得动，但模型包更大、延迟更高；query rewrite 这种短任务优先选 9B。

不需要在 Windows 宿主机安装 Ollama。使用 Docker profile 启动可选 Ollama 服务：

```powershell
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull qwen3.5:9b
```

启用 API 的 query rewrite：

```powershell
$env:QUERY_REWRITE_BACKEND="ollama"
$env:QUERY_REWRITE_MODEL="qwen3.5:9b"
$env:QUERY_REWRITE_TIMEOUT_SECONDS="60"
docker compose --profile llm up -d qdrant ollama api
```

如果 Ollama 没启动、模型没拉取或改写超时，`/evidence` 会自动退回原始 query 检索，
并在返回 JSON 的 `query_rewrite.error` 中说明原因。
`qwen3.5:9b` 冷启动加载可能需要 30 秒以上，所以默认给 query rewrite 留 60 秒超时。

## 测试

单元测试默认不依赖外部 LLM、Hugging Face 下载或 Qdrant 服务：

```powershell
docker compose run --rm test
```

默认 `api` 和 `cli` 服务使用轻量 FastEmbed 路径，不安装 `sentence-transformers/torch`。
默认 dense 模型为 `BAAI/bge-small-zh-v1.5`，适合中文知识库 baseline 和低内存本地运行。
默认 sparse backend 为 `SPARSE_BACKEND=bm25_jieba`，不会下载 FastEmbed sparse 模型；
只有显式设置 `SPARSE_BACKEND=qdrant_bm25` 时才会使用 `SPARSE_MODEL=Qdrant/bm25`。
Qdrant 镜像约为 `271MB`。首次真实 ingest 会把 FastEmbed/Hugging Face 模型权重下载到
Docker volume `legal-rag_model_cache`。

如果要启用 cross-encoder reranker 或 SentenceTransformers embedding backend，再显式构建重型镜像：

```powershell
$env:INSTALL_MODEL_EXTRAS="true"
$env:RERANKER_MODEL="BAAI/bge-reranker-large"
docker compose build api
docker compose up -d qdrant api
```

注意：重型 extras 会安装 PyTorch；默认 PyPI 解析可能拉入 GPU/CUDA wheel，镜像可达到数 GB。

如果之前下载过旧 embedding 或 Ollama 模型，可以在 Docker Desktop 启动后只清理模型缓存卷：

```powershell
docker compose down
docker volume rm legal-rag_model_cache legal-rag_ollama_models
```

不要为了清理模型缓存直接执行 `docker compose down -v`，除非你也想删除 Qdrant 数据卷。

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
