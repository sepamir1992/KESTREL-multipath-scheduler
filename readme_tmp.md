# TicketAssist RAG Evaluator (DeepEval)

This component benchmarks the end-to-end quality of the **TicketAssist Search Engine** using **DeepEval**. It supports:

- **CLI evaluation** via `python rag_evaluator.py` (inside the `rag-evaluator` container)
- **Interactive evaluation UI** via **Streamlit** (the `ui.py` you provided)

The evaluator measures:
- **Correctness** (GEval)
- **Answer Relevancy**
- **Contextual Recall**
- **Contextual Precision**
- **Contextual Relevancy**

## What is being tested

**System Under Test (SUT): TicketAssist Search Engine**
- Search: `POST {SEARCH_ENGINE_URL}/v1/search`
- State snapshot: `GET {SEARCH_ENGINE_URL}/v1/state`
- Health: `GET {SEARCH_ENGINE_URL}/v1/health`

**Judge LLM (DeepEval judge)**
- ELI Gateway: `POST {ELI_API_URL}/api/v1/llm/chat` (Bearer token)

**Ground Truth (GT)**
- Excel file with required columns: `question`, `answer`

---

## Repository layout (typical)

```text
rag-evaluator/
  rag_evaluator.py
  ui.py
  dataset/
    groundtruth_filtered.xlsx
  requirements.txt
  Dockerfile
  README.md
```

---

## How it works

For each GT row:
1. The evaluator calls the Search Engine:
   - `POST /v1/search` with `query_mode="chat"` so generation is enabled
2. It extracts:
   - `answer` → becomes `LLMTestCase.actual_output`
   - retrieval context → flattened from Search Engine’s grouped UI schema by reading `doc_text` from:
     - `JIRA`
     - `TREQs`
     - `CPI`
3. It runs DeepEval metrics with a judge implemented via ELI Gateway `/api/v1/llm/chat`
4. It returns:
   - `df_res`: per-question detailed metrics
   - `df_summary`: aggregated averages and pass rates

### Search Engine payload (base)

The evaluator builds a base payload and then merges optional overrides:

```json
{
  "query": "<question>",
  "query_mode": "chat",
  "search_type": "hybrid",
  "limit": 5,
  "product": {"JIRA": {"PCELS": {}}},
  "inference_engine": "ericai|eli",
  "retriever_type": "weaviate|eli"
}
```

> Note: `product` is currently pinned to `{"JIRA":{"PCELS":{}}}` in `call_search_engine_rag()`. Expand this when GT coverage grows beyond PCELS.

---

## Prerequisites

1. **Search Engine reachable from the rag-evaluator container**
   - Recommended in Docker Compose: `http://search-engine:8080` (your UI default)

2. **ELI Gateway API key**
   - Must be available either:
     - through the UI sidebar (`ELI API Key`) **or**
     - via container env var `ELI_API_KEY`

3. **Ground truth Excel**
   - Contains `question`, `answer`
   - Default: `dataset/groundtruth_filtered.xlsx`

---

## Running evaluation inside the container (CLI)

### 1) Enter the container

```bash
docker exec -it rag-evaluator bash
```

### 2) Set required environment (recommended)

```bash
export SEARCH_ENGINE_URL="http://search-engine:8080"
export ELI_API_URL="https://gateway.eli.gaia.gic.ericsson.se"
export ELI_API_KEY="***"
```

### 3) Run evaluation

**Run with defaults (uses all rows if `--max-rows 0`, otherwise see sampling flags):**
```bash
python rag_evaluator.py --eli-api-key "$ELI_API_KEY"
```

**Evaluate N rows (first N):**
```bash
python rag_evaluator.py --max-rows 50 --eli-api-key "$ELI_API_KEY"
```

**Random sample N rows:**
```bash
python rag_evaluator.py --max-rows 50 --sample --seed 42 --eli-api-key "$ELI_API_KEY"
```

**Print Search Engine snapshot (`/v1/state` + `/v1/health`) to logs:**
```bash
python rag_evaluator.py --print-snapshot --eli-api-key "$ELI_API_KEY"
```

---

## CLI parameters (rag_evaluator.py)

### Core choices

- `--inference-engine` (`ericai` default)  
  Choices: `eli`, `ericai`

- `--retriever-type` (`weaviate` default)  
  Choices: `eli`, `weaviate`

- `--gt-path` (default `dataset/groundtruth_filtered.xlsx`)

- `--search-engine-url` (default from env `SEARCH_ENGINE_URL` else `http://localhost:8080`)

### Ground truth selection

- `--max-rows` (default `0`)
  - `0` means evaluate **all** GT rows

- `--sample` (default `False`)
  - Only applies if `--max-rows > 0`

- `--seed` (default `42`)

### Overrides forwarded into `/v1/search` payload

These are only sent if “meaningful”:
- numeric > 0
- string non-empty
- boolean parsed from `true/false` string

| CLI flag | `/v1/search` key | Default | Notes |
|---|---|---:|---|
| `--override-top-k` | `top_k` | 0 | Retrieval size |
| `--override-pool-k` | `pool_k` | 0 | Candidate pool size |
| `--override-num-rag-evidences` | `num_rag_evidences` | 0 | Evidences used downstream |
| `--override-rerank` | `rerank` | "" | `true` / `false` / empty |
| `--override-reranker-model` | `reranker_model` | "" | Model name |
| `--override-eli-llm-model` | `eli_llm_model` | "" | Generation model |
| `--override-ericai-llm-model` | `ericai_llm_model` | "" | Generation model |
| `--override-hybrid-alpha` | `hybrid_alpha` | 0 | Hybrid weight |
| `--override-rag-method` | `rag_method` | "" | e.g., `v2` |
| `--override-stage2-threshold` | `stage2_score_threshold` | 0 | Stage-2 threshold |

### Judge / ELI gateway

- `--eli-api-url` (default `https://gateway.eli.gaia.gic.ericsson.se`)
- `--eli-api-key` (default from env `ELI_API_KEY` else empty → required)

### Rate limiting / stability

- `--sleep-between-cases` (default `1.0`)
- `--judge-max-retry` (default `8`)
- `--judge-backoff-base` (default `2.0`)
- `--judge-model` (default `mistral`)

### Advanced

- `--deepeval-async` (default `False`)
  - increases concurrency; higher chance of HTTP 429 from judge

- `--no-dotenv` (default `False`)
  - skip `.env` loading

### CLI-only

- `--plot` show matplotlib plot (not used in Streamlit)
- `--print-snapshot` include `/v1/state` and `/v1/health` in logs

---

## Running and using the Streamlit UI (ui.py)

### 1) Start the UI

Inside the container:

```bash
streamlit run ui.py --server.address 0.0.0.0 --server.port 8501
```

Expose port in Docker/Compose (example):
- Map `8501:8501`

Then open:
- `http://localhost:8501`

---

## UI walkthrough (matches your code)

### A) Configuration (Sidebar)

#### Search Engine
- **Search Engine URL** (default): `http://search-engine:8080`

#### Core Choices
- **Inference Engine**: `ericai` or `eli`
- **Retriever**: `weaviate` or `eli`

#### Ground Truth
- **GT Path** (default): `dataset/groundtruth_filtered.xlsx`
- **Max Rows** (UI default): `50`  
  - Note: in UI, min is 1 (unlike CLI where 0 = all)
- **Random Sample** checkbox
- **Seed** (default): 42

#### Overrides (Optional)
Visible by default:
- **Reranker Model** (default): `eli-reranker-small-1`
- **Hybrid Alpha** slider (default): `0.5`

Under **More Overrides** expander:
- `top_k`, `pool_k`, `num_rag_evidences`
- `rerank` (`""`, `"true"`, `"false"`)
- `eli_llm_model`, `ericai_llm_model`
- `rag_method`, `stage2_score_threshold`

#### Judge / ELI
- **ELI API URL** default: `https://gateway.eli.gaia.gic.ericsson.se`
- **ELI API Key** (password input)
  - If empty, UI uses environment variable `ELI_API_KEY`

#### Rate Limiting
- Sleep Between Cases (default `1.0`)
- Judge Max Retry (default `8`)
- Judge Backoff Base (default `2.0`)
- Judge Model (default `mistral`)

#### Advanced
- DeepEval async_mode (higher 429 risk)
- Skip dotenv load
- Print Search Engine snapshot in logs (default `True`)

### B) Status panel (Main page)

The UI can show backend status without running evaluation:

- `/v1/state` shown in “Runtime Configuration”
- `/v1/health` shown in “Health”

Buttons:
- **Refresh Status**: forces fresh calls
- Otherwise, UI caches status in `st.session_state` to avoid repeated backend calls

### C) Run evaluation

Click **Run Evaluation**.

Under the hood, the UI:
1. Builds a `CustomArgs` object with attributes matching `rag_evaluator.main(args)` expectations
2. Captures evaluator stdout using `contextlib.redirect_stdout(...)`
3. Calls:
   ```python
   df_res, df_summary = main(args_obj)
   ```
4. Stores `df_res`, `df_summary`, and logs into Streamlit session state so results persist across reruns

### D) Results and downloads

Once evaluation completes, the UI shows two tabs:

1. **Average Scores**
   - bar chart of `df_summary["avg_score"]` by `metric`
   - summary table

2. **Detailed Data**
   - full `df_res` table
   - download button: **Detailed Results (CSV)**

### Log capture

All `print(...)` output from `rag_evaluator.main()` is shown under:
- **Execution Logs** expander

---

## How the UI passes config into the evaluator (exact mapping)

Your `ui.py` builds:

```python
args_obj.eli_api_key = (eli_api_key or os.getenv("ELI_API_KEY", "")).strip()
```

So:
- If sidebar key is non-empty → used
- Else fallback to environment `ELI_API_KEY`
- If still empty → UI blocks run with an error

Selected overrides map directly to evaluator attributes:

- `override_reranker_model` → `args_obj.override_reranker_model`
- `override_hybrid_alpha` → `args_obj.override_hybrid_alpha`
- plus all “More Overrides” fields

Finally, Streamlit disables evaluator plotting:
- `args_obj.plot = False` (UI renders plots itself)

---

## Troubleshooting

### Search Engine unreachable
From inside container, confirm:

```bash
curl -sS http://search-engine:8080/v1/health
```

If that fails:
- check Docker Compose network / service name
- confirm Search Engine is running and port is correct

### Judge rate-limited (HTTP 429)
Mitigate via UI or CLI:
- increase **Sleep Between Cases**
- keep **DeepEval async_mode** off
- increase retries

### Missing columns in GT
GT Excel must include:
- `question`
- `answer`

### UI “Max Rows” behavior vs CLI
- UI enforces min `1` and default `50`
- CLI supports `--max-rows 0` meaning “use all rows”

If you want UI parity with CLI:
- change `max_rows` input min to `0`, and treat `0` as “all”

---

## Recommended defaults (practical)

For stable runs:
- `Sleep Between Cases`: 1.0–2.0
- `DeepEval async_mode`: Off
- `Judge Max Retry`: 8–12
