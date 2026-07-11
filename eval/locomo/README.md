# Benchmarking atomir on LOCOMO (via mem0's memory-benchmarks)

`atomir_client.py` is a drop-in replacement for mem0's `Mem0Client`, so you can
run mem0's own LOCOMO harness against **atomir** and get apples-to-apples numbers
(same dataset, same grader, same models) versus mem0's published results.

## 1. Set up the harness

```bash
git clone https://github.com/mem0ai/memory-benchmarks.git
cd memory-benchmarks
pip install -r requirements.txt
pip install "atomir[qdrant]"          # atomir must be importable here
```

## 2. Drop in the atomir adapter (one file + one line)

```bash
cp /path/to/atomir/eval/locomo/atomir_client.py benchmarks/common/atomir_client.py
# swap the import in the runner:
sed -i '' \
  's/from benchmarks.common.mem0_client import Mem0Client, format_search_results/from benchmarks.common.atomir_client import Mem0Client, format_search_results/' \
  benchmarks/locomo/run.py
```

(atomir runs **in-process** — you do NOT need mem0's Docker server for the atomir run.)

## 3. Start Ollama (free, local models for both sides)

```bash
ollama pull llama3.1            # extraction (atomir) + answering/judging
ollama pull nomic-embed-text   # embeddings (atomir)
```

## 4. Run atomir on a subset first (~free, learn the harness)

```bash
# --- atomir uses Ollama for extraction + embeddings ---
export LLM_BACKEND=ollama MODEL=llama3.1
export EMBED_BACKEND=ollama EMBED_DIM=768
export STORE_BACKEND=qdrant

# --- the harness's answerer + judge also route to local Ollama (OpenAI-compatible) ---
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama          # dummy; Ollama ignores it

python -m benchmarks.locomo.run \
  --project-name atomir-subset \
  --conversations 0 --max-questions 5 \
  --answerer-model llama3.1 --judge-model llama3.1 \
  --max-workers 2
```

Full run: drop `--conversations`/`--max-questions` (all 10 convs, ~300 Q) — **hours** on local Ollama.

### Faster (small cost): use Groq/OpenAI for answerer+judge

```bash
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_API_KEY=$GROQ_API_KEY
# --answerer-model llama-3.3-70b-versatile --judge-model llama-3.3-70b-versatile
```
(Groq free tier is capped at 100K tokens/day — a full LOCOMO run needs the paid Dev tier or gpt-4o-mini.)

## 5. Run mem0 for comparison (same models)

Use the **original** repo (unpatched `mem0_client`) with mem0's Docker OSS server
(`configs/ollama.yaml` → `docker compose up -d`), same `--answerer-model` /
`--judge-model`, then compare the two result sets.

## Reading the results

- Report **per-category** accuracy (`--categories 1,2,3,4`) plus latency and tokens.
- **Known gap:** atomir has no temporal reasoning, so LOCOMO's temporal category
  will likely be its weakest — surface that honestly rather than hide it in an average.
- If atomir is competitive, submit to the leaderboard at
  [agentmemorybenchmark.ai](https://agentmemorybenchmark.ai/dataset/locomo).

## Status

The adapter interface is verified against the harness (async `add`/`search` +
`format_search_results`). The full run and any leaderboard submission are yours
to execute — they need hours of compute (or a small API budget) and your accounts.
