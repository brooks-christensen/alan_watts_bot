# Alan Watts Bot

A retrieval-grounded Alan Watts chatbot that answers user questions with concise, reflective responses anchored in a curated lecture corpus.

This project is built around a simple principle:

> **Style should never outrun grounding.**

The system uses a curated Alan Watts corpus, embedding-based retrieval, and an OpenAI generation model to produce answers that feel lecture-like without pretending to recover hidden facts or invent unsupported claims.

---

## What this project is

This repository is an experiment in building an **interactive philosopher chatbot** that is:

- grounded in a real Alan Watts source corpus
- usable as a live web demo
- explicit about retrieval and traceability
- structured for iterative model improvement
- deployable as a small API-backed service

The current MVP supports:

- a static frontend demo embedded into a broader personal website
- a backend API for retrieval-grounded generation
- Cloudflare Turnstile verification before requests are accepted
- display of supporting excerpts alongside the answer
- a data pipeline for preparing dialogic fine-tuning examples
- supervised fine-tuning experiments for lecture-like response style
- side-by-side comparison tooling for baseline vs fine-tuned models

---

## Current status

### Production / MVP path

The current MVP is a **RAG-first** system.

At the time of writing, the active generation path is configured to use:

- `gpt-5.4-mini` for answer generation
- `text-embedding-3-small` for retrieval embeddings
- `FAISS` for vector search
- `top_k = 3` retrieved chunks
- `max_output_tokens = 400`
- `temperature = 0.2`

The current system prompt is intentionally conservative:

- answer from the retrieved excerpts first
- do not mention retrieval internals
- do not invent outside facts or biography
- keep the tone calm, lucid, and meaningful
- avoid theatrical imitation

### Fine-tuning path

A first supervised fine-tuning pass was completed on a **chunk-safe, best-per-chunk dataset** prepared from the Alan corpus.

That run produced a fine-tuned experimental model:

- `ft:gpt-4.1-mini-2025-04-14:personal:alan-watts-sft-v1:DOXSTfMY`

The fine-tuned model showed some gains in compactness and lecture-like phrasing, but in side-by-side comparisons the baseline production model was still generally better grounded and more restrained. For that reason, the fine-tuned model is currently treated as an **experimental branch**, not the default production generator.

---

## Design philosophy

This project went through several important design iterations.

### 1. Monologic text generation was not enough

A style-only Alan Watts generator was interesting, but it was not enough for a trustworthy system. The project direction shifted toward **retrieval-augmented generation** so the model could answer from source material rather than from style alone.

### 2. Retrieval had to be visible

A good answer is more convincing when the user can inspect the passages that supported it. The interface therefore includes:

- the user question
- the generated answer
- supporting excerpt cards
- lightweight metadata about the answer path

### 3. The frontend needed to feel conversational

The initial UI displayed only an answer block. A later iteration improved the UX by explicitly showing the **user question alongside the model answer**, which made the demo feel more like a real conversation and improved the mobile experience.

### 4. Fine-tuning needed to optimize style, not factual memory

The fine-tuning objective is **not** “memorize Alan Watts.”

It is:

- keep the answer grounded in retrieved source material
- improve cadence, flow, warmth, and lecture-like delivery
- preserve restraint when the excerpts are incomplete

### 5. Dataset quality mattered more than dataset size

The first SFT preparation pass kept all examples, but a later revision moved to a cleaner setup:

- chunk-safe splitting
- `80 / 10 / 10` train / validation / test
- one selected example per chunk (`best_per_chunk`)
- reduced redundancy before spending credits on fine-tuning

That turned out to be the better first-pass training strategy.

---

## High-level architecture

```text
User question
    ↓
Static frontend (index.html)
    ↓
Turnstile verification
    ↓
Backend API (app.py)
    ↓
Embed question
    ↓
Retrieve top-k source chunks from FAISS index
    ↓
Build grounded prompt from retrieved excerpts
    ↓
Call generation model
    ↓
Return answer + supporting excerpts
```

### Main components

- **Frontend**: static HTML/CSS/JS demo shell embedded into the public website
- **Backend**: ASGI app served with Uvicorn
- **Retrieval**: FAISS + `text-embedding-3-small`
- **Generation**: OpenAI chat/completions-style generation path
- **Verification / abuse control**: Cloudflare Turnstile + backend limits
- **Fine-tuning utilities**: dataset prep, job launch, and comparison tooling

---

## Repository layout

This is the working layout implied by the current runtime artifacts and Docker image:

```text
.
├── app.py
├── config/
│   └── config.yaml
├── data/
│   ├── indexes/
│   ├── processed/
│   └── fine_tuning/
├── src/
│   ├── prepare_openai_sft_dataset.py
│   ├── prepare_openai_sft_dataset_v2.py
│   ├── launch_openai_sft_job.py
│   └── compare_openai_models.py
├── index.html
├── Dockerfile
└── requirements*.txt
```

Your local repository may contain more files than are shown here. This README focuses on the portions that were actively used in the current MVP and fine-tuning loop.

---

## Configuration

The current config is organized into the following blocks:

- `dialogic_convert`
- `validate_dialogic`
- `build_index`
- `retrieve`
- `generate`
- `api`

### Important current settings

```yaml
build_index:
  embedding_model: text-embedding-3-small

retrieve:
  top_k: 3

generate:
  model: gpt-5.4-mini
  temperature: 0.2
  max_output_tokens: 400
  top_k: 3
  max_context_chars_per_chunk: 2500

api:
  max_query_chars: 600
  rate_limit_requests: 12
  rate_limit_window_seconds: 300
  require_turnstile: true
```

### CORS / allowed origins

The API is currently configured to allow:

- `https://brookschristensen.com`
- `https://www.brookschristensen.com`
- `http://localhost:3000`
- `http://127.0.0.1:3000`

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/brooks-christensen/alan_watts_bot.git
cd alan_watts_bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If you are working from the full local repo, install the runtime dependencies you actually use.

Example:

```bash
pip install -r requirements.txt
```

If you are deploying from Docker, the image uses `requirements-api.txt` for the runtime container layer.

---

## Required environment variables

At minimum, the backend needs an OpenAI API key.

Depending on your deployment path, you may also need your Turnstile secret and any additional app-specific runtime secrets.

Typical example:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export TURNSTILE_SECRET_KEY="your_turnstile_secret"
```

---

## Running locally

### Option A: Run the API directly

The Dockerfile shows the production entrypoint as:

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT}
```

A typical local run looks like:

```bash
uvicorn app:app --host 127.0.0.1 --port 8080 --reload
```

### Option B: Run the container

```bash
docker build -t alan-watts-bot .

docker run --rm \
  -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e TURNSTILE_SECRET_KEY="$TURNSTILE_SECRET_KEY" \
  alan-watts-bot
```

### Frontend

The public demo frontend is static HTML. In the website integration flow, `index.html` is deployed alongside the broader website and calls the backend API for answers.

---

## Data and retrieval pipeline

The chatbot depends on a processed RAG corpus and a FAISS index.

### Corpus inputs

This project has used:

- cleaned Alan Watts lecture paragraphs
- RAG chunk JSONL exports
- generated dialogic Q/A pairs for SFT experiments

### Index build settings

The current retrieval path uses:

- `text-embedding-3-small`
- FAISS index output under `data/indexes`
- `top_k = 3`

You may already have local scripts for chunking / index creation beyond what is documented here. This README focuses on the verified commands and artifacts used in the fine-tuning cycle.

---

## Fine-tuning workflow

This repository now includes a practical OpenAI-hosted SFT pipeline.

### Step 1: Prepare the fine-tuning dataset

The recommended dataset-prep script is the chunk-safe v2 version.

```bash
python src/prepare_openai_sft_dataset_v2.py \
  --input data/processed/dialogic_dataset_enriched_full.jsonl \
  --messages-input data/processed/dialogic_training_pairs_full.jsonl \
  --output-dir data/fine_tuning/openai_sft
```

This version:

- uses chunk-safe splitting
- defaults to `80 / 10 / 10` train / validation / test
- selects the best example per chunk
- produces train / validation / test artifacts plus lineage and reports

### Step 2: Inspect the dataset report

Relevant outputs include:

- `openai_sft_train.jsonl`
- `openai_sft_validation.jsonl`
- `openai_sft_test_chat.jsonl`
- `openai_sft_test_eval.jsonl`
- `openai_sft_report.json`
- `openai_sft_report.md`

### Step 3: Launch a hosted fine-tuning job

```bash
python src/launch_openai_sft_job.py create \
  --train-file data/fine_tuning/openai_sft/openai_sft_train.jsonl \
  --validation-file data/fine_tuning/openai_sft/openai_sft_validation.jsonl \
  --run-dir data/fine_tuning/openai_sft_runs \
  --suffix alan-watts-sft-v1 \
  --metadata project=alan_watts \
  --metadata stage=sft_v1 \
  --metadata base_model=gpt-4.1-mini-2025-04-14 \
  --wait \
  --write-model-id-to data/fine_tuning/openai_sft_runs/latest_fine_tuned_model.txt
```

### Step 4: Compare baseline vs fine-tuned model

#### First-pass comparison

```bash
python src/compare_openai_models.py \
  --test-file data/fine_tuning/openai_sft/openai_sft_test_chat.jsonl \
  --baseline-model gpt-5.4-mini \
  --candidate-model ft:gpt-4.1-mini-2025-04-14:personal:alan-watts-sft-v1:DOXSTfMY \
  --output-dir data/fine_tuning/evals/alan_watts_sft_v1_compare \
  --max-cases 20 \
  --temperature 0.7 \
  --max-output-tokens 500
```

#### Production-like comparison

```bash
python src/compare_openai_models.py \
  --test-file data/fine_tuning/openai_sft/openai_sft_test_chat.jsonl \
  --baseline-model gpt-5.4-mini \
  --candidate-model ft:gpt-4.1-mini-2025-04-14:personal:alan-watts-sft-v1:DOXSTfMY \
  --output-dir data/fine_tuning/evals/alan_watts_sft_v1_compare_prodlike \
  --max-cases 20 \
  --temperature 0.2 \
  --max-output-tokens 400
```

### Current conclusion from the first SFT pass

The fine-tuned model improved some stylistic qualities, especially compactness and lecture-like phrasing, but the baseline `gpt-5.4-mini` remained more consistently grounded and restrained. For that reason:

- **baseline `gpt-5.4-mini` remains the default production generator**
- **the fine-tuned GPT-4.1 mini model remains experimental**

---

## Dataset summary from the v2 SFT pass

The current SFT dataset preparation selected:

- **374 selected records**
- **299 train**
- **38 validation**
- **37 test**
- **0 chunk overlap** across train / validation / test

This dataset was derived from:

- **1122 input / kept records**
- **374 unique chunks**
- one chosen example per chunk (`best_per_chunk`)

Top recurring themes included:

- ego
- self
- meditation
- nonduality
- zen
- identity
- spontaneity
- present moment
- interdependence
- play
- death
- nature
- opposites
- control
- unity

---

## UI notes

The current demo UI aims to be visually clean and legible to a first-time visitor.

Key UI choices:

- the user’s question is displayed explicitly in the response flow
- the answer is shown in a separate card-like area
- supporting excerpts are shown alongside the answer
- request verification is required before generation
- the language is intentionally calm and minimal rather than chatty or theatrical

This is meant to feel more like an **interactive philosophical lecture demo** than a generic chatbot widget.

---

## Safety and restraint goals

This project is deliberately conservative in a few places.

### The system should not:

- invent biography not present in the excerpts
- cite “Alan Watts says…” as a theatrical roleplay gimmick
- mention internal retrieval mechanics in the answer itself
- bluff when the context is thin
- optimize style at the expense of grounding

### The system should:

- stay modest when source material is incomplete
- answer in a reflective, lucid, lecture-like cadence
- synthesize multiple supported angles naturally
- remain readable for non-technical users

---

## Known limitations

- The current MVP is **text-first**.
- The frontend is static and intentionally lightweight.
- Retrieval quality is bounded by corpus cleaning and chunk quality.
- Fine-tuning improved style somewhat, but has not yet surpassed the stronger baseline model for production use.
- The system is not yet a real-time speech-to-speech conversational experience.
- Voice cloning / TTS styling is a future layer, not part of the current deployed MVP.

---

## Recommended next steps

### Near term

- keep `gpt-5.4-mini` as production default
- build a DPO preference dataset from baseline-vs-candidate comparisons
- improve preference for grounded, restrained, excerpt-faithful answers
- continue refining retrieval quality before spending heavily on larger tuning runs

### Medium term

- evaluate a fine-tuned `gpt-4.1` run rather than `gpt-4.1-mini`
- add a shadow-eval path using the full live retrieval stack
- expand automated evaluation around grounding / restraint / style

### Longer term

- add a speech input layer
- add a clearly disclosed synthetic lecture-style voice layer
- support a richer multi-turn conversational UX without sacrificing grounding

---

## Deployment notes

The runtime container is designed to be small and simple:

- `python:3.11-slim`
- non-root app user
- exposed on port `8080`
- ASGI app served through `uvicorn`

This matches the project’s broader design goal of keeping the system **lean, inspectable, and easy to reason about**.

---

## Credits and intent

This project is an homage to a philosophical voice and style that helped many people think more deeply about identity, suffering, play, death, and the nature of self.

The intent is **not** to produce fake historical artifacts or to present generated text as genuine Alan Watts transcripts. The goal is to build a reflective, clearly synthetic, retrieval-grounded conversational system inspired by his lecture style and philosophical cadence.

---

## License Note

I want this work to be useful. The code in this repository is released under the MIT License, including commercial use, modification, and redistribution.

That said, the software is provided “as is,” without warranty of any kind, and I assume no liability for any problems, damages, or claims arising from its use.

If you reuse corpus materials or source-derived data, please make sure you have the right to do so.

---

## Contact

**Brooks Christensen**

`brookschristensen.com`

`hello@brookschristensen.com`

