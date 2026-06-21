# Multi-Modal Damage-Claim Evidence Review

Verifies damage claims (car / laptop / package) from submitted images, a claim
conversation, and user history. For each row in `dataset/claims.csv` it produces
one row in `output.csv` matching the 14-column schema in `problem_statement.md`.

## Design: three layers

1. **Preprocessing** (`claimreview/preprocessing/`) — pure Python, no model calls.
   Loads the CSVs, joins `user_history.csv`, attaches the relevant
   `evidence_requirements.csv` rows, and downscales + base64-encodes images.
2. **Provider** (`claimreview/provider/`) — one structured VLM call per claim
   behind a `VLMProvider` interface, backed by **LiteLLM**. The model returns a
   factual **observation** (what is in each image), not a final ruling. The active
   model is chosen in `config.yaml` (`active_model`), so Anthropic / OpenAI /
   Gemini can be swapped without touching agent code — which makes the required
   "compare two model configurations" evaluation nearly free. Responses are
   content-hash cached to disk to avoid re-billing identical inputs.
3. **Rules** (`claimreview/rules/`) — deterministic decision logic that maps the
   observation to the 14 output columns. Images are the source of truth; user
   history and authenticity flags only add `risk_flags` (and may set
   `manual_review_required`) but never flip a supported decision;
   `evidence_requirements` gate `evidence_standard_met`; low confidence →
   `not_enough_information`. In-claim / in-image text is treated as untrusted data
   (prompt-injection safe, surfaced as `text_instruction_present`).

## Layout

```text
code/
├── main.py                  # entry point: preprocess / provider-dryrun / smoke / run / validate
├── config.yaml              # provider-agnostic config; secrets via env vars only
├── claimreview/
│   ├── enums.py             # allowed output values (verbatim from problem statement)
│   ├── schema.py            # typed input/prepared records (pydantic)
│   ├── observation.py       # structured VLM observation schema (pydantic)
│   ├── prompts.py           # versioned system/user prompts (v1, v2)
│   ├── config.py            # config loader + path resolution
│   ├── preprocessing/       # CSV loaders, image encode, join pipeline
│   ├── provider/            # VLMProvider interface, LiteLLM impl, messages, disk cache, batch runner
│   └── rules/               # decision engine + output.csv writer
├── tests/
│   └── test_rules.py        # rule-layer unit tests (zero API spend, synthetic observations)
└── evaluation/
    ├── main.py              # run configs on sample_claims.csv, score vs labels, write report
    └── evaluation_report.md # accuracy, confusion, ops analysis, model de-risking, final choice
```

## Setup

Uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

API keys are read from environment variables only — never hardcoded, never
committed. Put them in a `.env` file at the repo root (gitignored):

```bash
ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / GEMINI_API_KEY
```

## Usage

```bash
# Summarize preprocessing for the labeled sample set or the test set (no API calls)
uv run code/main.py preprocess --set test

# Offline check of the model-call wiring + observation schema (no API key needed)
uv run code/main.py provider-dryrun --set sample

# Live end-to-end sanity check on the first N claims (real API calls)
uv run code/main.py smoke --set sample --n 2

# Produce predictions for all rows in dataset/claims.csv -> output.csv
uv run code/main.py run --set test

# Validate the rule layer against labeled sample rows using cached observations only (zero spend)
uv run code/main.py validate

# Evaluation: run/compare configurations on the labeled sample and (re)write evaluation_report.md
uv run code/evaluation/main.py
```

> Tip: in a restricted sandbox where `uv run` can't reach its global cache, run
> the same commands with the project interpreter directly, e.g.
> `.venv/bin/python code/main.py run --set test`.

## Configuration

Edit `code/config.yaml`:

- `active_model` — which entry in `models:` to use (each has `provider`, `model`,
  `api_key_env`, and per-million-token pricing for accurate cost accounting).
- `prompt_version` — system/user prompt version (`v2` is current).
- `image.max_long_side` / `image.jpeg_quality` — downscaling (main cost lever).
- `runtime.*` — temperature, concurrency, retries, timeout for the model layer.
- `cache.*` — content-hash caching to avoid re-billing identical inputs.

## Final configuration

`claude-haiku@v2` is the locked production configuration. It won the labeled
sample comparison (`claim_status` 85% vs sonnet 80%) at ~1/3 the cost, and a full
44-row `claude-sonnet@v2` cross-check on the test set confirmed no systematic
difference (equal contradiction rates, 8/8 agreement on `wrong_object` cases).
The full rationale is in `evaluation/evaluation_report.md`. The committed
`output.csv` (repo root) is the `claude-haiku@v2` run over `dataset/claims.csv`.
