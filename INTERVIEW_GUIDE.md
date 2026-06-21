# INTERVIEW_GUIDE.md — Multi-Modal Evidence Review

A study + rehearsal document for the live technical interview. Every factual
claim cites `file:line`. Anything I assert as rationale that is **not** written
in the code or its comments is explicitly tagged **(inferred, not in code)**.

Architecture in one sentence: **pure-Python preprocessing → one structured VLM
"observation" call per claim (LiteLLM, provider-agnostic) → a deterministic rule
engine that maps observations to the 14 required output columns**
(`code/claimreview/rules/decide.py:1-15`, `code/README.md:7-20`).

> Note on the two `main.py` files: the repo-root `main.py:1-7` is just the `uv`
> starter stub (`print("Hello...")`) and is **not** used by the solution. The real
> entry point is `code/main.py` (`code/main.py:1-9`, contract in `AGENTS.md` §6.1).

---

## PART 1 — End-to-end execution trace

Command: `uv run code/main.py run --set test` → `cmd_run` (`code/main.py:227-264`).
I'll follow one car claim (`user_002`, `dataset/images/test/case_001/...`, which is
row 2 of the committed `output.csv`) from CSV to a written row.

### Step 0 — CLI dispatch and config
- `build_parser` registers the `run` subcommand defaulting to `--set test`
  (`code/main.py:348-350`); `main()` calls `args.func` (`code/main.py:360-363`).
- `cmd_run` loads config: `load_config(args.config)` reads `code/config.yaml` and
  validates it into a typed `Config` (`code/claimreview/config.py:91-96`).
  **Before:** YAML text. **After:** `Config` object with `active_model="claude-haiku"`,
  `prompt_version="v2"`, pricing, paths, runtime knobs (`code/config.yaml:6,62`,
  `code/claimreview/config.py:67-74`).
- `_load_env(cfg)` calls `load_dotenv(REPO_ROOT/.env)` so `ANTHROPIC_API_KEY`
  enters the environment (`code/main.py:216-224`). Key presence is checked; if
  missing, it aborts with exit 2 **before** any spend (`code/main.py:234-237`).

### Step 1 — CSV load
- `prepare_dataset("test", cfg)` resolves `dataset/claims.csv`
  (`code/claimreview/preprocessing/pipeline.py:68-69`) and calls `load_claims`.
- `load_claims` uses `csv.DictReader` with `newline=""`, so the multi-line,
  comma-bearing quoted `user_claim` cell is parsed as ONE field
  (`code/claimreview/preprocessing/loaders.py:15-30`, `:19-20`).
  **Before:** a raw CSV line. **After:** `ClaimInput(user_id="user_002",
  image_paths="images/test/case_001/img_1.jpg;...;img_3.jpg",
  user_claim="Customer: Morning...", claim_object="car")`
  (`code/claimreview/schema.py:25-31`).

### Step 2 — user-history join
- `load_user_history` builds a `user_id -> UserHistory` dict
  (`code/claimreview/preprocessing/loaders.py:33-52`). Numeric columns are coerced
  (empty → 0) by a validator (`code/claimreview/schema.py:54-66`).
- In `prepare_claims`, the join is `history_map.get(claim.user_id)` — a missing
  user yields `None`, never a crash (`code/claimreview/preprocessing/pipeline.py:49`).
  `UserHistory.flag_list` parses `history_flags` (";"-split, `"none"`→`[]`)
  (`code/claimreview/schema.py:68-72`).
  **After:** the claim now carries its history flags (e.g. `["user_history_risk"]`)
  or `None`.

### Step 3 — evidence-requirements attach
- `load_evidence_requirements` loads all rows (`...loaders.py:55-69`).
- `_requirements_for` keeps rows where `claim_object` is the object **or** `"all"`
  (`code/claimreview/preprocessing/pipeline.py:26-30`,
  `code/claimreview/schema.py:83-84`).
  **After:** `PreparedClaim.evidence_requirements` = the `car` + `all` rules only.
  > **Known limitation (Finding B):** these requirements are attached and put in
  > the prompt (`prompts.py:154-155`) but the rule engine never reads them — the
  > `evidence_standard_met` gate is computed from observation booleans, not the
  > requirement text (see Part 2, "evidence gate" and Part 4 Q11).

### Step 4 — image downscale / encode / hash
- For each path, `encode_image` runs (`code/claimreview/preprocessing/images.py:33-91`):
  - Resolves `dataset_dir / rel_path`; missing file → `EncodedImage(exists=False,
    error="file_not_found")` with no raise (`images.py:48-55`).
  - Opens with Pillow, applies `ImageOps.exif_transpose` (honors camera rotation),
    converts to RGB (`images.py:58-61`).
  - `_downscale` shrinks so the long side ≤ `max_long_side` (1024) using LANCZOS,
    never upscales (`images.py:22-30`, `code/config.yaml:45`).
  - Saves JPEG at quality 85 to an in-memory buffer (`images.py:63-65`,
    `code/config.yaml:46`).
  - Computes `content_hash = sha256(encoded_bytes)` and a
    `data:image/jpeg;base64,...` `data_url` (`images.py:78,89-90`).
  - Decode failures are captured as `error=...`, not raised (`images.py:66-75`).
  **Before:** a file path. **After:** `EncodedImage` with `data_url`, `content_hash`,
  scaled dims; `.usable` is True iff found + no error + has data_url
  (`code/claimreview/schema.py:103-106`).
- The assembled unit is a `PreparedClaim` (input + history + requirements + images)
  — "no further disk access required" (`code/claimreview/schema.py:109-128`).

### Step 5 — provider + batch runner
- `LiteLLMProvider(cfg)` builds `litellm_model = "anthropic/claude-haiku-4-5"` and a
  `DiskCache` at `.cache` (`code/claimreview/provider/litellm_provider.py:47-60`,
  `code/config.yaml:18-23,57-58`).
- `analyze_claims(provider, prepared, max_concurrency=4, _progress)` submits each
  claim to a `ThreadPoolExecutor`; results are stored **by original index** so order
  is preserved, and a worker exception becomes an error `ProviderResult` rather than
  aborting the batch (`code/claimreview/provider/runner.py:17-49`,
  `code/config.yaml:52`).

### Step 6 — message construction
- `build_messages` makes `[system, user]`: the system prompt for `v2`, then a user
  message whose `content` is a text block followed by one `image_url` block per
  **usable** image (`code/claimreview/provider/messages.py:19-35`).
- `build_user_text` lays out: claim object; the conversation fenced as untrusted
  data inside `"""`; the image IDs; the minimum-evidence requirement lines; and the
  allowed enum value lists for this object (`code/claimreview/prompts.py:130-161`).
  **After:** an OpenAI-style multimodal message list LiteLLM will translate to
  Anthropic's format (`messages.py:1-6`).

### Step 7 — cache check
- `analyze` computes `cache_key` = `sha256(provider, model, prompt_version,
  system_prompt, user_text, sorted(image content_hashes))`
  (`code/claimreview/provider/messages.py:38-58`).
- On a hit it returns the stored `ProviderResult` with `cached=True` and
  `cost_usd=0.0` — zero new spend (`litellm_provider.py:101-109`). On the locked
  run this was a miss for all 44 rows (different test images), so it called the API.

### Step 8 — VLM call
- `_call` invokes `litellm.completion(model, messages, temperature=0,
  timeout=90, api_key=...)` (`litellm_provider.py:66-78`, `code/config.yaml:51,54`).
  `temperature=0` is the determinism lever.
  **After:** a provider-normalized response object; text at
  `resp.choices[0].message.content` (`litellm_provider.py:127`).

### Step 9 — JSON parse / repair
- `extract_json` strips ```` ``` ```` fences, tries `json.loads`, and falls back to
  the outermost `{...}` slice (`litellm_provider.py:25-44`).
- If parsing fails, ONE in-call repair turn asks for "ONLY the JSON object", then
  re-parses; a second failure sets `last_err="json_parse_failed"` and the attempt
  loop continues (`litellm_provider.py:130-148`). Token/cost from **every** call
  (initial + repair + retries) are accumulated (`litellm_provider.py:117-126,142-143`).

### Step 10 — Pydantic validation → observation
- `VLMObservation.model_validate(data)` coerces the dict (`litellm_provider.py:150`).
  Field validators force every enum field to a legal value or `"unknown"`
  (`code/claimreview/observation.py:49-77,93-111`, via `coerce`/`coerce_list` in
  `code/claimreview/enums.py:166-181`). So even a malformed/hallucinated enum can't
  reach the rule layer.
  **After:** a `VLMObservation` with normalized claim fields + a list of
  `ImageObservation` (`observation.py:31-47,80-91`). The successful result, with
  accumulated tokens/cost, is cached via `DiskCache.set` (`litellm_provider.py:151-165`).
- If all retries fail, the result has `observation=None` and an `error`
  (`litellm_provider.py:172-180`).

### Step 11 — rule engine
- `cmd_run` zips prepared claims with results and calls `decide(p, r.observation)`
  for each (`code/main.py:252`). `decide` is the deterministic core
  (`code/claimreview/rules/decide.py:148-322`) — full branch walk in Part 2.
  **After:** an `OutputRow` (`code/claimreview/rules/output.py:21-39`). For `user_002`
  the model saw a scratch on the front bumper, so `claim_status="supported"`,
  `issue_type="scratch"`, `object_part="front_bumper"`, `supporting_image_ids=["img_2"]`
  (matches `output.csv` row 2).

### Step 12 — 14-column row → CSV write
- `write_output_csv` writes with `csv.DictWriter`, `fieldnames=OUTPUT_COLUMNS`
  (exact order), `quoting=csv.QUOTE_ALL` (`code/claimreview/rules/output.py:60-70`,
  `code/claimreview/enums.py:101-116`).
- `to_csv_dict` renders bools as `"true"/"false"` and list fields (`risk_flags`,
  `supporting_image_ids`) as `";"`-joined or `"none"`
  (`output.py:13-18,41-57`).
  **After:** `output.csv` at the repo root, 44 rows, written; `cmd_run` prints
  `calls/cached/errors/tokens/cost` (`code/main.py:254-263`).

---

## PART 2 — Every branch in the decision core (`decide.py`)

`decide(claim, observation)` (`code/claimreview/rules/decide.py:148-322`). The order
of branches IS the precedence. Across all branches, output enum safety is enforced
at construction: `object_part` via `_safe_part` (`:114-116`), `issue_type` clamped to
`ISSUE_TYPES` (`:315`), `severity` from `max_severity` (always in-enum), and
`risk_flags` only ever drawn from the fixed `_FLAG_ORDER` set (`:32-46,100-103`).

### Setup (runs before any decision)
- Echo inputs; read history flags into `hist_flags` (`:149-152`).
- Build `risk` set from per-image observations: quality flags ∩ `_QUALITY`,
  authenticity flags ∩ `_AUTHENTICITY`, `embedded_text_present` → `text_instruction_present`;
  and `injection_text_in_claim` → `text_instruction_present` (`:180-188`). These are
  **flags only**; they do not select a status.
- `claimed` = ordered claimed parts minus `"unknown"` (`:49-55,177`).

### Branch 1 — Provider-failure safe NEI
- **Condition:** `observation is None or not observation.images` (`:155`).
- **Output:** `not_enough_information`, `evidence_standard_met=false`,
  `valid_image=false`, `risk_flags` always include `manual_review_required` (+
  `user_history_risk` if present, + `damage_not_visible` if no usable image)
  (`:156-173`).
- **Invariant:** "low confidence / can't analyze → NEI" and route to a human; never
  guess (problem_statement.md:120-123). It cannot be `supported`/`contradicted`
  because there is no visual evidence at all.

### `valid_image` computation (all non-failure paths)
- `valid_image = has_usable_image AND ∃ image with no authenticity flag and not
  cropped_or_obstructed` (`:190-194`). It is a usability signal, NOT a decision input.

### Branch 2 — Assessability gate → NEI
- **Condition:** NOT `assessable`, where `assessable = ∃ image with
  claimed_part_visible OR damage_present OR not matches_claim_object` (`:196-200,202`).
- **Output:** `not_enough_information`, `evidence_standard_met=false`,
  `risk_flags += damage_not_visible` (+ history flags) (`:202-224`).
- **Invariant:** if nothing about the claim can be seen, the evidence standard is not
  met → NEI (problem_statement.md:115,120).
- > **Known limitation (Finding B):** this is the only "evidence standard" logic; it
  > is computed from observation booleans, not from the attached
  > `minimum_image_evidence` text. Honest framing in Part 4 Q11.

### Support test (feeds Branches 3–4)
- `support_images = images where _damage_on_claimed_part(im, claimed, claimed_issue)`
  (`:227-229`). `_damage_on_claimed_part` (`:84-97`) requires:
  `matches_claim_object AND damage_present` (`:87`), **issue-type compatibility**
  via `_issue_compatible` (`:89`), AND damage on a named claimed part OR on the
  **visible** claimed part where the model returned `observed_object_part="unknown"`
  (`:91-97`).
  - `_issue_compatible` (`:70-81`): lenient — `unknown`/`none`/`""` on either side
    is compatible; else equal or same `_ISSUE_FAMILIES` group (dent/scratch,
    crack/glass_shatter, broken_part/missing_part, torn/crushed packaging,
    water_damage/stain) (`:61-67`). This is the **Finding-A fix**: the old second
    clause was a tautology that dropped the "damage on visible claimed part, part
    unnamed" case into a false contradiction (now test #9, `tests/test_rules.py:220-239`).
- `exaggeration` = there is support, the customer stated ≥ high, but observed
  severity on the claimed part is none/low (`:231-237`). This is how a clear visual
  under-match of an over-stated claim becomes a contradiction.

### Branch 3 — Supported
- **Condition:** `support_images and not exaggeration` (`:244`).
- **Output:** `claim_status="supported"`, `evidence_standard_met=true`; decisive
  image = strongest by severity (`_strongest`, `:325-328`); `issue_type` = claimed
  issue (else observed) (`:248-252`); `object_part` = observed part, falling back to
  the claimed primary if observed is `unknown` (`:253-255`); `supporting_image_ids` =
  decisive image IDs (`:300,319`).
- **Invariant:** only direct visual confirmation supports (problem_statement.md:13,24).

### Branch 4 — `_contradict_or_nei` (everything else)
`_contradict_or_nei` (`:331-381`), evaluated in this order:
1. **Exaggeration → contradicted** (`:343-347`): `claim_mismatch` added; decisive =
   strongest support image; issue/part from that image. (Over-stated severity vs
   visible reality.)
2. **wrong_object → contradicted** (`:349-351,361-363`): images where
   `not matches_claim_object and object_in_image` is non-empty; emits
   `wrong_object` + `claim_mismatch`, `object_part="unknown"`, `issue_type="unknown"`.
   (The photo isn't even the claimed object.)
3. **part_visible_no_damage → contradicted** (`:352-354,365-372`): claimed part is
   clearly visible with `damage_present=false`; emits `damage_not_visible`,
   `issue_type="none"`. (We can see the part and there's no damage.)
4. **damage_other_part → contradicted** (`:356-359,374-379`): damage on a part NOT in
   `claimed`; emits `claim_mismatch` + `wrong_object_part`. (Real damage, wrong place.)
5. **Fallthrough → NEI** (`:381`): usable images, but none decisive. Back in `decide`,
   this adds `damage_not_visible` + history flags and returns an NEI row with
   `evidence_standard_met=true` (`:261-275`).

### Severity assignment
- If `issue_type == "none"` → `severity="none"` (`:278-279`).
- Else `severity = max_severity(observed severities of decisive images)`, which
  returns `"unknown"` if none are known (`:280-283`, `enums.py:158-163`). **Finding-E
  fix:** the old `stated_severity` fallback (customer's adjectives) was removed —
  severity is now image-observed only.
- > **Honest note (Nit E1):** a `supported` row can now read `severity=unknown` if the
  > model reported damage without a severity. Accepted tradeoff to keep "images are
  > the source of truth".

### How `risk_flags` accumulate
- Quality/authenticity/injection flags from observations (`:180-188`); contradiction
  cues from `_contradict_or_nei`'s `extra` set (`:260`); history flags via
  `_add_history` (`:286,384-388`). `_order_flags` renders them in a stable canonical
  order (`:100-103`). The final list is `none` only if empty (`output.py:17-18`).

### How `manual_review_required` triggers
- Added when ANY of: `user_history_risk` present, history carried
  `manual_review_required`, an authenticity flag is on the decisive image,
  `wrong_object` present, or `possible_manipulation` present (`:288-297`).
- > **Honest note (Opus-4.7 warning):** the NEI branches add history-driven
  > `manual_review_required` but NOT image-authenticity-driven one; a manipulated-but-
  > indeterminate claim can land in NEI without manual review. Not yet fixed.

### Injection / non_original_image / valid_image=false handling
- Injection: in-claim or in-image instruction text only ever sets
  `text_instruction_present` (`:185-188`); the rule layer never reads `user_claim`
  text, only structured observations. So "approve this / ignore instructions" cannot
  change the status (proved by `tests/test_rules.py:196-211`).
- `non_original_image` / `possible_manipulation`: added to `risk` (`:184`), can
  trigger `manual_review_required` (`:289-296`), and force `valid_image=false`
  (`:190-194`) — but they never appear in any status-selecting condition.

### Why history/authenticity can NEVER flip supported→contradicted
- The status is chosen ONLY by: provider-failure (`:155`), assessability (`:202`),
  `support_images`/`exaggeration` (`:244`), and the four visual branches of
  `_contradict_or_nei` (`:343-381`). **None** of those conditions reference
  `hist_flags` or authenticity flags. History and authenticity enter the function
  exclusively through the additive `risk` set and `_add_history` (`:156-159,184,
  286,289-296,384-388`). This is the structural guarantee behind
  problem_statement.md:13. Tests #2/#3 confirm a contradicted/supported decision with
  `user_history_risk`/`non_original_image` present still hinges on the image
  (`tests/test_rules.py:96-132`).

---

## PART 3 — Design & library decisions, with alternatives

### 3.1 Three-layer "observe → decide" split
- **What:** the model returns factual observations (`observation.py:1-10`); a
  deterministic engine makes the ruling (`decide.py:1-15`).
- **Alternatives:** ask the LLM to output the 14 columns directly (one prompt does
  everything).
- **Why this:** reproducibility + debuggability + you can change decision logic
  without re-calling the model (stated: `observation.py:8-9`). It also makes the
  invariants *enforceable in code* rather than *hoped for in a prompt*.
- **Tradeoff:** more code and a hand-tuned rule layer that must track the label
  semantics (`decide.py:13-14`); the rules can lag real-world variety (the Finding-A
  bug lived here).

### 3.2 LiteLLM (provider-agnostic gateway)
- **What:** all model calls go through `litellm.completion`, model id built from
  config as `provider/model` (`litellm_provider.py:57,66-78`).
- **Alternatives:** native Anthropic/OpenAI SDKs; LangChain.
- **Why this:** one OpenAI-style message format works across Anthropic/OpenAI/Gemini
  (`messages.py:1-6`), so "compare two models" is a config change, not a code change
  (`config.yaml:1-32`, `code/README.md:12-16`). vs **native SDKs**: would need a
  separate code path and message shape per vendor. vs **LangChain**: much heavier
  abstraction (chains/agents/memory) we don't need for a single structured call —
  LiteLLM is a thin routing layer with built-in cost tables (`litellm_provider.py:90-93`).
  **(inferred, not in code:** the LangChain-vs-LiteLLM weighting is my reasoning;
  the code only commits to LiteLLM.)
- **Tradeoff:** a third-party dependency in the hot path and reliance on its model
  registry/pricing; we override pricing in config to stay authoritative
  (`config.yaml:15-16,22-23`, `config.py:30-37`).

### 3.3 Pydantic everywhere
- **What:** typed models for config, inputs, observation, result, and output rows
  (`config.py:21-74`, `schema.py:25-128`, `observation.py:31-117`, `base.py:13-31`,
  `output.py:21-39`).
- **Alternatives:** raw dict parsing + manual `if`-checks.
- **Why this:** the model's JSON is validated and **coerced to legal enums** at the
  boundary (`observation.py:49-77`, `enums.py:166-181`), which is the backbone of the
  "always schema-valid output" guarantee. Config typos fail fast at load
  (`config.py:91-96`).
- **Tradeoff:** silent coercion can mask a bad model field as `"unknown"` rather than
  surfacing it (acceptable here; we *want* a safe default).

### 3.4 `uv` for env/deps
- **What:** `uv sync` from `pyproject.toml`; non-packaged app (`code/README.md:37-43`,
  `pyproject.toml:1-15`).
- **Alternatives:** pip + venv + requirements.txt; Poetry; conda.
- **Why this (inferred, not in code):** fast, lockfile-based, reproducible resolves;
  `uv.lock` pins all 58 transitive deps. The only code-visible facts are the
  `pyproject.toml` deps and `requires-python>=3.11`.
- **Tradeoff:** judges must have `uv` (mitigated: the README documents a
  `.venv/bin/python` fallback, `code/README.md` usage tip).

### 3.5 Pillow downscaling
- **What:** long side ≤ 1024, LANCZOS, JPEG q85, EXIF transpose (`images.py:22-30,
  58-65`, `config.yaml:44-47`).
- **Alternatives:** send full-resolution images; use provider auto-resize.
- **Why this:** "image tokens dominate VLM spend, and most claim damage is assessable
  at ≤1024px" (`images.py:1-7`). This is the primary cost lever
  (`code/README.md:67-68`).
- **Tradeoff:** very fine damage (hairline scratch) could be lost at 1024px
  **(inferred, not in code)**.

### 3.6 Disk content-hash cache
- **What:** JSON file per `sha256` key including model + prompt_version + prompt text
  + image content hashes (`cache.py:1-53`, `messages.py:38-58`).
- **Alternatives:** no cache; in-memory cache; an external KV store.
- **Why this:** "keeps the model-calling layer from re-billing identical inputs
  across reruns and across model comparisons" (`cache.py:1-4`). Hashing the encoded
  bytes (not base64) keeps keys small (`messages.py:43-47`).
- **Tradeoff:** the key omits runtime params (e.g. temperature) and an observation
  schema version, so a future change there could serve stale hits (GPT-5.3 / Opus-4.7
  warning; Part 4 Q12). The Finding-C fix made concurrent writes safe
  (`cache.py:38-53`).

### 3.7 Bounded-concurrency runner
- **What:** `ThreadPoolExecutor(max_workers=4)`, ordered results, per-claim error
  isolation (`runner.py:1-49`, `config.yaml:52`).
- **Alternatives:** sequential loop; `asyncio`; unbounded threads.
- **Why this:** `completion` is blocking I/O, so threads give real network concurrency
  while a fixed worker count caps RPM/TPM pressure (`runner.py:1-6`).
- **Tradeoff:** the trailing `[r for r in results if r is not None]` could silently
  drop a slot and misalign `zip(prepared, results)` if a worker ever left a `None`
  (Opus warning; Part 4 Q10). Today `_work` catches all `Exception`, so it's full.

### 3.8 Structured-output-then-validate (+ JSON repair)
- **What:** prompt asks for a single JSON object (`prompts.py:87-89,93-121`); robust
  extraction + one repair turn + Pydantic validation (`litellm_provider.py:25-44,
  122-150`).
- **Alternatives:** provider "JSON mode"/tool-calling; regex scraping; free-text +
  post-hoc parsing.
- **Why this (inferred, not in code):** a provider-agnostic gateway can't assume every
  vendor supports the same structured-output API, so prompt-for-JSON + tolerant parse
  + schema coercion is the portable lowest common denominator. On total failure it
  degrades to a safe NEI row (`decide.py:155-173`) instead of crashing.
- **Tradeoff:** an extra repair call costs tokens; cross-vendor JSON mode might be
  stricter where available.

### 3.9 Prompt v1 → v2
- **What:** v2 adds an explicit anti-confirmation-bias instruction and a strict
  severity rubric (`prompts.py:45-90`).
- **Why this:** comment states it was "motivated by sample errors: the model
  over-confirmed claims (false 'supported') and defaulted severity to 'high'"
  (`prompts.py:45-47`); the report records the labeled gain (claim_status 80%→85%,
  severity-within-1 to 90%) (`config.yaml:60-61`).
- **Tradeoff:** prompts are versioned and part of the cache key, so changing them
  invalidates cached observations (`messages.py:49-58`) — intended, but it means a
  prompt tweak = a full re-spend.

---

## PART 4 — Interview Q&A (read aloud; each grounded in code)

**Q1. Walk me through your architecture and why the three layers.**
I split it into preprocessing, a VLM observation call, and a deterministic rule
engine (`decide.py:1-15`). The model only *observes* what's in each image; it never
decides the claim (`observation.py:1-10`). The rule layer turns those observations
into the 14 columns (`decide.py:148-322`). I did this so the problem's invariants —
images are truth, history only adds risk — are enforced in code I can test, not just
requested in a prompt. It also means I can fix decision logic without paying to
re-call the model.

**Q2. Why LiteLLM instead of the native Anthropic SDK or LangChain?**
LiteLLM lets me write one OpenAI-style message format and swap providers with a config
line (`messages.py:1-6`, `litellm_provider.py:57`, `config.yaml:1-32`). That made the
required two-model comparison essentially free. Native SDKs would mean a separate code
path and payload per vendor; LangChain brings chains/agents/memory machinery I don't
need for a single structured call. LiteLLM is a thin router with built-in cost
estimation (`litellm_provider.py:90-93`), and I override its pricing in config to keep
cost numbers authoritative (`config.py:30-37`).

**Q3. How do you guarantee the output is always schema-valid (14 columns, legal enums)?**
Two gates. At the model boundary, every enum field is coerced to a legal value or
`"unknown"` by Pydantic validators (`observation.py:49-77`, `enums.py:166-181`), so a
hallucinated value can't propagate. At the output boundary, I write with
`csv.DictWriter(fieldnames=OUTPUT_COLUMNS, quoting=QUOTE_ALL)` (`output.py:60-70`) and
clamp `issue_type`/`object_part`/`severity` at row construction
(`decide.py:114-116,253-255,315`). I verified the locked `output.csv` had 44 rows,
exact ordered header, and zero enum violations.

**Q4. How do you know your predictions are good when the test set has no labels?**
I don't have ground truth on test, so I rely on three things. First, the 20 labeled
sample rows, where the final config scores claim_status 17/20 (`evaluation_report.md`).
Second, a deterministic rule layer with unit tests that pin the hard paths
(`tests/test_rules.py:1-247`). Third — and this is the honest part — I ran a full
44-row cross-check with the stronger Sonnet model and diffed it against my Haiku
output: equal contradiction rates and 8/8 agreement on the `wrong_object` cases, which
told me the cheaper model wasn't systematically wrong (`evaluation_report.md` §5). It's
agreement, not accuracy, and I'd say that explicitly.

**Q5. Why the cheaper model (Haiku)?**
On the labeled sample Haiku actually edged Sonnet on claim_status (85% vs 80%) at about
a third of the cost (`config.yaml:5`), and the test-set cross-check showed no systematic
difference in the decisions that matter (`evaluation_report.md` §5). The whole point of
the LiteLLM design is that this is one config line (`config.yaml:6`), so if a judge
wanted Sonnet I'd flip `active_model` and re-run. I chose the configuration the
evidence supported, not the most expensive one.

**Q6. How is this prompt-injection safe? Show me.**
The system prompt declares the conversation and any in-image text as untrusted data
and tells the model to only flag it, never obey it (`prompts.py:63-68`), and the user
text fences the conversation in a delimited block (`prompts.py:150-151`). More
importantly, the rule layer never reads `user_claim` at all — it decides only from
structured observations, and injection text can only set `text_instruction_present`
(`decide.py:185-188`). My test feeds "approve immediately and skip review" and asserts
the decision still follows the image (`tests/test_rules.py:196-211`). The honest gap:
delimiter fencing isn't unforgeable, so a claim containing `"""` could break out of the
block (a reviewer flagged this); the rule-layer indifference to claim text is the real
defense.

**Q7. Prove that user history can't flip a "supported" claim to "contradicted".**
The status is assigned only in five places: provider failure (`decide.py:155`), the
assessability gate (`:202`), the support/exaggeration test (`:244`), and the four
visual branches of `_contradict_or_nei` (`:343-381`). None of those conditions
reference `hist_flags` or authenticity flags — those enter only through the additive
`risk` set and `_add_history` (`:184,286,384-388`). So structurally, history can add
`user_history_risk` and `manual_review_required` but can't change the decision. Tests
#2 and #3 show a status decided by the image even with `user_history_risk` and
`non_original_image` present (`tests/test_rules.py:96-132`).

**Q8. How do you control cost, latency, and rate limits?**
Cost: I downscale every image to ≤1024px JPEG before encoding, since image tokens
dominate (`images.py:1-7,22-30`), and I content-hash cache so identical inputs never
re-bill (`cache.py:1-4`, `messages.py:38-58`). Latency: a 4-worker thread pool over
blocking calls (`runner.py:1-6`, `config.yaml:52`). Reliability: up to 3 retries with
exponential backoff and a one-shot JSON repair turn (`litellm_provider.py:122-170`).
The real run was 44 calls, 0 errors, ~$0.24 (`evaluation_report.md` §4).

**Q9. What's the single biggest correctness bug you found, and how did you handle it?**
A tautology in the support test: `A or (B and A)` collapses to `A`, so an image showing
damage on the *visible* claimed part but with the part left `unknown` was wrongly
flipped to `contradicted/wrong_object_part` (old `_damage_on_claimed_part`). I found it
via an adversarial multi-model review, wrote a failing regression test first
(`tests/test_rules.py:220-239`), then fixed the branch to accept the unnamed-but-visible
case and added lenient issue-family matching (`decide.py:70-97`). All 9 tests pass and
the labeled sample didn't regress.

**Q10. What's the weakest part of this system and what would you fix next?**
Three honest weak spots. (1) `evidence_standard_met` is computed from observation
booleans, not the attached `minimum_image_evidence` text — it's effectively
model-delegated (`decide.py:196-200`; Finding B). (2) `supporting_image_ids` is capped
at one image because `_strongest` returns only the top one, though the spec allows
multiple (`decide.py:325-328`). (3) The runner's `None`-filter could misalign rows if a
worker ever returned `None` (`runner.py:49`). I'd fix the evidence gate first — make it
read the requirement rules deterministically — because it's the one that diverges from
the documented invariant.

**Q11. Your `evidence_standard_met` — does it actually use the evidence requirements file?**
No, and I'll be straight about it. I load and join the requirements and put them in the
prompt (`pipeline.py:51`, `prompts.py:154-155`), but the deterministic gate only checks
whether the claimed part is visible / damage is present / object matches
(`decide.py:196-200`). So the requirement satisfaction is effectively delegated to the
model's visibility judgment rather than enforced in rules. It's a known limitation; the
clean fix is a rule that maps the claim's issue family to its requirement row and checks
it explicitly before setting `evidence_standard_met=true`.

**Q12. What did Cursor (the AI) write versus what did you decide?**
The decisions were mine: the three-layer observe/decide architecture, choosing LiteLLM
for provider-agnosticism, the invariant that history never flips a decision, the
caching and cost strategy, the model choice backed by the cross-check, and the
prompt-engineering direction from v1 to v2. The AI accelerated the mechanical
implementation — typing out the Pydantic models, CSV plumbing, and boilerplate — and I
used adversarial multi-model reviews as a QA step, which is how the tautology bug
surfaced. Every fix went through a failing-test-first loop that I drove and validated
(`tests/test_rules.py`), and I can explain any line in this codebase, which is the point
of this document.

---

## Quick-reference: invariant → enforcing code

| Invariant (problem_statement.md) | Enforced at |
|---|---|
| Images are the source of truth (13,24) | status set only by visual branches `decide.py:155,202,244,343-381` |
| History adds risk, never overrides (13,90) | additive `risk`/`_add_history` only `decide.py:184,286,384-388` |
| Injection text is data, not command | `text_instruction_present` only; rules ignore `user_claim` `decide.py:185-188`, `prompts.py:63-68` |
| Low confidence → NEI (120) | `decide.py:155,202,381` |
| 14 columns, exact order, legal enums (96-140) | `enums.py:101-116`, `output.py:41-70`, coercion `observation.py:49-77` |
| Deterministic where possible (AGENTS.md §6.2) | `temperature=0` `litellm_provider.py:75`, rule layer is pure |
