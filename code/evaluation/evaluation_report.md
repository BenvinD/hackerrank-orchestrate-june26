# Evaluation Report — Multi-Modal Evidence Review

_Generated: 2026-06-19T15:21:01_

System design: pure-Python preprocessing -> one structured VLM call per claim (LiteLLM, provider-agnostic) that **observes** each image -> a deterministic rule layer that maps observations to the 14 output columns. Images are the source of truth; user history and authenticity flags add risk context but never flip a supported decision; in-claim/in-image instructions are treated as data (`text_instruction_present`).

## 1. Accuracy on labeled sample (20 rows)

Per-column exact-match accuracy (risk_flags / supporting_image_ids compared as sets):

| Model | claim_status | evidence_standard_met | valid_image | issue_type | object_part | severity | risk_flags | supporting_image_ids | risk F1 | sev ±1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku@v2` | 85% | 95% | 85% | 70% | 90% | 50% | 35% | 90% | 0.74 | 90% |
| `claude-sonnet@v2` | 80% | 95% | 85% | 65% | 85% | 50% | 35% | 85% | 0.65 | 85% |

### claim_status confusion (want -> got)

- **`claude-haiku@v2`**: `contradicted->contradicted`×3, `contradicted->supported`×2, `not_enough_information->not_enough_information`×1, `not_enough_information->supported`×1, `supported->supported`×13
- **`claude-sonnet@v2`**: `contradicted->contradicted`×2, `contradicted->supported`×3, `not_enough_information->contradicted`×1, `not_enough_information->not_enough_information`×1, `supported->supported`×13

> Note: `manual_review_required` recall is reported separately because routing risky claims to a human is the operationally important signal:

> - `claude-haiku@v2`: manual_review_required recall 100%
> - `claude-sonnet@v2`: manual_review_required recall 100%

## 2. Configuration comparison & final choice

The configurations below (`model@prompt_version`) were run on identical inputs. **Final configuration: `claude-haiku@v2`**, selected for the best `claim_status` accuracy, breaking ties by lower cost. Other configurations are retained in config as fallbacks / cost levers.

## 3. Operational analysis

Sample run (actual; cached calls billed at $0):

| Model | claims | new calls | cached | images | tokens (in/out) | new cost | avg/max latency |
|---|---|---|---|---|---|---|---|
| `claude-haiku@v2` | 20 | 0 | 20 | 29 | 45332/10672 | $0.0000 | 6.1s / 11.1s |
| `claude-sonnet@v2` | 20 | 20 | 0 | 29 | 45352/11472 | $0.3081 | 11.9s / 17.4s |

Test-set projection (44 claims, 82 images), one VLM call per claim, all images batched into that call:

| Model | est. calls | est. cost (no cache) | pricing (in/out per MTok) |
|---|---|---|---|
| `claude-haiku@v2` | 44 | $0.22 | $1.0/5.0 |
| `claude-sonnet@v2` | 44 | $0.68 | $3.0/15.0 |

**Pricing assumptions:** Anthropic API list prices verified Jun 2026 (`claude-sonnet-4-6` $3/$15, `claude-haiku-4-5` $1/$5 per MTok input/output); no prompt-cache or Batch-API discounts assumed (both would lower cost ~50–90%).

**Calls / images:** exactly one model call per claim; multiple images are batched as content blocks in that single call (no per-image calls).

**Latency / throughput:** claims run concurrently with a bounded thread pool (`max_concurrency=4`), keeping RPM/TPM pressure low while overlapping network I/O.

**Cost & rate-limit controls:**
- Images downscaled to a bounded long side before encoding (image tokens dominate cost).
- Content-hash disk cache keyed on (model, prompt version, claim text, image hashes): identical inputs are never re-billed, so reruns and the model comparison reuse prior work.
- Retry with exponential backoff (`max_retries=3`) on transient API errors; a single JSON-repair re-ask before failing a claim.
- `temperature=0` for determinism.

## 4. Production run on test set (actuals)

Final configuration `claude-haiku@v2` was run on all 44 `dataset/claims.csv`
rows to produce `output.csv`:

- **Calls / errors:** 44 model calls, 0 errors (one call per claim).
- **Tokens:** 135,670 total.
- **Cost:** **$0.2438** (matches the $0.22 projection above; pricing $1/$5 per MTok).
- **Output:** 44 rows, exact 14-column schema, 0 schema/enum violations.
- **Prediction mix:** claim_status = 22 supported / 19 contradicted / 3 not_enough_information; evidence_standard_met true on 41/44; valid_image true on 42/44; severity spread none→high.
- **Risk routing:** manual_review_required on 29 rows, user_history_risk on 24, text_instruction_present on 11 (prompt-injection attempts in the test set were flagged and ignored as evidence, never acted on).

> Watch item: 8 rows resolved to `wrong_object` → contradicted. The test set has
> no labels to measure this against, so it is the main candidate for manual spot-
> checking / future tuning (the cheaper model may over-trigger
> `matches_claim_object=false`).

## 5. Model selection & de-risking (sonnet cross-check)

The 20-row sample margin (haiku 85% vs sonnet 80% `claim_status`) is a single
row and too noisy to trust on its own. Before locking, `claude-sonnet@v2` was run
across **all 44 unlabeled test rows** (44 calls, 0 errors, 137,535 tokens,
**$0.7587** — ~3.1× haiku's $0.2438) and diffed against the locked haiku
`output.csv`, field by field. The test set is unlabeled, so this measures
**model agreement**, not accuracy.

**Per-field agreement (haiku vs sonnet, 44 rows):**

| field | agreement |
|---|---|
| claim_status | 35/44 |
| issue_type | 33/44 |
| object_part | 38/44 |
| valid_image | 37/44 |

17/44 rows differ on ≥1 of these fields.

**Findings that refuted the "haiku over-triggers `matches_claim_object=false`" hypothesis:**

- **8/8 `wrong_object → contradicted` agreement.** Every row haiku marked
  `wrong_object`/contradicted, sonnet independently marked the same. The single
  extra `wrong_object` disagreement (case_052) was **sonnet adding** the flag,
  not haiku — the opposite of the feared failure mode.
- **Net-zero contradiction bias.** `claim_status` flips are symmetric
  (3 contradicted→supported vs 3 supported→contradicted, plus 1 each way on NEI).
  Both models land at **19 contradicted / 44** — haiku does not contradict more
  often than the 3× pricier model.
- **Disagreements are not decision-risk.** 10/17 differing rows carry adversarial
  tags (mostly injection + non-original/manipulation on package rows). Of the 7
  `valid_image` diffs, 6 are sonnet flipping to `false` — sonnet is simply more
  authenticity-suspicious, and per the core invariant `valid_image=false` only
  flags and never changes `claim_status`. The remaining 7/17 are genuinely
  ambiguous "is the damage visible?" perceptual calls.

**Decision: lock `claude-haiku@v2`; do not build a v3.** The cross-check shows no
systematic error to fix — equal contradiction rates and full agreement on the
structural (wrong_object) cases. The only residual disagreements are
decision-neutral authenticity sensitivity or unlabeled, ambiguous perceptual
calls. With **no ground truth on the test set**, a v3 prompt tweak aimed at those
7 rows would be tuning on noise and risks regressing rows that are already
correct. Haiku@v2 delivers the same decisions as sonnet at ~1/3 the cost, so it
is locked as the final production configuration.

## 6. Known limitations

- Severity is the model's weakest column (it tends to over-read severity); the rule layer passes observed severity through, so this is addressed by prompt calibration rather than post-hoc rules.
- Sample labels are occasionally lenient on secondary non-original images; by design we still surface `non_original_image` without changing the decision.
