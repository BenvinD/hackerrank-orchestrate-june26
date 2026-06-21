"""Evaluation entry point (project contract: code/evaluation/main.py).

Runs the system on dataset/sample_claims.csv for one or more model configs,
scores predictions against the labeled outputs, compares models, and writes
evaluation/evaluation_report.md (metrics + operational analysis).

Usage:
    uv run code/evaluation/main.py                       # compare default models
    uv run code/evaluation/main.py --models claude-sonnet claude-haiku
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claimreview.config import REPO_ROOT, load_config  # noqa: E402
from claimreview.evaluation import EvalRun, run_model  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.md"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass


def _test_set_size(cfg) -> tuple[int, int]:
    """(claims, images) in the test set, counted without encoding/API."""
    path = cfg.resolve(cfg.paths.claims_csv)
    claims = images = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            claims += 1
            images += len([p for p in (row.get("image_paths") or "").split(";") if p.strip()])
    return claims, images


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _estimate_test_cost(run: EvalRun, cfg, test_claims: int) -> float | None:
    mc = cfg.models[run.model_name]
    if mc.input_per_mtok is None or mc.output_per_mtok is None or run.ops.claims == 0:
        return None
    avg_p = run.ops.prompt_tokens / run.ops.claims
    avg_c = run.ops.completion_tokens / run.ops.claims
    per_claim = (avg_p * mc.input_per_mtok + avg_c * mc.output_per_mtok) / 1_000_000
    return per_claim * test_claims


def _parse_specs(models: list[str], runs: list[str] | None, default_pv: str) -> list[tuple[str, str]]:
    """Build (model_name, prompt_version) specs from --runs (model:prompt) or --models."""
    if runs:
        out: list[tuple[str, str]] = []
        for spec in runs:
            model, _, pv = spec.partition(":")
            out.append((model, pv or default_pv))
        return out
    return [(m, default_pv) for m in models]


def _render(runs: list[EvalRun], cfg, test_claims: int, test_images: int, final: str) -> str:
    L: list[str] = []
    L.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    L.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    L.append(
        "System design: pure-Python preprocessing -> one structured VLM call per "
        "claim (LiteLLM, provider-agnostic) that **observes** each image -> a "
        "deterministic rule layer that maps observations to the 14 output columns. "
        "Images are the source of truth; user history and authenticity flags add "
        "risk context but never flip a supported decision; in-claim/in-image "
        "instructions are treated as data (`text_instruction_present`).\n"
    )

    n = runs[0].metrics.n if runs[0].metrics else 0
    L.append(f"## 1. Accuracy on labeled sample ({n} rows)\n")
    L.append("Per-column exact-match accuracy (risk_flags / supporting_image_ids compared as sets):\n")
    cols = ["claim_status", "evidence_standard_met", "valid_image", "issue_type",
            "object_part", "severity", "risk_flags", "supporting_image_ids"]
    header = "| Model | " + " | ".join(cols) + " | risk F1 | sev ±1 |"
    sep = "|" + "---|" * (len(cols) + 3)
    L.append(header)
    L.append(sep)
    for r in runs:
        m = r.metrics
        if not m:
            continue
        cells = [_fmt_pct(m.column_accuracy.get(c, 0.0)) for c in cols]
        row = (f"| `{r.label}` | " + " | ".join(cells)
               + f" | {m.risk_f1:.2f} | {_fmt_pct(m.severity_within_one)} |")
        L.append(row)
    L.append("")

    L.append("### claim_status confusion (want -> got)\n")
    for r in runs:
        if not r.metrics:
            continue
        L.append(f"- **`{r.label}`**: " + ", ".join(
            f"`{k}`×{v}" for k, v in sorted(r.metrics.status_confusion.items())))
    L.append("")
    L.append(
        "> Note: `manual_review_required` recall is reported separately because "
        "routing risky claims to a human is the operationally important signal:\n"
    )
    for r in runs:
        if r.metrics:
            L.append(f"> - `{r.label}`: manual_review_required recall "
                     f"{_fmt_pct(r.metrics.manual_review_recall)}")
    L.append("")

    L.append("## 2. Configuration comparison & final choice\n")
    L.append(
        f"The configurations below (`model@prompt_version`) were run on identical "
        f"inputs. **Final configuration: `{final}`**, selected for the best "
        f"`claim_status` accuracy, breaking ties by lower cost. Other configurations "
        f"are retained in config as fallbacks / cost levers.\n"
    )

    L.append("## 3. Operational analysis\n")
    L.append("Sample run (actual; cached calls billed at $0):\n")
    L.append("| Model | claims | new calls | cached | images | tokens (in/out) | new cost | avg/max latency |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        o = r.ops
        L.append(
            f"| `{r.label}` | {o.claims} | {o.new_calls} | {o.cached} | {o.images} "
            f"| {o.prompt_tokens}/{o.completion_tokens} | ${o.cost_usd:.4f} "
            f"| {o.avg_latency_s:.1f}s / {o.max_latency_s:.1f}s |")
    L.append("")

    L.append(f"Test-set projection ({test_claims} claims, {test_images} images), "
             f"one VLM call per claim, all images batched into that call:\n")
    L.append("| Model | est. calls | est. cost (no cache) | pricing (in/out per MTok) |")
    L.append("|---|---|---|---|")
    for r in runs:
        mc = cfg.models[r.model_name]
        est = _estimate_test_cost(r, cfg, test_claims)
        est_s = f"${est:.2f}" if est is not None else "n/a"
        price = (f"${mc.input_per_mtok}/{mc.output_per_mtok}"
                 if mc.input_per_mtok is not None else "n/a")
        L.append(f"| `{r.label}` | {test_claims} | {est_s} | {price} |")
    L.append("")
    L.append(
        "**Pricing assumptions:** Anthropic API list prices verified Jun 2026 "
        "(`claude-sonnet-4-6` $3/$15, `claude-haiku-4-5` $1/$5 per MTok input/output); "
        "no prompt-cache or Batch-API discounts assumed (both would lower cost ~50–90%).\n"
    )
    L.append("**Calls / images:** exactly one model call per claim; multiple images "
             "are batched as content blocks in that single call (no per-image calls).\n")
    L.append(
        "**Latency / throughput:** claims run concurrently with a bounded thread "
        f"pool (`max_concurrency={cfg.runtime.max_concurrency}`), keeping RPM/TPM "
        "pressure low while overlapping network I/O.\n"
    )
    L.append(
        "**Cost & rate-limit controls:**\n"
        "- Images downscaled to a bounded long side before encoding (image tokens "
        "dominate cost).\n"
        "- Content-hash disk cache keyed on (model, prompt version, claim text, "
        "image hashes): identical inputs are never re-billed, so reruns and the "
        "model comparison reuse prior work.\n"
        f"- Retry with exponential backoff (`max_retries={cfg.runtime.max_retries}`) "
        "on transient API errors; a single JSON-repair re-ask before failing a claim.\n"
        "- `temperature=0` for determinism.\n"
    )
    L.append("## 4. Known limitations\n")
    L.append(
        "- Severity is the model's weakest column (it tends to over-read severity); "
        "the rule layer passes observed severity through, so this is addressed by "
        "prompt calibration rather than post-hoc rules.\n"
        "- Sample labels are occasionally lenient on secondary non-original images; "
        "by design we still surface `non_original_image` without changing the decision.\n"
    )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate + compare models on sample")
    parser.add_argument("--models", nargs="+", default=["claude-sonnet", "claude-haiku"])
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Explicit run specs 'model:prompt_version' (overrides --models)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _load_env()
    key_env = cfg.active_model_config.api_key_env or ""
    if not os.environ.get(key_env):
        print(f"ERROR: {key_env} not set (check .env). Aborting.")
        return 2

    specs = _parse_specs(args.models, args.runs, cfg.prompt_version)
    test_claims, test_images = _test_set_size(cfg)
    runs: list[EvalRun] = []
    for name, pv in specs:
        print(f"\n=== evaluating {name}@{pv} ===")
        done = {"n": 0}

        def _p(_i, res):
            done["n"] += 1
            tag = "cache" if res.cached else ("ERR" if res.error else "ok")
            print(f"  [{done['n']:>2}] {tag:<5} {res.user_id}")

        run = run_model(cfg, name, "sample", _p, prompt_version=pv)
        runs.append(run)
        if run.metrics:
            m = run.metrics
            print(f"  claim_status acc: {_fmt_pct(m.column_accuracy.get('claim_status', 0))} "
                  f"| risk F1: {m.risk_f1:.2f} | new cost: ${run.ops.cost_usd:.4f}")

    scored = [r for r in runs if r.metrics]
    final = max(
        scored,
        key=lambda r: (r.metrics.column_accuracy.get("claim_status", 0), -r.ops.cost_usd),
    ).label if scored else f"{specs[0][0]}@{specs[0][1]}"

    report = _render(runs, cfg, test_claims, test_images, final)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nfinal model: {final}")
    print(f"wrote report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
