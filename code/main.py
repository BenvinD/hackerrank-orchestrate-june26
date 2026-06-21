"""Terminal entry point (project contract: code/main.py).

Current stage: Layer 1 (preprocessing) is implemented. Layers 2 (VLM provider)
and 3 (rule layer) are added next; once present, ``run`` will write output.csv.

Usage:
    uv run code/main.py preprocess --set sample   # summarize preprocessing
    uv run code/main.py preprocess --set test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the local package importable when run as a script (sys.path[0] == code/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from claimreview.config import load_config  # noqa: E402
from claimreview.preprocessing import prepare_dataset  # noqa: E402
from claimreview.observation import VLMObservation  # noqa: E402
from claimreview.provider.litellm_provider import extract_json  # noqa: E402
from claimreview.provider.messages import build_messages, cache_key  # noqa: E402
from claimreview.rules import decide, write_output_csv  # noqa: E402


def cmd_preprocess(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    prepared = prepare_dataset(args.set, cfg)

    total_images = sum(len(p.images) for p in prepared)
    usable_images = sum(len(p.usable_images) for p in prepared)
    missing = [
        (p.claim_input.user_id, img.rel_path)
        for p in prepared
        for img in p.images
        if not img.exists
    ]
    no_history = [p.claim_input.user_id for p in prepared if p.user_history is None]

    print(f"set:                 {args.set}")
    print(f"active_model:        {cfg.active_model}")
    print(f"claims prepared:     {len(prepared)}")
    print(f"images total:        {total_images}")
    print(f"images usable:       {usable_images}")
    print(f"images missing:      {len(missing)}")
    print(f"claims w/o history:  {len(no_history)}")
    if missing:
        print("  missing files:")
        for uid, rel in missing[:20]:
            print(f"    {uid}: {rel}")
    if no_history:
        print(f"  users w/o history:  {sorted(set(no_history))}")

    # Spot-check the first prepared claim so encoding is visibly working.
    if prepared:
        p0 = prepared[0]
        img0 = p0.images[0] if p0.images else None
        print("\nfirst claim:")
        print(f"  user_id:        {p0.claim_input.user_id}")
        print(f"  claim_object:   {p0.claim_input.claim_object}")
        print(f"  image_ids:      {p0.claim_input.image_ids}")
        print(f"  requirements:   {len(p0.evidence_requirements)} attached")
        if img0:
            print(
                f"  image[0]:       {img0.image_id} "
                f"{img0.orig_width}x{img0.orig_height} -> {img0.width}x{img0.height}, "
                f"hash={img0.content_hash[:12] if img0.content_hash else None}"
            )
    return 0


def cmd_provider_dryrun(args: argparse.Namespace) -> int:
    """Offline check of the layer-2 wiring: build messages + cache key and
    validate the observation schema. Makes NO API call (no key required)."""
    cfg = load_config(args.config)
    prepared = prepare_dataset(args.set, cfg)
    model_cfg = cfg.active_model_config

    print(f"active_model:   {cfg.active_model} ({model_cfg.provider}/{model_cfg.model})")
    print(f"prompt_version: {cfg.prompt_version}")
    print(f"claims:         {len(prepared)}")

    keys = set()
    total_img_blocks = 0
    for p in prepared:
        msgs = build_messages(p, cfg.prompt_version, model_cfg.provider)
        img_blocks = sum(
            1 for b in msgs[1]["content"] if b.get("type") == "image_url"
        )
        total_img_blocks += img_blocks
        keys.add(cache_key(p, model_cfg, cfg.prompt_version))

    print(f"messages built: {len(prepared)} (2 roles each: system + user)")
    print(f"image blocks:   {total_img_blocks} attached across all user messages")
    print(f"cache keys:     {len(keys)} unique")

    p0 = prepared[0]
    msgs0 = build_messages(p0, cfg.prompt_version, model_cfg.provider)
    print("\nfirst claim message shape:")
    print(f"  system chars: {len(msgs0[0]['content'])}")
    text_block = msgs0[1]["content"][0]["text"]
    print(f"  user text chars: {len(text_block)}")
    print(f"  user image blocks: {sum(1 for b in msgs0[1]['content'] if b.get('type') == 'image_url')}")

    # Validate the observation schema end-to-end with a synthetic model reply,
    # including a deliberately-invalid enum that must coerce to 'unknown'.
    fake_reply = (
        '```json\n{"detected_language":"en","claim_summary":"rear bumper dent",'
        '"claimed_issue_type":"dent","claimed_object_part":"rear_bumper",'
        '"claimed_parts":["rear_bumper"],"stated_severity":"medium",'
        '"injection_text_in_claim":false,"images":[{"image_id":"img_1",'
        '"object_in_image":"car","matches_claim_object":true,'
        '"claimed_part_visible":true,"visible_parts":["rear_bumper"],'
        '"damage_present":true,"observed_issue_type":"dent",'
        '"observed_object_part":"rear_bumper","observed_severity":"medium",'
        '"quality_flags":[],"authenticity_flags":[],"embedded_text_present":false,'
        '"supports_claim":true,"notes":"visible dent"}],'
        '"overall_notes":"supports claim","claimed_issue_type_bogus":"x"}\n```'
    )
    parsed = extract_json(fake_reply)
    obs = VLMObservation.model_validate(parsed)
    print("\nobservation schema self-test:")
    print(f"  parsed JSON: {'ok' if parsed else 'FAILED'}")
    print(f"  validated: claim_summary={obs.claim_summary!r}, images={len(obs.images)}")

    bad = extract_json('{"claimed_issue_type":"not_a_real_enum","images":[]}')
    obs_bad = VLMObservation.model_validate(bad)
    print(
        f"  invalid-enum coercion: claimed_issue_type "
        f"'not_a_real_enum' -> {obs_bad.claimed_issue_type!r}"
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Live end-to-end check against the first N sample claims (real API calls).

    Loads .env, runs the configured provider, and reports JSON validity, schema
    validation, token usage + cost, and cache writes. Never prints the API key.
    """
    import json as _json

    from claimreview.config import REPO_ROOT
    from claimreview.provider.litellm_provider import LiteLLMProvider

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not load .env ({exc}); relying on existing env")

    cfg = load_config(args.config)
    model_cfg = cfg.active_model_config
    key_env = model_cfg.api_key_env or ""
    if not os.environ.get(key_env):
        print(f"ERROR: env var {key_env} is not set (check .env). Aborting.")
        return 2
    print(f"{key_env}: present (value hidden)")
    print(f"active_model: {cfg.active_model} ({model_cfg.provider}/{model_cfg.model})")

    prepared = prepare_dataset(args.set, cfg)[: args.n]
    print(f"running live on {len(prepared)} claim(s)...\n")

    provider = LiteLLMProvider(cfg)
    cache_dir = cfg.resolve(cfg.cache.dir)
    before = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0

    ok = True
    for p in prepared:
        res = provider.analyze(p)
        uid = res.user_id
        print(f"=== {uid} ({p.claim_input.claim_object}) ===")
        if res.error:
            print(f"  ERROR: {res.error}")
            ok = False
            continue
        valid = res.observation is not None
        print(f"  cached:          {res.cached}")
        print(f"  json->schema ok: {valid}")
        print(f"  tokens:          prompt={res.prompt_tokens} "
              f"completion={res.completion_tokens} total={res.total_tokens}")
        print(f"  cost_usd:        {res.cost_usd:.6f}")
        print(f"  latency_s:       {res.latency_s:.2f}")
        if valid:
            obs = res.observation
            print(f"  claim_summary:   {obs.claim_summary!r}")
            print(f"  claimed:         issue={obs.claimed_issue_type} "
                  f"part={obs.claimed_object_part} parts={obs.claimed_parts} "
                  f"injection={obs.injection_text_in_claim}")
            for im in obs.images:
                print(f"  image[{im.image_id}]: object={im.object_in_image!r} "
                      f"matches={im.matches_claim_object} part_visible={im.claimed_part_visible} "
                      f"issue={im.observed_issue_type} part={im.observed_object_part} "
                      f"sev={im.observed_severity} supports={im.supports_claim} "
                      f"quality={im.quality_flags} auth={im.authenticity_flags}")
        else:
            ok = False
        print()

    after = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
    print(f"cache files: {before} -> {after} (dir: {cache_dir})")

    # Show one full raw observation so you can eyeball a real response.
    first_ok = next((provider.analyze(p) for p in prepared), None)
    if first_ok and first_ok.observation is not None:
        print("\nfull observation for first claim (from cache, no new call):")
        print(_json.dumps(first_ok.observation.model_dump(), indent=2, ensure_ascii=False))

    return 0 if ok else 1


def _load_env(cfg) -> None:
    from claimreview.config import REPO_ROOT

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end: preprocess -> VLM -> rule layer -> output.csv (live API calls)."""
    from claimreview.provider import analyze_claims
    from claimreview.provider.litellm_provider import LiteLLMProvider

    cfg = load_config(args.config)
    _load_env(cfg)
    model_cfg = cfg.active_model_config
    if not os.environ.get(model_cfg.api_key_env or ""):
        print(f"ERROR: {model_cfg.api_key_env} not set; cannot run live.")
        return 2

    prepared = prepare_dataset(args.set, cfg)
    print(f"running {cfg.active_model} on {len(prepared)} {args.set} claim(s)...")

    provider = LiteLLMProvider(cfg)
    done = {"n": 0}

    def _progress(_idx, res):
        done["n"] += 1
        tag = "cache" if res.cached else ("ERR " if res.error else "ok  ")
        print(f"  [{done['n']:>3}/{len(prepared)}] {tag} {res.user_id}")

    results = analyze_claims(provider, prepared, cfg.runtime.max_concurrency, _progress)

    rows = [decide(p, r.observation) for p, r in zip(prepared, results)]
    out_path = cfg.resolve(cfg.paths.output_csv)
    write_output_csv(rows, out_path)

    new_cost = sum(r.cost_usd for r in results if not r.cached)
    new_calls = sum(1 for r in results if not r.cached and not r.error)
    cached = sum(1 for r in results if r.cached)
    errors = sum(1 for r in results if r.error)
    toks = sum(r.total_tokens for r in results)
    print(f"\nwrote {len(rows)} rows -> {out_path}")
    print(f"calls: {new_calls} new, {cached} cached, {errors} errors | "
          f"tokens: {toks} | new cost: ${new_cost:.4f}")
    return 0 if errors == 0 else 1


def _read_expected(path) -> list[dict]:
    import csv as _csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(_csv.DictReader(fh))


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the rule layer against labeled sample rows using ONLY cached
    observations (zero new API spend). Rows without a cache hit are skipped."""
    from claimreview.provider.cache import DiskCache
    from claimreview.provider.litellm_provider import LiteLLMProvider

    cfg = load_config(args.config)
    model_cfg = cfg.active_model_config
    prepared = prepare_dataset("sample", cfg)
    expected = _read_expected(cfg.resolve(cfg.paths.sample_csv))
    cache = DiskCache(cfg.resolve(cfg.cache.dir), enabled=True)
    provider = LiteLLMProvider(cfg)

    set_cols = {"risk_flags", "supporting_image_ids"}
    compare_cols = [
        "evidence_standard_met", "claim_status", "issue_type", "object_part",
        "valid_image", "severity", "risk_flags", "supporting_image_ids",
    ]
    totals = {c: [0, 0] for c in compare_cols}  # [correct, counted]
    n_eval = 0

    for p, exp in zip(prepared, expected):
        key = cache_key(p, model_cfg, cfg.prompt_version)
        if cache.get(key) is None:
            continue  # no cache -> skip to avoid an API call
        n_eval += 1
        res = provider.analyze(p)  # cache hit, no spend
        row = decide(p, res.observation).to_csv_dict()

        print(f"\n=== {p.claim_input.user_id} ({p.claim_input.claim_object}) ===")
        for c in compare_cols:
            got, want = row[c], exp.get(c, "")
            if c in set_cols:
                ok = set(filter(None, got.split(";"))) == set(filter(None, want.split(";")))
            else:
                ok = got.strip() == want.strip()
            totals[c][1] += 1
            totals[c][0] += int(ok)
            mark = "OK " if ok else "XX "
            print(f"  {mark}{c:<24} got={got!r:<28} want={want!r}")

    print(f"\n--- per-column accuracy on {n_eval} cached sample row(s) ---")
    if n_eval == 0:
        print("  (no cached sample observations; run `smoke`/`run --set sample` first)")
        return 0
    for c in compare_cols:
        corr, cnt = totals[c]
        print(f"  {c:<24} {corr}/{cnt}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-modal damage-claim review")
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (default: code/config.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Load + encode a dataset and print a summary")
    pp.add_argument("--set", choices=["sample", "test"], default="sample")
    pp.set_defaults(func=cmd_preprocess)

    dr = sub.add_parser(
        "provider-dryrun",
        help="Offline check of layer-2 wiring (no API call / no key needed)",
    )
    dr.add_argument("--set", choices=["sample", "test"], default="sample")
    dr.set_defaults(func=cmd_provider_dryrun)

    sm = sub.add_parser("smoke", help="Live end-to-end check on the first N claims")
    sm.add_argument("--set", choices=["sample", "test"], default="sample")
    sm.add_argument("--n", type=int, default=2, help="Number of claims to run live")
    sm.set_defaults(func=cmd_smoke)

    run = sub.add_parser("run", help="Produce output.csv end-to-end (live API)")
    run.add_argument("--set", choices=["sample", "test"], default="test")
    run.set_defaults(func=cmd_run)

    val = sub.add_parser(
        "validate", help="Validate rule layer vs labeled sample (cached only, no spend)"
    )
    val.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
