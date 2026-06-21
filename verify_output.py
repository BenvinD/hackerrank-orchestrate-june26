import csv, sys, pathlib

OUT = "output.csv"
CLAIMS = "dataset/claims.csv"

EXPECTED_HEADER = [
    "user_id","image_paths","user_claim","claim_object",
    "evidence_standard_met","evidence_standard_met_reason","risk_flags",
    "issue_type","object_part","claim_status","claim_status_justification",
    "supporting_image_ids","valid_image","severity",
]

CLAIM_STATUS = {"supported","contradicted","not_enough_information"}
ISSUE_TYPE = {"dent","scratch","crack","glass_shatter","broken_part","missing_part",
              "torn_packaging","crushed_packaging","water_damage","stain","none","unknown"}
SEVERITY = {"none","low","medium","high","unknown"}
BOOL = {"true","false"}
OBJECT_PARTS = {  # union across car/laptop/package + unknown
    "front_bumper","rear_bumper","door","hood","windshield","side_mirror","headlight",
    "taillight","fender","quarter_panel","body","screen","keyboard","trackpad","hinge",
    "lid","corner","port","base","box","package_corner","package_side","seal","label",
    "contents","item","unknown",
}
RISK_FLAGS = {"none","blurry_image","cropped_or_obstructed","low_light_or_glare","wrong_angle",
              "wrong_object","wrong_object_part","damage_not_visible","claim_mismatch",
              "possible_manipulation","non_original_image","text_instruction_present",
              "user_history_risk","manual_review_required"}

errors, warnings = [], []

# --- load claims.csv for row-count + alignment check ---
claims_rows = list(csv.DictReader(open(CLAIMS, newline="", encoding="utf-8")))
n_claims = len(claims_rows)

with open(OUT, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

header, data = rows[0], rows[1:]

# 1. header exact match + order
if header != EXPECTED_HEADER:
    errors.append(f"HEADER mismatch.\n  got:      {header}\n  expected: {EXPECTED_HEADER}")

# 2. row count vs claims.csv
if len(data) != n_claims:
    errors.append(f"ROW COUNT: output has {len(data)} data rows, claims.csv has {n_claims}")

# 3. per-row checks
col = {name: i for i, name in enumerate(EXPECTED_HEADER)}
for idx, r in enumerate(data, start=1):
    if len(r) != 14:
        errors.append(f"row {idx}: has {len(r)} columns, expected 14"); continue
    def chk(field, allowed, multi=False):
        val = r[col[field]].strip()
        if val == "":
            errors.append(f"row {idx}: empty {field}"); return
        vals = [v.strip() for v in val.split(";")] if multi else [val]
        for v in vals:
            if v not in allowed:
                errors.append(f"row {idx}: {field}='{v}' not in allowed values")
    chk("claim_object", {"car","laptop","package"})
    chk("evidence_standard_met", BOOL)
    chk("risk_flags", RISK_FLAGS, multi=True)
    chk("issue_type", ISSUE_TYPE)
    chk("object_part", OBJECT_PARTS)
    chk("claim_status", CLAIM_STATUS)
    chk("valid_image", BOOL)
    chk("severity", SEVERITY)
    # free-text fields just must be non-empty
    for ftxt in ("evidence_standard_met_reason","claim_status_justification","supporting_image_ids"):
        if r[col[ftxt]].strip() == "":
            warnings.append(f"row {idx}: empty {ftxt}")
    # alignment: user_id + claim_object should match claims.csv same row
    if idx-1 < n_claims:
        cr = claims_rows[idx-1]
        if r[col["user_id"]] != cr.get("user_id",""):
            warnings.append(f"row {idx}: user_id '{r[col['user_id']]}' != claims.csv '{cr.get('user_id','')}' (order drift?)")

# --- report ---
print(f"\noutput.csv: {len(data)} data rows | claims.csv: {n_claims} rows")
print(f"errors: {len(errors)} | warnings: {len(warnings)}\n")
for e in errors: print("  ERROR:", e)
for w in warnings[:10]: print("  warn :", w)
if len(warnings) > 10: print(f"  ... +{len(warnings)-10} more warnings")

print("\n" + ("✅ PASS — safe to submit" if not errors else "❌ FAIL — fix errors before submitting"))
sys.exit(1 if errors else 0)