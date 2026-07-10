import json
import math
from pathlib import Path
from collections import defaultdict
from engine.optimizer import UserRequest, optimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'goal_sector_test_results.json'

sector_sets = [
    ("sample_mix", ["IT", "Banks", "Auto", "Pharma", "FMCG"]),
    ("single_it", ["IT"]),
    ("single_banks", ["Banks"]),
    ("cyclical", ["Energy", "Metal", "Realty"]),
    ("consumer_infra_fin", ["Consumer", "Infrastructure", "FinancialServices"]),
    ("broad_mix", ["IT", "Banks", "Auto", "Pharma", "FMCG", "Energy", "Metal", "Realty"]),
    ("defence", ["Defence"]),
    ("it_banks", ["IT", "Banks"]),
    ("all_default", []),
]

goals = [
    "max_sharpe",
    "min_risk",
    "max_return",
    "balanced",
    "min_tail_risk",
    "min_drawdown",
    "max_sortino",
    "black_litterman",
    "inverse_vol",
    "max_diversification",
]

weight_caps = [0.30, 0.20]
match_modes = [False, True]

rows = []

for goal in goals:
    for name, sectors in sector_sets:
        for match_all in match_modes:
            for w_max in weight_caps:
                try:
                    req = UserRequest(
                        sectors=sectors or None,
                        primary_goal=goal,
                        w_min=0.0,
                        w_max=w_max,
                        lookback_years=5,
                        risk_free_rate=0.065,
                        sector_match_all=match_all,
                    )
                    res = optimize(req)
                    weights = res.weights or {}
                    metrics = res.metrics or {}
                    max_w = max(weights.values()) if weights else 0.0
                    n_assets = len(weights)
                    flags = []
                    if not res.feasible:
                        flags.append("infeasible")
                    if max_w > w_max + 1e-9:
                        flags.append("weight_cap_breach")
                    if n_assets <= 2:
                        flags.append("very_concentrated")
                    if metrics.get("ann_return", 0) > 0.50:
                        flags.append("very_high_return")
                    if metrics.get("ann_vol", 0) > 0.60:
                        flags.append("very_high_vol")
                    if metrics.get("sharpe", 0) > 1.5:
                        flags.append("very_high_sharpe")
                    if metrics.get("max_drawdown", 0) > 0.40:
                        flags.append("very_high_drawdown")
                    if metrics.get("ann_return", 0) < -0.20:
                        flags.append("negative_return")

                    rows.append({
                        "goal": goal,
                        "sector_set": name,
                        "sectors": sectors,
                        "match_all": match_all,
                        "w_max": w_max,
                        "model": res.chosen_model,
                        "feasible": bool(res.feasible),
                        "ann_return": round(metrics.get("ann_return", float("nan")), 6),
                        "ann_vol": round(metrics.get("ann_vol", float("nan")), 6),
                        "sharpe": round(metrics.get("sharpe", float("nan")), 6),
                        "max_drawdown": round(metrics.get("max_drawdown", float("nan")), 6),
                        "n_assets": n_assets,
                        "max_weight": round(max_w, 6),
                        "weights": {k: round(float(v), 6) for k, v in sorted(weights.items())},
                        "flags": flags,
                    })
                except Exception as exc:
                    rows.append({
                        "goal": goal,
                        "sector_set": name,
                        "sectors": sectors,
                        "match_all": match_all,
                        "w_max": w_max,
                        "model": "ERROR",
                        "feasible": False,
                        "ann_return": None,
                        "ann_vol": None,
                        "sharpe": None,
                        "max_drawdown": None,
                        "n_assets": None,
                        "max_weight": None,
                        "weights": {},
                        "flags": [str(exc)],
                    })

OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')

print(f"Completed {len(rows)} runs")
print(f"Saved to {OUT}")

# summary by goal and sector set
summary = defaultdict(lambda: {"runs": 0, "feasible": 0, "flagged": 0, "avg_return": [], "avg_vol": [], "avg_sharpe": []})
for r in rows:
    key = (r["goal"], r["sector_set"], r["match_all"])
    summary[key]["runs"] += 1
    summary[key]["feasible"] += int(r["feasible"])
    summary[key]["flagged"] += int(bool(r["flags"]))
    if r["ann_return"] is not None:
        summary[key]["avg_return"].append(r["ann_return"])
    if r["ann_vol"] is not None:
        summary[key]["avg_vol"].append(r["ann_vol"])
    if r["sharpe"] is not None:
        summary[key]["avg_sharpe"].append(r["sharpe"])

print("\nSUMMARY")
for key in sorted(summary):
    s = summary[key]
    avg_ret = round(sum(s["avg_return"]) / len(s["avg_return"]), 4) if s["avg_return"] else None
    avg_vol = round(sum(s["avg_vol"]) / len(s["avg_vol"]), 4) if s["avg_vol"] else None
    avg_sh = round(sum(s["avg_sharpe"]) / len(s["avg_sharpe"]), 4) if s["avg_sharpe"] else None
    print(key, {"runs": s["runs"], "feasible": s["feasible"], "flagged": s["flagged"], "avg_ret": avg_ret, "avg_vol": avg_vol, "avg_sharpe": avg_sh})

print("\nSUSPICIOUS_CASES")
for r in rows:
    if r["flags"]:
        print(json.dumps({k: r[k] for k in ["goal", "sector_set", "sectors", "match_all", "w_max", "model", "feasible", "ann_return", "ann_vol", "sharpe", "max_drawdown", "n_assets", "max_weight", "flags"]}, default=str))
