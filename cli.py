"""CLI. Usage:
    python cli.py optimize path/to/request.json
    python cli.py backtest path/to/request.json --rebalance Q --lookback 3
    python cli.py sectors                              # list available sector tags
    python cli.py indices --sector IT                  # list indices in a sector
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from engine.optimizer import UserRequest, optimize
from engine.backtest import walk_forward
from engine.sector_map import build_map, available_sectors, indices_for_sectors


def _load_req(path: str) -> UserRequest:
    with open(path) as f:
        data = json.load(f)
    return UserRequest(**data)


def _pct(x, d=2):
    return f"{x*100:.{d}f}%" if x is not None else "-"


def cmd_optimize(args):
    req = _load_req(args.request)
    result = optimize(req)
    print("\n=== PORTFOLIO OPTIMIZATION RESULT ===\n")
    print(f"Chosen model     : {result.chosen_model}")
    print(f"Feasible         : {result.feasible}  {('('+result.reason+')') if result.reason else ''}")
    print(f"Universe size    : {result.n_assets_used} indices")
    print(f"\nExpected return  : {_pct(result.metrics.get('ann_return'))}")
    print(f"Volatility (ann) : {_pct(result.metrics.get('ann_vol'))}")
    print(f"Sharpe           : {result.metrics.get('sharpe', 0):.3f}")
    print(f"Sortino          : {result.metrics.get('sortino', 0):.3f}")
    print(f"Max drawdown     : {_pct(result.metrics.get('max_drawdown'))}")
    print(f"CVaR 5% (daily)  : {_pct(result.metrics.get('cvar'))}")
    print(f"Calmar           : {result.metrics.get('calmar', 0):.3f}")

    print("\n--- Weights (top holdings) ---")
    sorted_w = sorted(result.weights.items(), key=lambda kv: -kv[1])
    for name, w in sorted_w[:25]:
        print(f"  {name:<50s}  {w*100:6.2f}%")
    if len(sorted_w) > 25:
        print(f"  ... {len(sorted_w) - 25} more, total = {sum(w for _, w in sorted_w):.4f}")

    print("\n--- All candidate models ---")
    for c in result.all_candidates:
        m = c.get("metrics", {}) or {}
        print(f"  [{c.get('model'):<25s}]  feas={c.get('feasible')}  "
              f"score={c.get('score', 0):.4f}  "
              f"ret={_pct(m.get('ann_return'))}  vol={_pct(m.get('ann_vol'))}  "
              f"dd={_pct(m.get('max_drawdown'))}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "chosen_model": result.chosen_model,
            "feasible": result.feasible,
            "reason": result.reason,
            "weights": result.weights,
            "metrics": result.metrics,
            "candidates": result.all_candidates,
        }, indent=2, default=str))
        print(f"\nSaved -> {args.out}")


def cmd_backtest(args):
    req = _load_req(args.request)
    result = walk_forward(req, rebalance=args.rebalance, lookback_years=args.lookback)
    print("\n=== WALK-FORWARD BACKTEST ===\n")
    m = result["metrics"]
    print(f"Rebalances       : {len(result['rebalance_dates'])}")
    print(f"Ann return       : {_pct(m['ann_return'])}")
    print(f"Ann vol          : {_pct(m['ann_vol'])}")
    print(f"Sharpe           : {m['sharpe']:.3f}")
    print(f"Max drawdown     : {_pct(m['max_drawdown'])}")
    print(f"Calmar           : {m['calmar']:.3f}")

    if args.out:
        eq = result["equity_curve"]
        eq.to_csv(args.out)
        print(f"Saved equity curve -> {args.out}")


def cmd_sectors(args):
    print(json.dumps(available_sectors(), indent=2))


def cmd_indices(args):
    picks = indices_for_sectors([args.sector])
    for p in picks:
        print(p)
    print(f"\nTotal: {len(picks)}")


def main():
    p = argparse.ArgumentParser(prog="macrowise")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("optimize");  a.add_argument("request")
    a.add_argument("--out", default=None); a.set_defaults(fn=cmd_optimize)

    b = sub.add_parser("backtest");  b.add_argument("request")
    b.add_argument("--rebalance", default="Q"); b.add_argument("--lookback", type=int, default=3)
    b.add_argument("--out", default=None); b.set_defaults(fn=cmd_backtest)

    c = sub.add_parser("sectors");   c.set_defaults(fn=cmd_sectors)

    d = sub.add_parser("indices");   d.add_argument("--sector", required=True)
    d.set_defaults(fn=cmd_indices)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
