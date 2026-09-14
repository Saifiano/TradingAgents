"""Run one TradingAgents analysis and save a GitHub Actions-friendly report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-agent market report")
    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker, e.g. BTC-USD")
    parser.add_argument("--date", default=date.today().isoformat(), help="Analysis date: YYYY-MM-DD")
    parser.add_argument("--output", default="results/smart-analysis", help="Report directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.getenv("TRADINGAGENTS_LLM_PROVIDER", "openai")
    config["deep_think_llm"] = os.getenv("TRADINGAGENTS_DEEP_THINK_LLM", "gpt-5.6")
    config["quick_think_llm"] = os.getenv(
        "TRADINGAGENTS_QUICK_THINK_LLM", "gpt-5.6-luna"
    )
    config["max_debate_rounds"] = int(os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1"))
    config["max_risk_discuss_rounds"] = int(
        os.getenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "1")
    )
    config["checkpoint_enabled"] = True

    graph = TradingAgentsGraph(debug=False, config=config)
    state, decision = graph.propagate(args.ticker.upper(), args.date)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.ticker.upper().replace('/', '-')}-{args.date}"
    report_path = output_dir / f"{stem}.md"
    metadata_path = output_dir / f"{stem}.json"

    report_path.write_text(
        "\n".join(
            [
                f"# Smart analysis: {args.ticker.upper()}",
                "",
                f"**Analysis date:** {args.date}",
                "",
                "> Research/paper-trading use only. This report does not execute orders.",
                "",
                "## Portfolio manager decision",
                "",
                str(decision),
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "ticker": args.ticker.upper(),
                "analysis_date": args.date,
                "provider": config["llm_provider"],
                "deep_model": config["deep_think_llm"],
                "quick_model": config["quick_think_llm"],
                "decision": str(decision),
                "state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
