"""Run one TradingAgents analysis and save a complete Arabic report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from openai import OpenAI
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


REPORT_KEYS = [
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_debate_state",
    "investment_plan",
    "trader_investment_plan",
    "risk_debate_state",
    "final_trade_decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-agent market report")
    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker, e.g. BTC-USD")
    parser.add_argument("--date", default=date.today().isoformat(), help="Analysis date: YYYY-MM-DD")
    parser.add_argument("--output", default="results/smart-analysis", help="Report directory")
    return parser.parse_args()


def readable(value: Any) -> str:
    if value is None:
        return "غير متوفر"
    if isinstance(value, str):
        return value.strip() or "غير متوفر"
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def collect_context(state: Any, decision: Any) -> str:
    if not isinstance(state, dict):
        return f"القرار الأصلي: {decision}"
    sections = [f"القرار الأصلي: {decision}"]
    for key in REPORT_KEYS:
        value = readable(state.get(key))
        sections.append(f"\n===== {key} =====\n{value[:12000]}")
    return "\n".join(sections)[:60000]


def build_arabic_report(ticker: str, analysis_date: str, context: str, model: str) -> str:
    instructions = """
أنت رئيس فريق تحليل أسواق مالية. حوّل مخرجات المحللين المرفقة إلى تقرير عربي واضح
ومفهوم لغير المتخصصين، من دون كود أو JSON. لا تخترع سعراً أو خبراً غير موجود في
المخرجات. إذا لم تتوفر بيانات كافية لهدف سعري موثوق، اذكر ذلك صراحة.

اكتب التقرير بهذه العناوين:
1. الخلاصة التنفيذية
2. السعر والاتجاه المتوقع
3. النطاق أو الحد السعري المتوقع قصير المدى ومتوسط المدى
4. أهم مستويات الدعم والمقاومة
5. السيناريو الصاعد وشروط تحققه
6. السيناريو الهابط وشروط تحققه
7. قراءة التحليل الفني
8. الأخبار ومعنويات السوق
9. المخاطر الرئيسية
10. القرار النهائي: شراء أو بيع أو انتظار، مع نسبة ثقة تقريبية وسبب القرار

استخدم لغة احتمالية لا جازمة، ووضّح أن التقرير للتحليل والتداول التجريبي وليس توصية
مالية أو أمراً لتنفيذ صفقة.
""".strip()
    prompt = (
        f"الأصل: {ticker}\n"
        f"تاريخ التحليل: {analysis_date}\n\n"
        f"مخرجات فريق TradingAgents:\n{context}"
    )
    response = OpenAI().responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=3500,
    )
    return response.output_text.strip()


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

    ticker = args.ticker.upper()
    graph = TradingAgentsGraph(debug=False, config=config)
    state, decision = graph.propagate(ticker, args.date)
    context = collect_context(state, decision)

    try:
        arabic_report = build_arabic_report(
            ticker, args.date, context, config["quick_think_llm"]
        )
    except Exception as exc:
        arabic_report = (
            "تعذر إنشاء الملخص العربي الإضافي، لكن التحليل الأساسي اكتمل.\n\n"
            f"القرار الأصلي: {decision}\n\n"
            f"تفاصيل المحللين:\n{context}\n\n"
            f"سبب تعذر التلخيص: {type(exc).__name__}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker.replace('/', '-')}-{args.date}"
    report_path = output_dir / f"{stem}.md"
    metadata_path = output_dir / f"{stem}.json"

    report_path.write_text(
        "\n".join(
            [
                f"# التقرير الذكي الكامل: {ticker}",
                "",
                f"**تاريخ التحليل:** {args.date}",
                "",
                "> للتحليل والتداول التجريبي فقط، ولا ينفذ أوامر تداول.",
                "",
                arabic_report,
                "",
                "## قرار مدير المحفظة الأصلي",
                "",
                str(decision),
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "analysis_date": args.date,
                "provider": config["llm_provider"],
                "deep_model": config["deep_think_llm"],
                "quick_model": config["quick_think_llm"],
                "decision": str(decision),
                "arabic_report": arabic_report,
                "state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved complete Arabic report to {report_path}")


if __name__ == "__main__":
    main()
