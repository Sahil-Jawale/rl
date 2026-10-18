from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "benchmark_summary.json"
REPORT_PATH = ROOT / "results" / "benchmark_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from benchmark results.")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH, help="Benchmark JSON file")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Markdown report destination")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Benchmark summary not found at {args.input}")

    with args.input.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    lines = [
        "# Benchmark report",
        "",
        "## Summary",
        "",
        "Each model is trained with the same episode budget and evaluated greedily on reproducibly seeded games.",
        "",
        "| Scenario | Algorithm | Evaluation games | Checkmates | Checkmate rate | Draw rate | Timeout rate | Mean reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in results:
        evaluation = item.get("evaluation")
        if evaluation:
            lines.append(
                f"| {item['scenario']} | {item['algorithm']} | {evaluation['games']} "
                f"| {evaluation['checkmates']} | {evaluation['checkmate_rate']:.1%} "
                f"| {evaluation['draw_rate']:.1%} | {evaluation['timeout_rate']:.1%} "
                f"| {evaluation['mean_reward']:.3f} |"
            )
        else:
            lines.append(
                f"| {item['scenario']} | {item['algorithm']} | "
                f"{item.get('summary', {}).get('count', 0)} | — | — | — | — "
                f"| {item.get('mean_reward', 0.0):.3f} |"
            )

    lines.extend([
        "",
        "## Benchmark configuration",
        "",
        "See benchmark_summary.json for per-seed metrics, training outcomes, model paths, and defender settings.",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()
