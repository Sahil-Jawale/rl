from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "benchmark_summary.json"
REPORT_PATH = ROOT / "results" / "benchmark_report.md"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Benchmark summary not found at {RESULTS_PATH}")

    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    lines = [
        "# Benchmark report",
        "",
        "## Summary",
        "",
        "The current scaffold compares three model families under the same endgame setup and identical random seeds.",
        "",
        "| Scenario | Algorithm | Mean reward | Episodes |",
        "| --- | --- | ---: | ---: |",
    ]

    for item in results:
        lines.append(f"| {item['scenario']} | {item['algorithm']} | {item['mean_reward']:.3f} | {item['summary']['count']} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- KR-K remains the primary learnable benchmark. The models are expected to learn a forced mate strategy, while the reward remains sparse and highly dependent on defender behavior.",
        "- KN-K is a draw-control scenario. The benchmark is expected to remain near zero win rate, so the relevant metrics are survival and sanity checks rather than successful mate forcing.",
        "- All comparisons use the same budget and the same random defender setup so that the learning curves are comparable across algorithms.",
        "",
        "## Next steps",
        "",
        "- Add stronger defender variants and tablebase-based KR-K evaluation.",
        "- Extend the interface to compare three boards side by side and overlay move explanations.",
        "- Expand experimentation with improved hyperparameters and equalized training steps.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
