"""
Unified Benchmark Orchestrator for Lumen.

Runs retrieval, forgetting, and perf suites sequentially and generates
a unified Markdown report at benchmarks/results/report.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUITES = ["retrieval", "forgetting", "perf"]


def run_suite(name: str) -> dict[str, Any]:
    script = BENCHMARKS_DIR / name / "run.py"
    if not script.exists():
        print(f"[ERROR] Suite script not found: {script}")
        return {"suite": name, "status": "missing", "results": None}

    print(f"\n{'='*60}")
    print(f"Running benchmark suite: {name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BENCHMARKS_DIR.parent),
        capture_output=False,
    )

    json_path = BENCHMARKS_DIR / name / "results" / f"{name}_results.json"
    data: dict[str, Any] = {"suite": name, "status": "unknown", "results": None}
    if result.returncode == 0 and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data["results"] = json.load(f)
            data["status"] = "success"
        except Exception as exc:
            data["status"] = f"json_error: {exc}"
    else:
        data["status"] = f"failed (exit code {result.returncode})"

    return data


def generate_report(suite_results: list[dict[str, Any]]) -> Path:
    lines = [
        "# Lumen Benchmark Suite Report",
        "",
        f"**Date:** {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Suite | Status | Key Metrics |",
        "|---|---|---|",
    ]

    for sr in suite_results:
        status = sr["status"]
        res = sr.get("results", {})
        if sr["suite"] == "retrieval" and isinstance(res, dict):
            metrics = []
            for cfg in ["bm25_only", "dense_only", "hybrid"]:
                m = res.get("results", {}).get(cfg, {})
                metrics.append(
                    f"{cfg}: R@10={m.get('recall@10_mean', 'N/A')} nDCG@10={m.get('ndcg@10_mean', 'N/A')}"
                )
            key = "; ".join(metrics)
        elif sr["suite"] == "forgetting" and isinstance(res, dict):
            key = (
                f"alive={res.get('final_state', {}).get('alive', 'N/A')}, "
                f"forgotten={res.get('final_state', {}).get('forgotten', 'N/A')} "
                f"({res.get('final_state', {}).get('pct_released', 'N/A')}% released)"
            )
        elif sr["suite"] == "perf" and isinstance(res, dict):
            metrics = []
            for r in res.get("per_size", []):
                lat = r.get("query_latency_ms", {})
                metrics.append(
                    f"{r.get('corpus_size', '?'):,}: p50={lat.get('p50_ms', 'N/A')}ms "
                    f"DB={r.get('db_size_mb', 'N/A')}MB"
                )
            key = "; ".join(metrics)
        else:
            key = "N/A"

        lines.append(f"| {sr['suite']} | {status} | {key} |")

    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])

    for sr in suite_results:
        lines.append(f"### {sr['suite']}")
        lines.append("")
        if isinstance(sr.get("results"), dict):
            lines.append("```json")
            lines.append(json.dumps(sr["results"], indent=2))
            lines.append("```")
        else:
            lines.append(f"Status: {sr['status']}")
        lines.append("")

    report_path = RESULTS_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main() -> int:
    suite_results: list[dict[str, Any]] = []
    for name in SUITES:
        suite_results.append(run_suite(name))

    report_path = generate_report(suite_results)
    print(f"\n[INFO] Unified report written to {report_path}")

    all_ok = all(r["status"] == "success" for r in suite_results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
