"""
Unified benchmark entrypoint for PINNFlow + ieeejen paper modules.
"""
from __future__ import annotations

from pinnflow.unified_benchmark import UnifiedBenchmarkRunner


def run_evaluation() -> None:
    runner = UnifiedBenchmarkRunner()
    results = runner.run_all()

    print("\n[V8] Consolidated benchmark artifacts:")
    for key, path in results["report_paths"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    run_evaluation()
