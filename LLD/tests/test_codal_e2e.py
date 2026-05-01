import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow.orchestrator_v2 import UnifiedOrchestrator


def test_codal_e2e():
    print("Testing Codal Intelligence E2E Orchestration...")

    orch = UnifiedOrchestrator()

    codes = orch.rule_store.list_codes()
    print(f"[OK] Rule Store OK. Loaded codes: {codes}")

    results = orch.run_e2e("refinery_compliance")

    print("[OK] E2E Run Complete.")
    print(f"   Report Summary: {results['report']['summary']}")
    print(f"   Compliance Score: {results['compliance_score']:.3f}")

    state = results["optimized_state"]
    report = []
    for agent in orch.agents:
        report.append(agent.evaluate(state, {}))

    print("[OK] Agent Critique Report:")
    for item in report:
        print(f"   [{item['agent']}] Status: {item['status']} | Penalty: {item['penalty']:.2f}")


if __name__ == "__main__":
    test_codal_e2e()
