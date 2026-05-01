import sys
import os
from pinnflow.ingestion.parser import RequirementParser
from pinnflow.scenarios.bank import ScenarioBank
from pinnflow.pinn import MultiTaskPINN

def test_phase1():
    print("Testing Ingestion & Scenarios...")
    
    # Instance PINN for validation (mock or small train)
    pinn = MultiTaskPINN(n_in=10)
    
    # 1. Test Ingestion
    parser = RequirementParser(pinn=pinn)
    result = parser.run_e2e("Example P&ID Data")
    print("✅ Ingestion OK. Lines found:", len(result['schema']['lines']))
    print("✅ Physics Validation OK. Overall Safety:", result['validation']['overall_safety'])

    # 2. Test Scenarios
    bank = ScenarioBank()
    scenario = bank.generate_scenario("high_pressure_gas")
    print("✅ Scenario Generation OK:", scenario['scenario_name'])
    print("   Inputs:", scenario['inputs'])

if __name__ == "__main__":
    test_phase1()
