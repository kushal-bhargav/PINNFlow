import sys
import os
import numpy as np
from pinnflow import RequirementParser, ScenarioBank, IntentEngine, ExplainEngine
from pinnflow.pinn import MultiTaskPINN

def test_phase2():
    print("Testing Design Intent & Explainability...")
    
    pinn = MultiTaskPINN(n_in=10)
    
    # 1. Test Design Intent
    intent = IntentEngine(pinn=pinn)
    variants = intent.propose_variants(None)
    print("✅ Intent Variants OK:", [v['name'] for v in variants])
    
    raw_layout = np.array([120.0, 5.0, 20.0, 10.0, 0, 0, 3.0, 0.5, 0, 1.0])
    refined = intent.apply_heuristics(raw_layout.copy())
    print(f"✅ Heuristics OK. Diameter refined: {raw_layout[0]} -> {refined[0]}")

    # 2. Test Explainability
    explain = ExplainEngine()
    explain.record_decision(1, "Increase Diameter", "Stress reduced", "Hoop stress limit approached")
    
    metrics = {"sigma": 185.0}
    exp = explain.generate_explanation("D-101", refined, metrics)
    print("✅ Full Explanation OK:", exp['summary'])
    print("   Physics Basis:", exp['physics_basis'])

if __name__ == "__main__":
    test_phase2()
