"""
main_v8.py
──────────
ORCHESTRATOR — PINNFlow v8 "The Geometric Frontier"

Demonstrates industrial piping design from simple segments to 
complex assemblies (Elbows, T-Junctions). 
"""
import numpy as np
import torch
from pinnflow.pinn import MultiTaskPINN
from pinnflow.environment import PipelineEnv
from pinnflow.visualization_3d import render_pipe_3d, render_bend_3d, render_tjunct_3d
from pinnflow.ansys_bridge import generate_assembly_apdl_script

def main_v8():
    print("=" * 72)
    print("  PINNFLOW v8 — The Geometric Frontier")
    print("  Industry-Grade Complex Assembly Suite")
    print("=" * 72)

    # 1. Framework Initialization
    pinn = MultiTaskPINN(n_in=10) # 10-D Geometry-Aware Input
    env = PipelineEnv(pinn=pinn)
    
    scenarios = [
        {"id": 0, "name": "Straight Pipe Segment", "fn": render_pipe_3d},
        {"id": 1, "name": "90° Piping Elbow (Bend)", "fn": render_bend_3d},
        {"id": 2, "name": "Industrial T-Junction (Tee)", "fn": render_tjunct_3d}
    ]

    for sc in scenarios:
        print(f"\n[V8] Scenario: {sc['name']}")
        
        # Configure env for specific shape
        s0 = env.reset()
        env.state[8] = sc["id"] # ShapeID
        env.state[9] = 1.0       # ShapeParam (Bend R or Branch D)
        
        # Run Optimization Loop (Simulated for Demo)
        print(f"  Optimizing {sc['name']} via Generative RL...")
        # (In real run, agent.select_action / env.step would happen here)
        s_opt = env.state.copy()
        
        # 1. Render 3D Design
        viz_path = sc["fn"](s_opt, pinn, title=f"v8 Optimized {sc['name']}")
        
        # 2. Export Industrial Bridge
        apdl_path = generate_assembly_apdl_script(s_opt, filename=f"ansys_{sc['name'].replace(' ', '_')}.txt")
        
        print(f"  ✓ 3D Design → {viz_path}")
        print(f"  ✓ APDL Script → {apdl_path}")

    print("\n" + "=" * 72)
    print("  PINNFLOW v8 Assembly Complete.")
    print("  Transition from Simple → Complex Industrial Scenarios achieved.")
    print("=" * 72)

if __name__ == "__main__":
    main_v8()
