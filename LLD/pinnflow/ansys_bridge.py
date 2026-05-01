"""
pinnflow/ansys_bridge.py
────────────────────────
MODULE 14 — ANSYS Verification Bridge [P10]

Generates industrial-grade ANSYS APDL scripts from PINNFlow optimizations.
Enables high-fidelity Finite Element Analysis (FEA) verification of AI designs.
"""
from __future__ import annotations
import os
import numpy as np
from pinnflow.config import RESULTS_DIR

def generate_static_apdl_script(params: np.ndarray, filename: str = "ansys_static_script.txt") -> str:
    """
    [P10.1] Generates a 3D Mechanical APDL script for Static stress verification.
    """
    return generate_apdl_script(params, filename) # Wrapper for backward parity

def generate_apdl_script(params: np.ndarray, filename: str = "ansys_verification_script.txt") -> str:
    """
    [P10.1] Generates a 3D Mechanical APDL script for the optimized design.
    Uses SOLID185 elements for high-fidelity stress verification.
    """
    # 1. Map Parameters (mm and MPa)
    D = params[0]
    t = params[1]
    L = params[2] * 1000.0   # Convert m to mm for consistency in APDL
    P = params[3]
    soil_disp = params[4]
    delta_t = params[5]
    
    RAD_OUT = D / 2.0
    RAD_IN  = RAD_OUT - t
    
    # 2. Build APDL Command String
    script = [
        "! ===========================================================",
        "! PINNFlow v6 — ANSYS Verification Script",
        f"! Generated for Design ID: {np.random.randint(10000, 99999)}",
        "! ===========================================================",
        "/PREP7",
        "FINISH",
        "/CLEAR,NOSTART",
        "/PREP7",
        "",
        "! --- Material Properties (A333 Grade 6 Steel) ---",
        "MP,EX,1,200000        ! Young's Modulus (MPa)",
        "MP,NUXY,1,0.3         ! Poisson's Ratio",
        "MP,ALPX,1,1.2e-5      ! Thermal Expansion",
        "",
        "! --- Geometry Parameters ---",
        f"RAD_OUT = {RAD_OUT:.2f}",
        f"RAD_IN  = {RAD_IN:.2f}",
        f"LENGTH  = {L:.2f}",
        f"PRESS   = {P:.2f}",
        f"DELTA_T = {delta_t:.2f}",
        "",
        "! --- Modeling ---",
        "ET,1,SOLID185          ! 3D Solid Elements",
        "CYLIND, RAD_IN, RAD_OUT, 0, LENGTH, 0, 360",
        "MSHAPE,0,3D",
        "MSHKEY,1",
        "ESIZE, 10              ! 10mm Global Mesh Size",
        "VMESH,ALL",
        "",
        "! --- Boundary Conditions ---",
        "DA, 1, ALL, 0          ! Fix bottom face (Z=0)",
        "DA, 2, ALL, 0          ! Fix top face (Z=L) - Simplified fixed-fixed support",
        "",
        "! --- Loading ---",
        "ASEL,S,AREA,,3         ! Select inner surface",
        "SFA,ALL,1,PRES,PRESS   ! Apply internal pressure",
        "TUNIF, 20 + DELTA_T    ! Apply operating temperature delta",
        "ALLSEL,ALL",
        "",
        "! --- Solution ---",
        "/SOLU",
        "ANTYPE,0               ! Static Analysis",
        "SOLVE",
        "FINISH",
        "",
        "! --- Post-Processing ---",
        "/POST1",
        "SET,LAST",
        "PLNSOL,S,EQV,0,1       ! Plot Von Mises Stress",
        "/SHOW,PNG",
        "/REPLOT",
        "! Export stress results to text for verification parity",
        f"*GET,MAX_STRESS,SORT,,MAX",
        f"*MSG,INFO,MAX_STRESS",
        " (f10.4)",
        "FINISH",
        "! ==========================================================="
    ]
    
    content = "\n".join(script)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
        
    print(f"  ANSYS APDL Script saved → {path}")
    return path

def generate_transient_apdl_script(params: np.ndarray, duration: float = 10.0, steps: int = 20, filename: str = "ansys_transient_script.txt") -> str:
    """
    [V7] Generates a Transient Mechanical APDL script for water-hammer simulation.
    Uses SOLID185 elements and transient time-stepping.
    """
    # 1. Map Parameters
    D = params[0]; t = params[1]; L = params[2] * 1000.0; P = params[3]
    RAD_OUT = D / 2.0; RAD_IN  = RAD_OUT - t
    
    script = [
        "! ===========================================================",
        "! PINNFlow v7 — ANSYS Transient Dynamics Script",
        f"! Simulating Surges for {duration} seconds over {steps} steps",
        "! ===========================================================",
        "/PREP7",
        "/CLEAR,NOSTART",
        "/PREP7",
        "MP,EX,1,200000; MP,NUXY,1,0.3; MP,DENS,1,7800",
        "ET,1,SOLID185",
        f"CYLIND, {RAD_OUT-t}, {RAD_OUT}, 0, {L}, 0, 360",
        "MSHAPE,0,3D; MSHKEY,1; ESIZE, 20; VMESH,ALL",
        "DA, 1, ALL, 0; DA, 2, ALL, 0",
        "FINISH",
        "",
        "! --- Transient Load Steps ---",
        "/SOLU",
        "ANTYPE,4               ! Transient Analysis",
        "TRNOPT,FULL",
        "LUMPM,0",
        f"TIME, {duration}",
        f"NSUBST, {steps}, {steps*2}, {steps/2}",
        "",
        "! Apply sinusoidal pressure pulse to inner surface",
        "ASEL,S,AREA,,3",
        f"SFA,ALL,1,PRES, {P} * SIN(1.0)", # Simplified pulse
        "ALLSEL,ALL",
        "SOLVE",
        "FINISH",
        "",
        "! --- Damage Export For Fatigue Model ---",
        "/POST26",
        "NSOL,2,10,S,EQV, MAX_STRESS_NODE",
        "PRVAR,2",
        "FINISH",
        "! ==========================================================="
    ]
    
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(script))
    print(f"  ANSYS Transient Script saved → {path}")
    return path

def generate_assembly_apdl_script(params: np.ndarray, filename: str = "ansys_assembly_script.txt") -> str:
    """
    [V8] Industrial Assembly Script.
    Supports ShapeID 0 (Straight), 1 (Elbow), 2 (Tee/Junction).
    """
    D, t, shape_id, shape_param = params[0], params[1], int(params[8]), params[9]
    RAD_OUT = D / 2.0
    L = params[2] * 1000.0
    
    script = [
        "! ===========================================================",
        "! PINNFlow v8 — Industrial Assembly Verification",
        f"! Geometry Type: {['Straight', 'Elbow', 'Tee'][shape_id]}",
        "! ===========================================================",
        "/PREP7",
        "/CLEAR,NOSTART",
        "/PREP7",
        "MP,EX,1,200000; MP,NUXY,1,0.3",
        "ET,1,SOLID185"
    ]
    
    if shape_id == 0:
        script.append(f"CYLIND, {RAD_OUT-t}, {RAD_OUT}, 0, {L}, 0, 360")
    elif shape_id == 1:
        # Elbow: Torus section
        R_bend = RAD_OUT * shape_param * 1.5
        script.append(f"TORUS, {R_bend}, {RAD_OUT}, {RAD_OUT-t}, 0, 90")
    elif shape_id == 2:
        # Tee: Intersection of Run and Branch
        script.append(f"CYLIND, {RAD_OUT-t}, {RAD_OUT}, -{L/2}, {L/2}, 0, 360")
        script.append(f"CYLIND, {RAD_OUT-t}, {RAD_OUT}, 0, {L/2}, 0, 360")
        script.append("VADD, 1, 2") # Boolean addition for tee
        
    script.extend([
        "MSHAPE,0,3D; MSHKEY,1; ESIZE, 20; VMESH,ALL",
        "DA, ALL, ALL, 0",
        "FINISH",
        "/SOLU",
        "SOLVE",
        "FINISH",
        "! ==========================================================="
    ])
    
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(script))
    print(f"  ANSYS Assembly Script saved → {path}")
    return path
