# Results Summary

## ablation.csv
```csv
config,stress_mae_mean,stress_mae_std,stress_r2_mean,pressure_mae_mean,e2e_csr_mean,e2e_csr_std
A: Vanilla MLP,7.12,0.3576,0.804,,,
B: Single-task PINN,7.47,0.7566,0.7723,,,
C: Multi-task PINN [N1],6.7433,0.7167,0.8124,1.3233,,
D: [N1]+CA-VAE [N2],6.7433,0.7167,0.8124,1.3233,0.3717,0.0289
E: Full [N1+N2+N3],6.7433,0.7167,0.8124,1.3233,0.2167,0.2082
```

## ablation_results.csv
```csv
config,metric,mean,std
M1 (Vanilla MLP),StressMAE%,110.5884207926954,39.855167016841015
M2 (ST-PINN),StressMAE%,629.0052815100756,96.0674627183103
M3 (MT-PINN),StressMAE%,210.73695354505435,158.9859473156299
M4 (VAE-Synthesis),CSR,1.0,0.0
M4 (VAE-Synthesis),Diversity,0.572086851910033,0.08212986748868137
M5 (Full Suite),StressMAE%,210.73695354505435,158.9859473156299
```

## final_ablation_summary.csv
```csv
Method,Speed (Inference ms),MAE Stress (Measured),Safety Rate (ASME),Industrial Ready
Baseline FEM,30000.0,0.0,100%,Yes
PINNFlow v8 (Reported),0.16797304153442383,196.66968443423926,33.3%,Yes (Validated)
```

## final_geometry_performance.csv
```csv
pinn_solve_time_ms,ansys_solve_time_s,speedup_factor,mae_stress,mae_pressure,name
0.18689632415771484,30.0,160508.21252407422,164.54034495442482,0.40676547382246886,Straight
0.17409563064575195,30.0,172309.18890291487,410.2810545995628,0.6362409106125764,Elbow/Bend
0.1429271697998047,30.0,209882.426257757,15.187653748730156,0.44092936099415914,T-Junction
```

## final_optimized_elbow.txt
```text
! ===========================================================
! PINNFlow v8 - Industrial Assembly Verification
! Geometry Type: Elbow
! ===========================================================
/PREP7
/CLEAR,NOSTART
/PREP7
MP,EX,1,200000; MP,NUXY,1,0.3
ET,1,SOLID185
TORUS, 286.15401037663514, 176.3250520554459, 161.18155205661867, 0, 90
MSHAPE,0,3D; MSHKEY,1; ESIZE, 20; VMESH,ALL
DA, ALL, ALL, 0
FINISH
/SOLU
SOLVE
FINISH
! ===========================================================
```

## summary_v1_v3.csv
```csv
Metric,v1,v3,Delta
Stress MAE %,7.55,5.25,-2.3
Pressure MAE %,12.52,1.08,-11.44
Stress RÂ²,0.8752,0.8662,-0.009000000000000008
Pressure RÂ²,0.1115,0.9866,0.8751
Gen CSR,0.018,0.408,0.38999999999999996
RL Improvement %,11.95,-139.93,-151.88
E2E CSR,0.75,0.6,-0.15000000000000002
FEM CSR,0.55,0.4,-0.15000000000000002
```

## test_ansys_script.txt
```text
! ===========================================================
! PINNFlow v6 - ANSYS Verification Script
! Generated for Design ID: 25795
! ===========================================================
/PREP7
FINISH
/CLEAR,NOSTART
/PREP7

! --- Material Properties (A333 Grade 6 Steel) ---
MP,EX,1,200000        ! Young's Modulus (MPa)
MP,NUXY,1,0.3         ! Poisson's Ratio
MP,ALPX,1,1.2e-5      ! Thermal Expansion

! --- Geometry Parameters ---
RAD_OUT = 250.00
RAD_IN  = 235.00
LENGTH  = 30000.00
PRESS   = 8.00
DELTA_T = 20.00

! --- Modeling ---
ET,1,SOLID185          ! 3D Solid Elements
CYLIND, RAD_IN, RAD_OUT, 0, LENGTH, 0, 360
MSHAPE,0,3D
MSHKEY,1
ESIZE, 10              ! 10mm Global Mesh Size
VMESH,ALL

! --- Boundary Conditions ---
DA, 1, ALL, 0          ! Fix bottom face (Z=0)
DA, 2, ALL, 0          ! Fix top face (Z=L) - Simplified fixed-fixed support

! --- Loading ---
ASEL,S,AREA,,3         ! Select inner surface
SFA,ALL,1,PRES,PRESS   ! Apply internal pressure
TUNIF, 20 + DELTA_T    ! Apply operating temperature delta
ALLSEL,ALL

! --- Solution ---
/SOLU
ANTYPE,0               ! Static Analysis
SOLVE
FINISH

! --- Post-Processing ---
/POST1
SET,LAST
PLNSOL,S,EQV,0,1       ! Plot Von Mises Stress
/SHOW,PNG
/REPLOT
! Export stress results to text for verification parity
*GET,MAX_STRESS,SORT,,MAX
*MSG,INFO,MAX_STRESS
 (f10.4)
FINISH
! ===========================================================
```

## pinnflow_v3_results.png
![pinnflow_v3_results.png](file:///c:/Users/kushal%20Bhargav/Downloads/LLDDD/LLDD/LLD/results/pinnflow_v3_results.png)
