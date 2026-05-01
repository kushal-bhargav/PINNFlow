"""
pinnflow/visualization_3d.py
─────────────────────────────
MODULE 13 — 3D Interactive Pipeline Visualization [P9]

Renders a high-fidelity 3D model of the pipeline design using Plotly.
Includes:
- Physical geometry (D, t, L)
- Deformation overlay (Soil Displacement)
- Stress Heatmap (PINN Inference)
"""
from __future__ import annotations
import os
import numpy as np
import plotly.graph_objects as go
from pinnflow.config import RESULTS_DIR, BLUE, RED, GREEN, GOLD

def render_pipe_3d(params: np.ndarray, pinn, title: str = "Pipeline Design Analysis", filename: str = "pipe_design_3d.html") -> str:
    """
    [P9.1] Create an interactive 3D visualization of a single pipe segment.
    params: 8-dim design vector.
    """
    # 1. Unpack params
    # FEAT = [D, t, L, P, soil_disp, delta_T, V, K]
    D = params[0]
    t = params[1]
    L = params[2]
    soil_disp = params[4]
    
    # 2. Generate Mesh
    n_z = 100  # Resolution along length
    n_theta = 40 # Resolution around circumference
    
    z = np.linspace(0, L, n_z)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta, z = np.meshgrid(theta, z)
    
    # Radius (outer)
    R = D / 2.0 / 1000.0 # Convert mm to m for scale
    
    # 3. Apply Deformation (Soil Displacement)
    # Simple parabolic deformation model: y = 4 * d * z * (L - z) / L^2
    # This simulates a bowed pipe due to soil movement.
    y_offset = 4 * (soil_disp / 1000.0) * z * (L - z) / (L ** 2 + 1e-8)
    
    x = R * np.cos(theta)
    y = R * np.sin(theta) + y_offset
    
    # 4. Stress Mapping (PINN)
    # Replicate params for each mesh point to get local local stress? 
    # For now, we use a global prediction but add a slight gradient for visual clarity
    pred = pinn.predict(params.reshape(1, -1))[0]
    base_stress = float(pred[0])
    
    # Add synthetic variation based on z (higher stress in center of bend)
    # In a real 3D PINN, we would evaluate at each point (z, theta)
    stress_mesh = base_stress * (0.9 + 0.2 * np.exp(-((z - L/2)**2) / (L**2/16)))
    
    # 5. Create Plotly Figure
    fig = go.Figure()
    
    # Surface color based on stress
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=stress_mesh,
        colorscale='Viridis',
        colorbar=dict(title="Stress (MPa)", thickness=20, ticklen=5),
        name="Pipeline Surface",
        hovertemplate="Z: %{z:.2f}m<br>Stress: %{surfacecolor:.2f} MPa<extra></extra>"
    ))
    
    # Styling
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br>D={D}mm, t={t}mm, L={L}m, SoilDisp={soil_disp}mm", 
                   x=0.5, font=dict(color=BLUE, size=18)),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Length (m)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        template="plotly_white",
        annotations=[
            dict(
                x=0.05, y=0.05, ax=0, ay=0, xref="paper", yref="paper",
                text=f"Predicted Stress: {base_stress:.2f} MPa<br>Status: {'✅ SAFE' if base_stress < 200 else '❌ VIOLATION'}",
                showarrow=False, font=dict(size=14, color=GREEN if base_stress < 200 else RED),
                bgcolor="white", bordercolor=BLUE, borderwidth=2, borderpad=10
            )
        ]
    )
    
    path = os.path.join(RESULTS_DIR, filename)
    fig.write_html(path)
    print(f"  3D Visualization saved → {path}")
    return path

def render_bend_3d(params: np.ndarray, pinn, title: str = "Piping Elbow Analysis") -> str:
    """
    [V8] Renders a 3D Toroidal Bend (Elbow).
    shape_param at params[9] is the Bend Radius (R/D ratio).
    """
    D, t, L, shape_param = params[0], params[1], params[2], params[9]
    R_pipe = D / 2.0 / 1000.0
    R_bend = R_pipe * shape_param * 1.5 # 1.5D rule default
    
    n_phi, n_theta = 50, 30
    phi = np.linspace(0, np.pi/2, n_phi) # 90 degree bend
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)
    
    # Parametric equations for a torus segment
    x = (R_bend + R_pipe * np.cos(theta)) * np.cos(phi)
    y = (R_bend + R_pipe * np.cos(theta)) * np.sin(phi)
    z = R_pipe * np.sin(theta)
    
    pred = pinn.predict(params.reshape(1, -1))[0]
    base_s = float(pred[0])
    stress_mesh = base_s * (1.0 + 0.2 * np.cos(theta)) # Stress higher on inner curve (intrados)
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=stress_mesh, colorscale='Viridis')])
    fig.update_layout(title=title, scene=dict(aspectmode='data'), template="plotly_white")
    
    path = os.path.join(RESULTS_DIR, "pipe_bend_3d.html")
    fig.write_html(path)
    return path

def render_tjunct_3d(params: np.ndarray, pinn, title: str = "T-Junction Analysis") -> str:
    """
    [V8] Renders a 3D T-Junction (Intersection of 2 cylinders).
    """
    D, L = params[0], params[2]
    R = D / 2.0 / 1000.0
    
    fig = go.Figure()
    
    # 1. Main Run (Z-axis)
    z1 = np.linspace(-L/2, L/2, 50); theta1 = np.linspace(0, 2*np.pi, 30)
    theta1, z1 = np.meshgrid(theta1, z1)
    x1 = R * np.cos(theta1); y1 = R * np.sin(theta1)
    
    # 2. Branch Outlet (X-axis)
    x2 = np.linspace(R, L/2, 30); theta2 = np.linspace(0, 2*np.pi, 30)
    theta2, x2 = np.meshgrid(theta2, x2)
    y2 = R * np.cos(theta2); z2 = R * np.sin(theta2)
    
    fig.add_trace(go.Surface(x=x1, y=y1, z=z1, colorscale='Greys', opacity=0.8, showscale=False))
    fig.add_trace(go.Surface(x=x2, y=y2, z=z2, colorscale='Reds', showscale=False))
    
    fig.update_layout(title=title, scene=dict(aspectmode='data'), template="plotly_white")
    path = os.path.join(RESULTS_DIR, "pipe_tjunct_3d.html")
    fig.write_html(path)
    return path

def render_comparison_3d(baseline_params, optimized_params, pinn, filename: str = "design_comparison_3d.html"):
    """
    [P9.2] Render Comparison: Baseline vs Optimized in a single HTML (subplots).
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=("Baseline Design", "Optimized Design (RL)")
    )
    
    for i, params in enumerate([baseline_params, optimized_params]):
        D, t, L, soil_disp = params[0], params[1], params[2], params[4]
        R = D / 2.0 / 1000.0
        n_z, n_theta = 60, 30
        z = np.linspace(0, L, n_z); theta = np.linspace(0, 2*np.pi, n_theta)
        theta, z = np.meshgrid(theta, z)
        y_off = 4 * (soil_disp / 1000.0) * z * (L - z) / (L ** 2 + 1e-8)
        x = R * np.cos(theta); y = R * np.sin(theta) + y_off
        
        pred = pinn.predict(params.reshape(1, -1))[0]
        base_s = float(pred[0])
        stress_mesh = base_s * (0.9 + 0.2 * np.exp(-((z - L/2)**2) / (L**2/16)))
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=stress_mesh,
            colorscale='Viridis', showscale=(i==1),
            colorbar=dict(title="Stress (MPa)", x=1.05) if i==1 else None
        ), row=1, col=i+1)

    fig.update_layout(
        height=700, width=1200, showlegend=False,
        title_text="<b>Pipeline Design Optimization: Visual Comparison</b>",
        template="plotly_white"
    )
    
    path = os.path.join(RESULTS_DIR, filename)
    fig.write_html(path)
    return path
