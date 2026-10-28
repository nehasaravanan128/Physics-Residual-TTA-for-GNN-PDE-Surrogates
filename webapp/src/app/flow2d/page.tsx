"use client";
import React, { useState } from "react";
import FlowVisualizer2D from "../components/FlowVisualizer2D";
import { Layers, Activity, Zap, ShieldCheck, ExternalLink, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function Flow2DPage() {
  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span className="badge badge-terracotta" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                <Layers size={13} />
                2D Navier-Stokes Dynamics
              </span>
              <span className="badge badge-sage">5 Bluff Body Geometries</span>
            </div>
            <h2>2D Obstacle Flow &amp; Physics-Residual TTA</h2>
            <p>
              Simulate fluid dynamics around complex obstacle bodies (Cylinders, Airfoils, Steps) and inspect how Physics-Residual Test-Time Adaptation restores incompressibility (∇ · u = 0).
            </p>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Link href="/inference" className="btn-subtle" style={{ fontSize: 11.5 }}>
              Switch to 1D Burgers
            </Link>
            <Link href="/viva" className="btn-subtle" style={{ fontSize: 11.5, color: "var(--accent-terracotta)", fontWeight: 600 }}>
              Viva Defense Q&amp;A
            </Link>
          </div>
        </div>
      </div>

      <div className="page-body">
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Quick Metrics */}
          <div className="grid-4" style={{ gap: 12 }}>
            <div className="metric-card terracotta">
              <div className="card-title">Cylinder Error</div>
              <div className="card-value">0.32%</div>
              <div className="card-sub">Trained FNO-2D, mean of 3 test trajectories</div>
            </div>
            <div className="metric-card sage">
              <div className="card-title">Airfoil Error</div>
              <div className="card-value">0.11%</div>
              <div className="card-sub">Trained FNO-2D, NACA 0012</div>
            </div>
            <div className="metric-card amber">
              <div className="card-title">Square Bluff Body Error</div>
              <div className="card-value">0.08%</div>
              <div className="card-sub">Trained FNO-2D vs 0.78% copy-previous</div>
            </div>
            <div className="metric-card dark">
              <div className="card-title">TTA on Obstacle Flows</div>
              <div className="card-value">Not helpful</div>
              <div className="card-sub">Residual drops 3–55%, error rises</div>
            </div>
          </div>

          {/* Interactive 2D Flow Canvas & Controls */}
          <FlowVisualizer2D />

          {/* Educational / Architectural Context */}
          <div className="card">
            <h3 style={{ fontSize: 14.5, fontWeight: 700, color: "var(--text-primary)", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
              <Zap size={16} color="var(--accent-terracotta)" />
              How 2D Physics-Residual Test-Time Adaptation Operates
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, fontSize: 12.8, color: "var(--text-secondary)", lineHeight: 1.6 }}>
              <div style={{ background: "var(--bg-card-subtle)", padding: 14, borderRadius: "var(--radius-sm)" }}>
                <strong style={{ color: "var(--text-primary)", display: "block", marginBottom: 4 }}>
                  1. Zero Ground-Truth Supervision
                </strong>
                Standard surrogates fail when evaluated past new obstacle shapes. At test time, TTA optimizes the model&apos;s output projection solely against the incompressibility condition:
                <code style={{ display: "block", marginTop: 6, padding: "4px 8px", background: "#1C1917", color: "#FDFCF9", borderRadius: 3, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  L_div = || ∂u_x/∂x + ∂u_y/∂y ||²
                </code>
              </div>

              <div style={{ background: "var(--bg-card-subtle)", padding: 14, borderRadius: "var(--radius-sm)" }}>
                <strong style={{ color: "var(--text-primary)", display: "block", marginBottom: 4 }}>
                  2. Obstacle Boundary-Layer Masking
                </strong>
                The physical obstacle silhouette (circular cylinder, NACA airfoil, or square bluff body) enforces a strict no-slip Dirichlet condition:
                <code style={{ display: "block", marginTop: 6, padding: "4px 8px", background: "#1C1917", color: "#FDFCF9", borderRadius: 3, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  u(x, y) = 0  ∀ (x, y) ∈ Obstacle
                </code>
              </div>

              <div style={{ background: "var(--bg-card-subtle)", padding: 14, borderRadius: "var(--radius-sm)" }}>
                <strong style={{ color: "var(--text-primary)", display: "block", marginBottom: 4 }}>
                  3. Manifold Anchor Regularization
                </strong>
                To prevent unphysical drift, adaptation is anchored to the surrogate&apos;s prior prediction with weight λ = 0.08:
                <code style={{ display: "block", marginTop: 6, padding: "4px 8px", background: "#1C1917", color: "#FDFCF9", borderRadius: 3, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                  L_total = L_div + λ * ||û - û_0||²
                </code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
