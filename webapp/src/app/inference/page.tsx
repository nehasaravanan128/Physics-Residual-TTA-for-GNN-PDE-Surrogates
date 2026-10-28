"use client";
import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { Sliders, AlertCircle, Zap, Activity, CheckCircle2, TrendingDown, Layers } from "lucide-react";
import FlowVisualizer, { StepResult } from "../components/FlowVisualizer";
import FlowVisualizer2D from "../components/FlowVisualizer2D";

const API = "http://localhost:3001";

interface InferResult {
  x_coord: number[];
  steps: StepResult[];
  dataset: string;
}

interface TTAResult {
  x_coord: number[];
  true: number[];
  baseline_predicted: number[];
  adapted_predicted: number[];
  baseline_mse: number;
  adapted_mse: number;
  baseline_relative_mse: number;
  adapted_relative_mse: number;
  improvement_pct: number;
  baseline_physics_loss: number;
  adapted_physics_loss: number;
  physics_loss_history: number[];
  tta_steps: number;
  nu: number;
}

function buildChartData(step: StepResult, xCoord: number[]) {
  return xCoord.map((x, i) => ({
    x: parseFloat(x.toFixed(3)),
    true: parseFloat(step.true[i]?.toFixed(5) ?? "0"),
    predicted: parseFloat(step.predicted[i]?.toFixed(5) ?? "0"),
    error: parseFloat(Math.abs((step.true[i] ?? 0) - (step.predicted[i] ?? 0)).toFixed(6)),
  }));
}

function buildTTAChartData(tta: TTAResult) {
  return tta.x_coord.map((x, i) => ({
    x: parseFloat(x.toFixed(3)),
    true: parseFloat(tta.true[i]?.toFixed(5) ?? "0"),
    baseline: parseFloat(tta.baseline_predicted[i]?.toFixed(5) ?? "0"),
    adapted: parseFloat(tta.adapted_predicted[i]?.toFixed(5) ?? "0"),
    baseline_err: parseFloat(Math.abs((tta.true[i] ?? 0) - (tta.baseline_predicted[i] ?? 0)).toFixed(6)),
    adapted_err: parseFloat(Math.abs((tta.true[i] ?? 0) - (tta.adapted_predicted[i] ?? 0)).toFixed(6)),
  }));
}

// Sample flow preview
function generateSampleFlow(): InferResult {
  const N = 128;
  const x_coord = Array.from({ length: N }, (_, i) => i / (N - 1));
  const numSteps = 6;
  const steps: StepResult[] = [];

  for (let s = 0; s < numSteps; s++) {
    const t = s * 5;
    const trueVals: number[] = [];
    const predVals: number[] = [];

    for (let i = 0; i < N; i++) {
      const x = x_coord[i];
      const shockPos = 0.35 + s * 0.04;
      const steepness = 12 + s * 8;
      const wave = Math.sin(2 * Math.PI * (x - 0.1 * s)) * Math.exp(-0.8 * s * 0.1);
      const shock = 0.5 * (1 - Math.tanh(steepness * (x - shockPos)));
      const base = 0.7 * shock + 0.3 * wave;
      
      trueVals.push(base);
      // Realistic smooth surrogate error (minor smoothing at steep shock front, no artificial noise)
      const shockSmoothDiff = 0.022 * Math.exp(-Math.pow((x - shockPos) * 16, 2));
      predVals.push(base - shockSmoothDiff);
    }

    steps.push({
      t,
      true: trueVals,
      predicted: predVals,
      mse: 0.00012 + s * 0.00008,
      relative_mse: 0.0015 + s * 0.0006,
    });
  }

  return { x_coord, steps, dataset: "burgers" };
}

export default function InferencePage() {
  const [activeMode, setActiveMode] = useState<"tta" | "rollout" | "flow2d">("tta");
  const [dataset, setDataset] = useState("burgers");
  const [dataPath, setDataPath] = useState("./1D_Burgers_Sols_Nu0.001.hdf5");
  const [checkpointPath, setCheckpointPath] = useState("../real_burgers_model.pt");
  const [trajIdx, setTrajIdx] = useState(0);
  const [timestepIdx, setTimestepIdx] = useState(10);
  const [numSteps, setNumSteps] = useState(6);

  // TTA Hyperparameters — 8 steps is optimal for rapid convergence without lag
  const [ttaSteps, setTtaSteps] = useState(8);
  const [ttaLr, setTtaLr] = useState(0.00008);
  const [viscosityNu, setViscosityNu] = useState(0.001);

  const [loading, setLoading] = useState(false);
  const [rolloutResult, setRolloutResult] = useState<InferResult | null>(() => generateSampleFlow());
  const [ttaResult, setTtaResult] = useState<TTAResult | null>(null);
  const [isSampleData, setIsSampleData] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  // Run Rollout
  async function runRollout() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          dataPath,
          checkpointPath,
          trajectoryIdx: trajIdx,
          trajIdx,
          timestepIdx,
          numSteps,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setRolloutResult(data);
      setIsSampleData(false);
      setActiveStep(0);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }

  // Run Test-Time Adaptation
  async function runTTA() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/tta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          dataPath,
          checkpointPath,
          trajectoryIdx: trajIdx,
          trajIdx,
          timestepIdx,
          ttaSteps,
          ttaLr,
          nu: viscosityNu,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setTtaResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "TTA Adaptation failed");
    } finally {
      setLoading(false);
    }
  }

  const currentStep = rolloutResult?.steps[activeStep];
  const chartData = currentStep && rolloutResult ? buildChartData(currentStep, rolloutResult.x_coord) : [];
  const ttaChartData = ttaResult ? buildTTAChartData(ttaResult) : [];

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Physics-Residual TTA &amp; Surrogate Inference</h2>
            <p>Unsupervised adaptation using PDE residuals and physical continuum flow dynamics.</p>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: "flex", gap: 3, background: "var(--bg-card-subtle)", padding: 3, borderRadius: 5, border: "1px solid var(--border)" }}>
            <button
              id="mode-tab-tta"
              onClick={() => setActiveMode("tta")}
              className={`btn-subtle ${activeMode === "tta" ? "active" : ""}`}
              style={{
                padding: "6px 14px",
                fontSize: 11.5,
                fontWeight: activeMode === "tta" ? 600 : 400,
                background: activeMode === "tta" ? "#FFFFFF" : "transparent",
                boxShadow: activeMode === "tta" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                cursor: "pointer",
              }}
            >
              <Zap size={13} style={{ marginRight: 5, color: "var(--accent-terracotta)" }} />
              Physics-Residual TTA
            </button>
            <button
              id="mode-tab-rollout"
              onClick={() => setActiveMode("rollout")}
              className={`btn-subtle ${activeMode === "rollout" ? "active" : ""}`}
              style={{
                padding: "6px 14px",
                fontSize: 11.5,
                fontWeight: activeMode === "rollout" ? 600 : 400,
                background: activeMode === "rollout" ? "#FFFFFF" : "transparent",
                boxShadow: activeMode === "rollout" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                cursor: "pointer",
              }}
            >
              <Activity size={13} style={{ marginRight: 5, color: "var(--accent-sage)" }} />
              Continuum Rollout
            </button>
            <button
              id="mode-tab-flow2d"
              onClick={() => setActiveMode("flow2d")}
              className={`btn-subtle ${activeMode === "flow2d" ? "active" : ""}`}
              style={{
                padding: "6px 14px",
                fontSize: 11.5,
                fontWeight: activeMode === "flow2d" ? 600 : 400,
                background: activeMode === "flow2d" ? "#FFFFFF" : "transparent",
                boxShadow: activeMode === "flow2d" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                cursor: "pointer",
              }}
            >
              <Layers size={13} style={{ marginRight: 5, color: "var(--accent-amber)" }} />
              2D Obstacle Flow Visualizer
            </button>
          </div>
        </div>
      </div>

      <div className="page-body">
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Simulation Config Panel */}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <h3 style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                <Sliders size={15} color="var(--accent-terracotta)" />
                {activeMode === "tta" ? "Adaptation Parameters & Geometry" : "Rollout Horizon & Initial State"}
              </h3>

              {activeMode === "tta" && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <button
                    type="button"
                    className="btn-subtle"
                    onClick={() => {
                      setDataPath("./1D_Burgers_Sols_Nu0.001.hdf5");
                      setTrajIdx(0);
                      setTimestepIdx(10);
                      setViscosityNu(0.001);
                      setTtaSteps(12);
                    }}
                    style={{ fontSize: 10.5 }}
                  >
                    Preset: In-Dist (ν=0.001)
                  </button>
                  <button
                    type="button"
                    className="btn-subtle"
                    onClick={() => {
                      setDataPath("./1D_Burgers_Sols_Nu0.001.hdf5");
                      setTrajIdx(3333);
                      setTimestepIdx(15);
                      setViscosityNu(0.002);
                      setTtaSteps(12);
                    }}
                    style={{ fontSize: 10.5 }}
                  >
                    Preset: Parameter Shift (ν=0.002)
                  </button>
                  <button
                    type="button"
                    className="btn-subtle"
                    onClick={() => {
                      setDataPath("./datasets/burgers_1d.hdf5");
                      setTrajIdx(0);
                      setTimestepIdx(10);
                      setViscosityNu(0.001);
                      setTtaSteps(10);
                    }}
                    style={{ fontSize: 10.5 }}
                  >
                    Preset: Fast Synthetic (N=256)
                  </button>
                </div>
              )}
            </div>

            <div className="grid-4" style={{ gap: 12, marginBottom: 14 }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Trajectory (Sample)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  value={trajIdx}
                  onChange={(e) => setTrajIdx(Number(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Timestep (t₀)</label>
                <input
                  className="form-input"
                  type="number"
                  min={0}
                  value={timestepIdx}
                  onChange={(e) => setTimestepIdx(Number(e.target.value))}
                />
              </div>

              {activeMode === "tta" ? (
                <>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <label className="form-label">TTA Step Budget</label>
                      <div style={{ display: "flex", gap: 3 }}>
                        {[
                          { label: "4 Turbo", val: 4 },
                          { label: "8 Fast", val: 8 },
                          { label: "12 Deep", val: 12 },
                        ].map((b) => (
                          <button
                            key={b.val}
                            type="button"
                            onClick={() => setTtaSteps(b.val)}
                            className={`btn-subtle ${ttaSteps === b.val ? "active" : ""}`}
                            style={{ fontSize: 9.5, padding: "1px 5px", lineHeight: "14px" }}
                          >
                            {b.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <input
                      className="form-input"
                      type="number"
                      min={1}
                      max={30}
                      value={ttaSteps}
                      onChange={(e) => setTtaSteps(Number(e.target.value))}
                    />
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Kinematic Viscosity (ν)</label>
                    <input
                      className="form-input"
                      type="number"
                      step="0.0005"
                      value={viscosityNu}
                      onChange={(e) => setViscosityNu(Number(e.target.value))}
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Rollout Horizon (Steps)</label>
                    <input
                      className="form-input"
                      type="number"
                      min={1}
                      max={20}
                      value={numSteps}
                      onChange={(e) => setNumSteps(Number(e.target.value))}
                    />
                  </div>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label">Dataset</label>
                    <select className="form-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                      <option value="burgers">Burgers' Equation (1D)</option>
                      <option value="cylinder">CylinderFlow (2D)</option>
                    </select>
                  </div>
                </>
              )}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
              <div style={{ fontSize: 11.5, color: "var(--text-muted)", fontFamily: "'JetBrains Mono', monospace", display: "flex", gap: 16, flexWrap: "wrap" }}>
                <span>Dataset: <span style={{ color: "var(--text-primary)" }}>{dataPath}</span></span>
                <span>Weights: <span style={{ color: "var(--text-primary)" }}>{checkpointPath}</span></span>
              </div>

              {activeMode === "tta" ? (
                <button
                  className="btn btn-primary"
                  onClick={runTTA}
                  disabled={loading}
                  style={{ padding: "8px 20px" }}
                >
                  {loading ? (
                    <>
                      <span className="spinner animate-spin" />
                      Optimizing via Physics Residual...
                    </>
                  ) : (
                    <>
                      <Zap size={13} />
                      Run Physics-Residual TTA
                    </>
                  )}
                </button>
              ) : (
                <button
                  className="btn btn-primary"
                  onClick={runRollout}
                  disabled={loading}
                  style={{ padding: "8px 20px" }}
                >
                  {loading ? (
                    <>
                      <span className="spinner animate-spin" />
                      Computing Rollout...
                    </>
                  ) : (
                    <>
                      <Activity size={13} />
                      Run Surrogate Rollout
                    </>
                  )}
                </button>
              )}
            </div>

            {error && (
              <div
                style={{
                  marginTop: 12,
                  padding: "9px 12px",
                  borderRadius: "var(--radius-sm)",
                  background: "#FBF0EF",
                  border: "1px solid #ECCBC8",
                  fontSize: 12.5,
                  color: "var(--accent-brick)",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <AlertCircle size={15} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* MODE 0: 2D OBSTACLE FLOW VISUALIZER */}
          {activeMode === "flow2d" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
              <FlowVisualizer2D />
            </div>
          )}

          {/* MODE 1: TEST-TIME ADAPTATION RESULTS */}
          {activeMode === "tta" && (
            <div>
              {ttaResult ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                  {/* Executive Metric Row */}
                  <div className="grid-4">
                    <div className="metric-card dark">
                      <div className="card-title">Pre-TTA Baseline MSE</div>
                      <div className="card-value">{ttaResult.baseline_mse.toExponential(3)}</div>
                      <div className="card-sub">Rel MSE: {ttaResult.baseline_relative_mse.toExponential(3)}</div>
                    </div>
                    <div className="metric-card terracotta">
                      <div className="card-title">Post-TTA Adapted MSE</div>
                      <div className="card-value">{ttaResult.adapted_mse.toExponential(3)}</div>
                      <div className="card-sub">Rel MSE: {ttaResult.adapted_relative_mse.toExponential(3)}</div>
                    </div>
                    <div className="metric-card sage">
                      <div className="card-title">Adaptation Accuracy Gain</div>
                      <div className="card-value" style={{ color: "var(--accent-sage)" }}>
                        {ttaResult.improvement_pct > 0 ? `+${ttaResult.improvement_pct.toFixed(2)}%` : "0.0%"}
                      </div>
                      <div className="card-sub">Error drop without labels</div>
                    </div>
                    <div className="metric-card amber">
                      <div className="card-title">Physics Residual Loss</div>
                      <div className="card-value">
                        {ttaResult.baseline_physics_loss.toFixed(1)} → {ttaResult.adapted_physics_loss.toFixed(1)}
                      </div>
                      <div className="card-sub">PDE residual minimization</div>
                    </div>
                  </div>

                  {/* Visual Comparison: True vs Baseline vs Adapted */}
                  <div className="grid-2" style={{ gap: 18 }}>
                    {/* Velocity Field Slice */}
                    <div className="chart-container">
                      <div className="chart-title">
                        <span>Physical Velocity Field Comparison (t={timestepIdx})</span>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={ttaChartData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#EBE5DC" />
                          <XAxis dataKey="x" tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <YAxis tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <Tooltip
                            contentStyle={{
                              background: "#FFFFFF",
                              border: "1px solid #E5E0D8",
                              borderRadius: 4,
                              fontSize: 11,
                              boxShadow: "0 2px 6px rgba(0,0,0,0.04)",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Line
                            type="monotone"
                            dataKey="true"
                            stroke="#2E6343"
                            strokeWidth={2.4}
                            dot={false}
                            name="Ground Truth (u)"
                          />
                          <Line
                            type="monotone"
                            dataKey="baseline"
                            stroke="#8C8275"
                            strokeWidth={1.8}
                            strokeDasharray="4 3"
                            dot={false}
                            name="Baseline Surrogate (Pre-TTA)"
                          />
                          <Line
                            type="monotone"
                            dataKey="adapted"
                            stroke="#B85226"
                            strokeWidth={2.2}
                            dot={false}
                            name="Adapted State (Post-TTA)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Pointwise Absolute Error */}
                    <div className="chart-container">
                      <div className="chart-title">
                        <span>Pointwise Error Delta |u_pred − u_true|</span>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={ttaChartData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#EBE5DC" />
                          <XAxis dataKey="x" tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <YAxis tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <Tooltip
                            contentStyle={{
                              background: "#FFFFFF",
                              border: "1px solid #E5E0D8",
                              borderRadius: 4,
                              fontSize: 11,
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Line
                            type="monotone"
                            dataKey="baseline_err"
                            stroke="#8C8275"
                            strokeWidth={1.8}
                            strokeDasharray="4 2"
                            dot={false}
                            name="Baseline Error"
                          />
                          <Line
                            type="monotone"
                            dataKey="adapted_err"
                            stroke="#B53526"
                            strokeWidth={2.0}
                            dot={false}
                            name="Adapted Error (TTA)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Physics Loss Convergence Chart */}
                  <div className="chart-container">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <div className="chart-title" style={{ marginBottom: 0 }}>
                        <TrendingDown size={15} color="var(--accent-terracotta)" />
                        Unsupervised PDE Residual Loss Convergence Trajectory
                      </div>
                      <span className="badge badge-gray" style={{ fontSize: 10 }}>
                        Loss: ||∂u/∂t + u ∂u/∂x - ν ∂²u/∂x²||²
                      </span>
                    </div>

                    <ResponsiveContainer width="100%" height={150}>
                      <LineChart
                        data={ttaResult.physics_loss_history.map((l, idx) => ({ step: idx, loss: l }))}
                        margin={{ top: 8, right: 12, left: -18, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#EBE5DC" />
                        <XAxis dataKey="step" tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} label={{ value: "TTA Step", position: "insideBottom", offset: -5, fontSize: 10, fill: "#8F8274" }} />
                        <YAxis tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                        <Tooltip contentStyle={{ background: "#FFFFFF", border: "1px solid #E5E0D8", borderRadius: 4, fontSize: 11 }} />
                        <Line type="monotone" dataKey="loss" stroke="#B85226" strokeWidth={2.0} dot={{ r: 2.5 }} name="Physics Residual Loss" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div className="card">
                  <div className="empty-state">
                    <Zap size={32} color="var(--text-muted)" />
                    <p style={{ fontSize: 13 }}>Click &ldquo;Run Physics-Residual TTA&rdquo; to adapt model weights on sample #{trajIdx} using unsupervised PDE residuals.</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* MODE 2: ROLLOUT & FLUID VISUALIZER */}
          {activeMode === "rollout" && rolloutResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
              <FlowVisualizer
                xCoord={rolloutResult.x_coord}
                steps={rolloutResult.steps}
                activeStep={activeStep}
                onStepChange={(idx) => setActiveStep(idx)}
                title={isSampleData ? "Eulerian Fluid Velocity Preview (Burgers' Equation)" : "Surrogate Flow Field Prediction"}
              />

              {currentStep && (
                <div className="grid-2" style={{ gap: 18 }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <div className="grid-2" style={{ gap: 12 }}>
                      <div className="metric-card sage">
                        <div className="card-title">Step MSE (t={currentStep.t})</div>
                        <div className="card-value">{currentStep.mse.toExponential(3)}</div>
                        <div className="card-sub">L2 error across mesh nodes</div>
                      </div>
                      <div className="metric-card terracotta">
                        <div className="card-title">Relative MSE</div>
                        <div className="card-value">{currentStep.relative_mse.toExponential(3)}</div>
                        <div className="card-sub">Normalized by kinetic energy</div>
                      </div>
                    </div>

                    <div className="chart-container">
                      <div className="chart-title">Pointwise Error |u_pred − u_true|</div>
                      <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#EBE5DC" />
                          <XAxis dataKey="x" tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <YAxis tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                          <Tooltip contentStyle={{ background: "#FFFFFF", border: "1px solid #E5E0D8", borderRadius: 4, fontSize: 11 }} />
                          <Line type="monotone" dataKey="error" stroke="#B53526" strokeWidth={1.8} dot={false} name="Absolute Error" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="chart-container">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                      <div className="chart-title" style={{ marginBottom: 0 }}>
                        Velocity Profile Slice (t={currentStep.t})
                      </div>
                      <div className="step-nav" style={{ marginBottom: 0 }}>
                        {rolloutResult.steps.map((s, i) => (
                          <button
                            key={i}
                            className={`step-btn ${activeStep === i ? "active" : ""}`}
                            onClick={() => setActiveStep(i)}
                          >
                            t={s.t}
                          </button>
                        ))}
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={chartData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#EBE5DC" />
                        <XAxis dataKey="x" tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                        <YAxis tick={{ fontSize: 9.5, fill: "#8F8274", fontFamily: "JetBrains Mono" }} />
                        <Tooltip contentStyle={{ background: "#FFFFFF", border: "1px solid #E5E0D8", borderRadius: 4, fontSize: 11 }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="true" stroke="#2E6343" strokeWidth={2.4} dot={false} name="True State (u)" />
                        <Line type="monotone" dataKey="predicted" stroke="#B85226" strokeWidth={2.0} dot={false} strokeDasharray="4 3" name="Surrogate State (û)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
