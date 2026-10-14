"use client";
import { useState, useEffect } from "react";
import {
  BarChart2,
  Play,
  AlertCircle,
  CheckCircle,
  Sliders,
  Network,
  Cpu,
  Layers,
  ArrowRight,
  TrendingDown,
  RefreshCw,
  Sparkles,
} from "lucide-react";

const API = "http://localhost:3001";

interface Metrics {
  mse: number;
  relative_mse: number;
  mae: number;
  max_error: number;
  num_predictions: number;
  checkpoint: string;
  dataset: string;
  device: string;
}

interface GeometricShiftData {
  name: string;
  num_nodes: number;
  mean_baseline_mse: number;
  mean_adapted_mse: number;
  mean_baseline_rel_mse: number;
  mean_adapted_rel_mse: number;
  error_reduction_pct: number;
  mean_residual_drop_pct: number;
  samples: Array<{
    trajectory: number;
    timestep: number;
    baseline_mse: number;
    adapted_mse: number;
    improvement_pct: number;
    residual_drop_pct: number;
  }>;
}

const metricDefs = [
  { key: "mse", label: "MSE", color: "sage", desc: "Mean Squared Error across field nodes" },
  { key: "relative_mse", label: "Relative MSE", color: "terracotta", desc: "Scale-invariant L2 error" },
  { key: "mae", label: "MAE", color: "amber", desc: "Mean Absolute Error magnitude" },
  { key: "max_error", label: "Max Error", color: "dark", desc: "Worst-case localized deviation" },
];

export default function EvaluatePage() {
  const [activeTab, setActiveTab] = useState<"standard" | "geometric">("geometric");

  // Standard evaluation state
  const [dataset, setDataset] = useState("burgers");
  const [dataPath, setDataPath] = useState("./1D_Burgers_Sols_Nu0.001.hdf5");
  const [checkpointPath, setCheckpointPath] = useState("../real_burgers_model.pt");
  const [metaPath, setMetaPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Geometric shift benchmark state
  const [geoResults, setGeoResults] = useState<Record<string, GeometricShiftData> | null>(null);
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);
  const [geoSamples, setGeoSamples] = useState(5);
  const [selectedShift, setSelectedShift] = useState<string>("jitter");

  // Load cached geometric benchmark on mount
  useEffect(() => {
    async function loadCachedGeo() {
      try {
        const res = await fetch(`${API}/api/evaluate-geometric/cached`);
        const data = await res.json();
        if (data.success && data.results) {
          setGeoResults(data.results);
        }
      } catch {
        // Fallback or server offline
      }
    }
    loadCachedGeo();
  }, []);

  async function runEvaluate() {
    setLoading(true);
    setError(null);
    setMetrics(null);
    try {
      const res = await fetch(`${API}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, dataPath, checkpointPath, metaPath: metaPath || undefined }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setMetrics(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  }

  async function runGeometricBenchmark() {
    setGeoLoading(true);
    setGeoError(null);
    try {
      const res = await fetch(`${API}/api/evaluate-geometric`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataPath: "files (5)/1D_Burgers_Sols_Nu0.001.hdf5",
          checkpointPath: "real_burgers_model.pt",
          numSamples: geoSamples,
          ttaSteps: 10,
          ttaLr: 0.0001,
          shiftType: "all",
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      if (data.success && data.results) {
        setGeoResults(data.results);
      }
    } catch (e: unknown) {
      setGeoError(e instanceof Error ? e.message : "Geometric benchmark failed");
    } finally {
      setGeoLoading(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Model Verification &amp; Evaluation</h2>
            <p>Compute formal PDE residual metrics across held-out test datasets, parameter shifts, and geometric distribution shifts.</p>
          </div>
          
          {/* Tab Switcher */}
          <div style={{ display: "flex", gap: 3, background: "var(--bg-card-subtle)", padding: 3, borderRadius: 5, border: "1px solid var(--border)" }}>
            <button
              onClick={() => setActiveTab("geometric")}
              className={`btn-subtle ${activeTab === "geometric" ? "active" : ""}`}
              style={{
                padding: "6px 14px",
                fontSize: 11.5,
                fontWeight: activeTab === "geometric" ? 600 : 400,
                background: activeTab === "geometric" ? "#FFFFFF" : "transparent",
                boxShadow: activeTab === "geometric" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                cursor: "pointer",
              }}
            >
              <Network size={13} style={{ marginRight: 6, color: "var(--accent-terracotta)" }} />
              Geometric Distribution Shift
            </button>
            <button
              onClick={() => setActiveTab("standard")}
              className={`btn-subtle ${activeTab === "standard" ? "active" : ""}`}
              style={{
                padding: "6px 14px",
                fontSize: 11.5,
                fontWeight: activeTab === "standard" ? 600 : 400,
                background: activeTab === "standard" ? "#FFFFFF" : "transparent",
                boxShadow: activeTab === "standard" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                cursor: "pointer",
              }}
            >
              <BarChart2 size={13} style={{ marginRight: 6, color: "var(--accent-sage)" }} />
              Standard Checkpoint Metrics
            </button>
          </div>
        </div>
      </div>

      <div className="page-body">
        {/* TAB 1: GEOMETRIC DISTRIBUTION SHIFT */}
        {activeTab === "geometric" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Context Callout */}
            <div
              style={{
                background: "linear-gradient(135deg, #FCFAF7 0%, #F5EFEB 100%)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "16px 20px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                    <Sparkles size={16} color="var(--accent-terracotta)" />
                    Research Evaluation: Unsupervised TTA Under Geometric Mesh Shift
                  </div>
                  <p style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 4, maxWidth: 860, lineHeight: 1.5 }}>
                    When deployed on unseen non-uniform spatial discretizations, GNN message-passing edge vectors (&Delta;x) undergo severe out-of-distribution shifts. 
                    This benchmark evaluates how <strong>Physics-Residual TTA</strong> adapts the decoder parameters &theta;<sub>dec</sub> on non-uniform geometric meshes without ground-truth labels.
                  </p>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <label style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "'JetBrains Mono', monospace" }}>Samples:</label>
                    <select
                      className="form-select"
                      value={geoSamples}
                      onChange={(e) => setGeoSamples(Number(e.target.value))}
                      style={{ padding: "4px 8px", fontSize: 11, height: 28 }}
                    >
                      <option value={5}>5 Trajectories</option>
                      <option value={10}>10 Trajectories</option>
                    </select>
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={runGeometricBenchmark}
                    disabled={geoLoading}
                    style={{ padding: "6px 14px", fontSize: 11.5 }}
                  >
                    {geoLoading ? (
                      <>
                        <RefreshCw size={12} className="animate-spin" />
                        Adapting On Shifted Meshes...
                      </>
                    ) : (
                      <>
                        <Play size={12} />
                        Run Geometric Benchmark
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {geoError && (
              <div style={{ padding: "10px 14px", borderRadius: "var(--radius-sm)", background: "#FBF0EF", border: "1px solid #ECCBC8", fontSize: 12.5, color: "var(--accent-brick)", display: "flex", alignItems: "center", gap: 8 }}>
                <AlertCircle size={15} />
                <span>{geoError}</span>
              </div>
            )}

            {/* 3 Geometric Shift Cards */}
            {geoResults ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                <div className="grid-3" style={{ gap: 14 }}>
                  {Object.entries(geoResults).map(([key, data]) => {
                    const isSelected = selectedShift === key;
                    const gainColor = data.error_reduction_pct > 0 ? "var(--accent-sage)" : "var(--accent-brick)";
                    return (
                      <div
                        key={key}
                        onClick={() => setSelectedShift(key)}
                        className="card"
                        style={{
                          cursor: "pointer",
                          borderColor: isSelected ? "var(--accent-terracotta)" : "var(--border)",
                          boxShadow: isSelected ? "0 0 0 1px var(--accent-terracotta), 0 4px 12px rgba(184, 82, 38, 0.08)" : "none",
                          transition: "all 0.15s ease",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                          <span className="badge badge-terracotta" style={{ fontSize: 10, textTransform: "uppercase" }}>
                            {key === "jitter" ? "Irregular Mesh" : key === "warped" ? "Boundary Layer" : "Scale Shift"}
                          </span>
                          <span style={{ fontSize: 11, fontWeight: 700, color: gainColor, fontFamily: "'JetBrains Mono', monospace" }}>
                            +{data.error_reduction_pct.toFixed(1)}% Gain
                          </span>
                        </div>

                        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8, lineHeight: 1.3 }}>
                          {data.name}
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border-subtle)" }}>
                          <div>
                            <div style={{ fontSize: 9.5, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Base MSE</div>
                            <div style={{ fontSize: 13.5, fontWeight: 600, color: "#8C8275", fontFamily: "'JetBrains Mono', monospace" }}>
                              {data.mean_baseline_mse.toExponential(2)}
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 9.5, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Adapted MSE</div>
                            <div style={{ fontSize: 13.5, fontWeight: 700, color: "var(--accent-terracotta)", fontFamily: "'JetBrains Mono', monospace" }}>
                              {data.mean_adapted_mse.toExponential(2)}
                            </div>
                          </div>
                        </div>

                        <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
                          <TrendingDown size={12} color="var(--accent-sage)" />
                          <span>Residual Drop: <strong>-{data.mean_residual_drop_pct.toFixed(1)}%</strong></span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Selected Shift Sample Breakdown Table */}
                {geoResults[selectedShift] && (
                  <div className="card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                      <div>
                        <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>
                          Detailed Per-Trajectory Adaptation Breakdown
                        </h3>
                        <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
                          {geoResults[selectedShift].name} · {geoResults[selectedShift].num_nodes} Mesh Nodes
                        </p>
                      </div>

                      <span className="badge badge-sage" style={{ fontSize: 11 }}>
                        Mean Accuracy Gain: +{geoResults[selectedShift].error_reduction_pct.toFixed(2)}%
                      </span>
                    </div>

                    <div style={{ overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                        <thead>
                          <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left", color: "var(--text-muted)", fontSize: 11, textTransform: "uppercase" }}>
                            <th style={{ padding: "8px 12px" }}>Sample Trajectory</th>
                            <th style={{ padding: "8px 12px" }}>Timestep (t)</th>
                            <th style={{ padding: "8px 12px" }}>Baseline MSE (Pre-TTA)</th>
                            <th style={{ padding: "8px 12px" }}>Adapted MSE (Post-TTA)</th>
                            <th style={{ padding: "8px 12px" }}>PDE Residual Drop</th>
                            <th style={{ padding: "8px 12px" }}>Accuracy Gain</th>
                          </tr>
                        </thead>
                        <tbody>
                          {geoResults[selectedShift].samples.map((s, idx) => {
                            const isPositive = s.improvement_pct >= 0;
                            return (
                              <tr key={idx} style={{ borderBottom: "1px solid var(--border-subtle)", fontFamily: "'JetBrains Mono', monospace" }}>
                                <td style={{ padding: "8px 12px", color: "var(--text-primary)", fontWeight: 600 }}>Trajectory #{s.trajectory}</td>
                                <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>t = {s.timestep}</td>
                                <td style={{ padding: "8px 12px", color: "#8C8275" }}>{s.baseline_mse.toExponential(3)}</td>
                                <td style={{ padding: "8px 12px", color: "var(--accent-terracotta)", fontWeight: 600 }}>{s.adapted_mse.toExponential(3)}</td>
                                <td style={{ padding: "8px 12px", color: "var(--accent-sage)" }}>-{s.residual_drop_pct.toFixed(1)}%</td>
                                <td style={{ padding: "8px 12px" }}>
                                  <span
                                    style={{
                                      display: "inline-block",
                                      padding: "2px 8px",
                                      borderRadius: 4,
                                      fontSize: 11,
                                      fontWeight: 600,
                                      background: isPositive ? "#EBF5EE" : "#FBF0EF",
                                      color: isPositive ? "#2E6343" : "#B53526",
                                    }}
                                  >
                                    {isPositive ? `+${s.improvement_pct.toFixed(2)}%` : `${s.improvement_pct.toFixed(2)}%`}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="card">
                <div className="empty-state">
                  <Network size={36} color="var(--sand-400)" />
                  <p>Click &ldquo;Run Geometric Benchmark&rdquo; to evaluate MeshGraphNet vs Physics-Residual TTA on irregular, warped, and scale-shifted meshes.</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: STANDARD EVALUATION */}
        {activeTab === "standard" && (
          <div className="grid-2" style={{ gap: 24, alignItems: "start" }}>
            {/* Config card */}
            <div className="card">
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <Sliders size={16} color="var(--accent-terracotta)" />
                Standard Checkpoint Evaluation
              </h3>

              <div className="form-group">
                <label className="form-label">Dataset</label>
                <select className="form-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  <option value="burgers">Burgers' Equation (1D)</option>
                  <option value="cylinder">CylinderFlow (2D)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Test HDF5 Path</label>
                <input className="form-input" value={dataPath} onChange={(e) => setDataPath(e.target.value)} />
              </div>

              <div className="form-group">
                <label className="form-label">Checkpoint Path</label>
                <input className="form-input" value={checkpointPath} onChange={(e) => setCheckpointPath(e.target.value)} />
              </div>

              {dataset === "cylinder" && (
                <div className="form-group">
                  <label className="form-label">Meta JSON Path</label>
                  <input className="form-input" value={metaPath} onChange={(e) => setMetaPath(e.target.value)} />
                </div>
              )}

              <button
                className="btn btn-primary"
                onClick={runEvaluate}
                disabled={loading}
                style={{ width: "100%", justifyContent: "center", marginTop: 8 }}
              >
                {loading ? (
                  <>
                    <span className="spinner animate-spin" />
                    Computing Benchmark Metrics...
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    Run Benchmark
                  </>
                )}
              </button>

              {error && (
                <div
                  style={{
                    marginTop: 16,
                    padding: "10px 14px",
                    borderRadius: "var(--radius-sm)",
                    background: "#FBF0EF",
                    border: "1px solid #ECCBC8",
                    fontSize: 13,
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

              {/* Explanation box */}
              <div style={{ marginTop: 22, padding: "14px 16px", borderRadius: "var(--radius-sm)", background: "var(--sand-100)", border: "1px solid var(--sand-200)" }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--sand-600)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
                  Metric Definitions
                </div>
                {metricDefs.map((m) => (
                  <div key={m.key} style={{ display: "flex", gap: 10, marginBottom: 6, fontSize: 12 }}>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: "var(--text-primary)", width: 95 }}>
                      {m.label}
                    </span>
                    <span style={{ color: "var(--text-secondary)" }}>{m.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Results column */}
            <div>
              {metrics ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {/* Status card */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      padding: "14px 18px",
                      borderRadius: "var(--radius)",
                      background: "#F1F7F2",
                      border: "1px solid #D5E7D8",
                    }}
                  >
                    <CheckCircle size={18} color="var(--accent-sage)" />
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--accent-sage)" }}>
                        Evaluation Complete
                      </div>
                      <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 1 }}>
                        {metrics.num_predictions.toLocaleString()} verification samples · Device: {metrics.device}
                      </div>
                    </div>
                  </div>

                  {/* 4 Cards */}
                  <div className="grid-2" style={{ gap: 14 }}>
                    {metricDefs.map((m) => (
                      <div key={m.key} className={`metric-card ${m.color}`}>
                        <div className="card-title">{m.label}</div>
                        <div className="card-value" style={{ fontSize: 22 }}>
                          {(metrics[m.key as keyof Metrics] as number).toExponential(4)}
                        </div>
                        <div className="card-sub">{m.desc}</div>
                      </div>
                    ))}
                  </div>

                  {/* Raw JSON */}
                  <div className="card">
                    <div className="card-title" style={{ marginBottom: 10 }}>Raw Metrics Output</div>
                    <pre
                      style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: 12,
                        color: "#FAF7F2",
                        background: "var(--sand-900)",
                        padding: 14,
                        borderRadius: "var(--radius-sm)",
                        overflow: "auto",
                        border: "1px solid var(--sand-800)",
                      }}
                    >
                      {JSON.stringify(metrics, null, 2)}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="card">
                  <div className="empty-state">
                    <BarChart2 size={36} color="var(--sand-400)" />
                    <p>Click &ldquo;Run Benchmark&rdquo; to evaluate the model checkpoint against test ground truth.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
