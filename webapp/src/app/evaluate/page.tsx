"use client";
import { useState } from "react";
import { BarChart2, Play, AlertCircle, CheckCircle, Sliders } from "lucide-react";

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

const metricDefs = [
  { key: "mse", label: "MSE", color: "sage", desc: "Mean Squared Error across field nodes" },
  { key: "relative_mse", label: "Relative MSE", color: "terracotta", desc: "Scale-invariant L2 error" },
  { key: "mae", label: "MAE", color: "amber", desc: "Mean Absolute Error magnitude" },
  { key: "max_error", label: "Max Error", color: "dark", desc: "Worst-case localized deviation" },
];

export default function EvaluatePage() {
  const [dataset, setDataset] = useState("burgers");
  const [dataPath, setDataPath] = useState("./1D_Burgers_Sols_Nu0.001.hdf5");
  const [checkpointPath, setCheckpointPath] = useState("../real_burgers_model.pt");
  const [metaPath, setMetaPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Model Verification &amp; Evaluation</h2>
            <p>Compute formal PDE residual metrics across held-out test datasets and checkpoints.</p>
          </div>
          <span className="badge badge-amber" style={{ padding: "5px 12px", fontSize: 12 }}>
            Evaluation Protocol: L2 Relative / L∞
          </span>
        </div>
      </div>

      <div className="page-body">
        <div className="grid-2" style={{ gap: 24, alignItems: "start" }}>
          {/* Config card */}
          <div className="card">
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <Sliders size={16} color="var(--accent-terracotta)" />
              Evaluation Benchmark
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
      </div>
    </div>
  );
}
