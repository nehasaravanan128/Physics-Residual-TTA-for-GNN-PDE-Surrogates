"use client";
import { useState, useRef, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Database, Search, ChevronLeft, ChevronRight, Play, Pause, Waves, Sparkles } from "lucide-react";

const API = "http://localhost:3001";

interface DatasetInfo {
  dataset: string;
  num_trajectories: number;
  num_timesteps: number;
  num_grid_points: number;
  x_min: number;
  x_max: number;
  file_size_mb: number;
  sample_trajectory: {
    index: number;
    frames: { t: number; values: number[] }[];
  };
}

export default function ExplorePage() {
  const [dataset, setDataset] = useState("burgers");
  const [dataPath, setDataPath] = useState("./1D_Burgers_Sols_Nu0.001.hdf5");
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [viewMode, setViewMode] = useState<"chart" | "heatmap">("chart");

  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);

  async function loadDataset() {
    setLoading(true);
    setError(null);
    setInfo(null);
    setFrameIdx(0);
    try {
      const res = await fetch(`${API}/api/dataset/info?dataset=${dataset}&dataPath=${encodeURIComponent(dataPath)}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setInfo(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load dataset metadata");
    } finally {
      setLoading(false);
    }
  }

  // Auto-play loop
  useEffect(() => {
    if (!isPlaying || !info) return;
    const frames = info.sample_trajectory.frames;
    if (frames.length === 0) return;

    const timer = setInterval(() => {
      setFrameIdx((prev) => (prev + 1) % frames.length);
    }, 450);

    return () => clearInterval(timer);
  }, [isPlaying, info]);

  const frames = info?.sample_trajectory.frames ?? [];
  const currentFrame = frames[frameIdx];
  const chartData = currentFrame
    ? currentFrame.values.map((v, i) => ({
        x: info ? parseFloat((info.x_min + (i / (currentFrame.values.length - 1)) * (info.x_max - info.x_min)).toFixed(3)) : i,
        u: parseFloat(v.toFixed(5)),
      }))
    : [];

  // Space-Time Heatmap rendering
  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || !info || viewMode !== "heatmap" || frames.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const margin = { top: 20, right: 30, bottom: 40, left: 55 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    ctx.fillStyle = "#FAF7F2";
    ctx.fillRect(0, 0, width, height);

    const T = frames.length;
    const X = frames[0].values.length;
    const cellW = plotW / X;
    const cellH = plotH / T;

    // Find min / max across all frames
    let minV = Infinity;
    let maxV = -Infinity;
    for (const f of frames) {
      for (const val of f.values) {
        if (val < minV) minV = val;
        if (val > maxV) maxV = val;
      }
    }

    for (let t = 0; t < T; t++) {
      const row = frames[t].values;
      for (let x = 0; x < X; x++) {
        const norm = (row[x] - minV) / (maxV - minV || 1);
        // Earth colormap: Linen -> Terracotta -> Espresso
        let r = 250, g = 247, b = 242;
        if (norm < 0.5) {
          const f = norm / 0.5;
          r = Math.round(244 - 60 * f);
          g = Math.round(239 - 146 * f);
          b = Math.round(230 - 176 * f);
        } else {
          const f = (norm - 0.5) / 0.5;
          r = Math.round(184 - 142 * f);
          g = Math.round(93 - 58 * f);
          b = Math.round(54 - 26 * f);
        }
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(margin.left + x * cellW, margin.top + t * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Active frame cursor line
    const curY = margin.top + (frameIdx / T) * plotH + cellH / 2;
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, curY);
    ctx.lineTo(width - margin.right, curY);
    ctx.stroke();

    ctx.strokeStyle = "#231C16";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(margin.left, curY);
    ctx.lineTo(width - margin.right, curY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.fillStyle = "#706050";
    ctx.font = "11px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Spatial Grid (x)", margin.left + plotW / 2, height - 10);

    ctx.save();
    ctx.translate(16, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Timestep t", 0, 0);
    ctx.restore();
  }, [info, frames, viewMode, frameIdx]);

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Dataset Explorer</h2>
            <p>Inspect ground truth trajectories, boundary conditions, and spatial mesh discretizations.</p>
          </div>
          {info && (
            <span className="badge badge-sage" style={{ padding: "5px 12px", fontSize: 12 }}>
              ✓ Loaded: {info.dataset} ({info.file_size_mb} MB)
            </span>
          )}
        </div>
      </div>

      <div className="page-body">
        {/* Load form */}
        <div className="card" style={{ marginBottom: 24, background: "var(--bg-card)" }}>
          <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div className="form-group" style={{ flex: "0 0 280px", marginBottom: 0 }}>
              <label className="form-label">Dataset Archetype &amp; Shape</label>
              <select
                className="form-select"
                value={dataset}
                onChange={(e) => {
                  const val = e.target.value;
                  setDataset(val);
                  if (val === "burgers") setDataPath("./1D_Burgers_Sols_Nu0.001.hdf5");
                  else setDataPath(`./datasets/${val}.hdf5`);
                }}
              >
                <optgroup label="1D PDE Benchmarks">
                  <option value="burgers">Viscous Burgers' 1D (PDEBench 8.2GB)</option>
                  <option value="burgers_1d">Synthetic Burgers' 1D (0.3MB)</option>
                </optgroup>
                <optgroup label="2D Aerodynamic &amp; Obstacle Bodies">
                  <option value="cylinder_flow_2d">2D Cylinder Flow (Karman Vortex)</option>
                  <option value="airfoil_flow_2d">2D NACA 0012 Airfoil (Wake Separation)</option>
                  <option value="square_flow_2d">2D Square Bluff Body (Corner Recirculation)</option>
                  <option value="step_flow_2d">2D Backward-Facing Step (Expansion Flow)</option>
                  <option value="multi_cylinder_flow_2d">2D Multi-Cylinder Array (Interfering Wake)</option>
                </optgroup>
                <optgroup label="2D Continuous Fluid &amp; Field PDEs">
                  <option value="heat_2d">2D Heat / Thermal Diffusion</option>
                  <option value="advection_2d">2D Advection / Rotating Flow</option>
                  <option value="burgers_2d">2D Coupled Burgers (Vector Velocity)</option>
                  <option value="navier_stokes_2d">2D Navier-Stokes (Vorticity Dynamics)</option>
                </optgroup>
              </select>
            </div>
            <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
              <label className="form-label">HDF5 File Path</label>
              <input className="form-input" value={dataPath} onChange={(e) => setDataPath(e.target.value)} />
            </div>
            <button
              className="btn btn-primary"
              onClick={loadDataset}
              disabled={loading}
              style={{ flexShrink: 0, padding: "9px 22px" }}
            >
              {loading ? (
                <>
                  <span className="spinner animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <Search size={14} />
                  Inspect Dataset
                </>
              )}
            </button>
          </div>

          {/* Quick Preset Selector Pills */}
          <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 600 }}>Quick Presets:</span>
            {[
              { id: "burgers", label: "Burgers 1D", path: "./1D_Burgers_Sols_Nu0.001.hdf5" },
              { id: "cylinder_flow_2d", label: "Cylinder 2D", path: "./datasets/cylinder_flow_2d.hdf5" },
              { id: "airfoil_flow_2d", label: "NACA Airfoil", path: "./datasets/airfoil_flow_2d.hdf5" },
              { id: "square_flow_2d", label: "Square Body", path: "./datasets/square_flow_2d.hdf5" },
              { id: "step_flow_2d", label: "Step Channel", path: "./datasets/step_flow_2d.hdf5" },
              { id: "heat_2d", label: "Heat 2D", path: "./datasets/heat_2d.hdf5" },
              { id: "navier_stokes_2d", label: "Navier-Stokes", path: "./datasets/navier_stokes_2d.hdf5" },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  setDataset(p.id);
                  setDataPath(p.path);
                }}
                style={{
                  fontSize: 10.5,
                  padding: "3px 8px",
                  borderRadius: 12,
                  border: "1px solid var(--border)",
                  background: dataPath === p.path ? "rgba(196, 112, 79, 0.1)" : "var(--bg-card-subtle)",
                  color: dataPath === p.path ? "var(--accent-terracotta)" : "var(--text-secondary)",
                  fontWeight: dataPath === p.path ? 600 : 400,
                  cursor: "pointer",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          {error && (
            <div
              style={{
                marginTop: 14,
                padding: "10px 14px",
                borderRadius: "var(--radius-sm)",
                background: "#FBF0EF",
                border: "1px solid #ECCBC8",
                fontSize: 13,
                color: "var(--accent-brick)",
              }}
            >
              {error}
            </div>
          )}
        </div>

        {info && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Metric summary */}
            <div className="grid-4">
              <div className="metric-card terracotta">
                <div className="card-title">Trajectories</div>
                <div className="card-value">{info.num_trajectories.toLocaleString()}</div>
                <div className="card-sub">Independent initial states</div>
              </div>
              <div className="metric-card sage">
                <div className="card-title">Timesteps / Run</div>
                <div className="card-value">{info.num_timesteps}</div>
                <div className="card-sub">Temporal depth (t₀ → t₂₀₀)</div>
              </div>
              <div className="metric-card amber">
                <div className="card-title">Spatial Discretization</div>
                <div className="card-value">{info.num_grid_points} pts</div>
                <div className="card-sub">1D uniform mesh grid</div>
              </div>
              <div className="metric-card dark">
                <div className="card-title">Domain Bounds</div>
                <div className="card-value" style={{ fontSize: 18 }}>
                  [{info.x_min.toFixed(2)}, {info.x_max.toFixed(2)}]
                </div>
                <div className="card-sub">Periodic boundary conditions</div>
              </div>
            </div>

            {/* Trajectory viewer */}
            <div className="card">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 12,
                  marginBottom: 16,
                  paddingBottom: 14,
                  borderBottom: "1px solid var(--border)",
                }}
              >
                <div>
                  <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                    <Waves size={18} color="var(--accent-terracotta)" />
                    Trajectory Sample #{info.sample_trajectory.index}
                  </h3>
                  <p style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>
                    Evolution of fluid velocity profile across time.
                  </p>
                </div>

                {/* View Toggles & Playback */}
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ display: "flex", gap: 4, background: "var(--bg-card-subtle)", padding: 3, borderRadius: 6 }}>
                    <button
                      onClick={() => setViewMode("chart")}
                      className={`btn-subtle ${viewMode === "chart" ? "active" : ""}`}
                      style={{ fontSize: 11, padding: "4px 10px" }}
                    >
                      1D Profile Slice
                    </button>
                    <button
                      onClick={() => setViewMode("heatmap")}
                      className={`btn-subtle ${viewMode === "heatmap" ? "active" : ""}`}
                      style={{ fontSize: 11, padding: "4px 10px" }}
                    >
                      <Sparkles size={11} style={{ marginRight: 4 }} />
                      Space-Time Waterfall
                    </button>
                  </div>

                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="btn btn-secondary"
                    style={{ padding: "6px 12px", fontSize: 12 }}
                  >
                    {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                    {isPlaying ? "Pause" : "Play"}
                  </button>
                </div>
              </div>

              {/* View 1: 1D Profile Chart */}
              {viewMode === "chart" && (
                <div>
                  <div className="step-nav" style={{ marginBottom: 16 }}>
                    {frames.map((f, i) => (
                      <button
                        key={i}
                        className={`step-btn ${frameIdx === i ? "active" : ""}`}
                        onClick={() => setFrameIdx(i)}
                      >
                        t={f.t}
                      </button>
                    ))}
                  </div>

                  <ResponsiveContainer width="100%" height={290}>
                    <LineChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#EDE6DC" />
                      <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#9C8E80", fontFamily: "JetBrains Mono" }} />
                      <YAxis tick={{ fontSize: 10, fill: "#9C8E80", fontFamily: "JetBrains Mono" }} />
                      <Tooltip
                        contentStyle={{
                          background: "#FFFFFF",
                          border: "1px solid #D9CCBB",
                          borderRadius: 4,
                          fontSize: 12,
                          boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
                        }}
                        formatter={(v: any) => [typeof v === "number" ? v.toFixed(5) : String(v), "Velocity u(x)"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="u"
                        stroke="#B85D36"
                        strokeWidth={2.5}
                        dot={false}
                        name="Velocity Field u"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* View 2: Space-Time Waterfall */}
              {viewMode === "heatmap" && (
                <div style={{ borderRadius: "var(--radius)", overflow: "hidden", border: "1px solid var(--border)" }}>
                  <canvas
                    ref={heatmapCanvasRef}
                    width={800}
                    height={300}
                    style={{ width: "100%", height: "300px", display: "block" }}
                  />
                </div>
              )}

              {/* Frame Slider */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
                <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 600 }}>TIMESTEP SCRUBBER:</span>
                <input
                  type="range"
                  min={0}
                  max={frames.length - 1}
                  value={frameIdx}
                  onChange={(e) => setFrameIdx(Number(e.target.value))}
                  style={{ flex: 1, accentColor: "var(--accent-terracotta)", cursor: "pointer" }}
                />
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
                  Frame {frameIdx + 1}/{frames.length} (t={currentFrame?.t ?? 0})
                </span>
              </div>
            </div>
          </div>
        )}

        {!info && !loading && (
          <div className="card">
            <div className="empty-state">
              <Database size={36} color="var(--sand-400)" />
              <p>Click &ldquo;Inspect Dataset&rdquo; above to browse trajectories, fluid shockwaves, and space-time waterfalls.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
