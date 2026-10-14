"use client";
import { useState, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { Cpu, Play, Square, AlertCircle, CheckCircle, Sliders } from "lucide-react";

const API = "http://localhost:3001";

interface EpochData {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

function parseLogLine(line: string): EpochData | null {
  // Matches: "Epoch  1 | train MSE: 0.012345 | val MSE: 0.023456"
  const m = line.match(/Epoch\s+(\d+)\s*\|\s*train MSE:\s*([\d.eE+\-]+)\s*\|\s*val MSE:\s*([\d.eE+\-]+)/);
  if (!m) return null;
  return { epoch: parseInt(m[1]), trainLoss: parseFloat(m[2]), valLoss: parseFloat(m[3]) };
}

export default function TrainPage() {
  const [dataset, setDataset] = useState("burgers");
  const [dataPath, setDataPath] = useState("./1D_Burgers_Sols_Nu0.001.hdf5");
  const [metaPath, setMetaPath] = useState("");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [lr, setLr] = useState(0.0001);
  const [checkpointPath, setCheckpointPath] = useState("baseline_model.pt");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [epochData, setEpochData] = useState<EpochData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const startTraining = useCallback(async () => {
    setRunning(true);
    setLogs([]);
    setEpochData([]);
    setError(null);
    setDone(false);

    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${API}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, dataPath, metaPath: metaPath || undefined, epochs, batchSize, lr, checkpointPath }),
        signal: abortRef.current.signal,
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;

        const text = decoder.decode(value);
        const lines = text.split("\n").filter((l) => l.startsWith("data: "));

        for (const line of lines) {
          try {
            const payload = JSON.parse(line.replace("data: ", ""));
            if (payload.log) {
              setLogs((prev) => [...prev, payload.log]);
              const parsed = parseLogLine(payload.log);
              if (parsed) setEpochData((prev) => [...prev, parsed]);
            }
            if (payload.error) {
              setLogs((prev) => [...prev, `[stderr] ${payload.error}`]);
            }
            if (payload.done) {
              setDone(true);
              setRunning(false);
            }
          } catch {
            // Ignore malformed SSE chunks
          }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : "Training failed");
      }
    } finally {
      setRunning(false);
    }
  }, [dataset, dataPath, metaPath, epochs, batchSize, lr, checkpointPath]);

  function stopTraining() {
    abortRef.current?.abort();
    setRunning(false);
  }

  const latestEpoch = epochData[epochData.length - 1];

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Model Training &amp; Convergence</h2>
            <p>Optimize MeshGraphNet weights via message-passing residuals with live SSE telemetry.</p>
          </div>
          {running && (
            <span className="badge badge-terracotta" style={{ padding: "5px 12px", fontSize: 12 }}>
              <span className="spinner animate-spin" style={{ width: 10, height: 10, marginRight: 5 }} />
              Active Training Run
            </span>
          )}
          {done && (
            <span className="badge badge-sage" style={{ padding: "5px 12px", fontSize: 12 }}>
              ✓ Training Finished
            </span>
          )}
        </div>
      </div>

      <div className="page-body">
        <div className="grid-2" style={{ gap: 24, alignItems: "start" }}>
          {/* Hyperparameters form */}
          <div className="card">
            <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <Sliders size={16} color="var(--accent-terracotta)" />
              Hyperparameters
            </h3>

            <div className="form-group">
              <label className="form-label">Dataset</label>
              <select className="form-select" value={dataset} onChange={(e) => setDataset(e.target.value)} disabled={running}>
                <option value="burgers">Burgers' Equation (1D)</option>
                <option value="cylinder">CylinderFlow (2D)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">HDF5 Data Path</label>
              <input className="form-input" value={dataPath} onChange={(e) => setDataPath(e.target.value)} disabled={running} />
            </div>

            {dataset === "cylinder" && (
              <div className="form-group">
                <label className="form-label">Meta JSON Path</label>
                <input className="form-input" value={metaPath} onChange={(e) => setMetaPath(e.target.value)} disabled={running} />
              </div>
            )}

            <div className="grid-2" style={{ gap: 10 }}>
              <div className="form-group">
                <label className="form-label">Epochs</label>
                <input className="form-input" type="number" min={1} max={500} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} disabled={running} />
              </div>
              <div className="form-group">
                <label className="form-label">Batch Size</label>
                <input className="form-input" type="number" min={1} max={64} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} disabled={running} />
              </div>
            </div>

            <div className="grid-2" style={{ gap: 10 }}>
              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input className="form-input" type="number" step="0.00005" value={lr} onChange={(e) => setLr(Number(e.target.value))} disabled={running} />
              </div>
              <div className="form-group">
                <label className="form-label">Checkpoint Target</label>
                <input className="form-input" value={checkpointPath} onChange={(e) => setCheckpointPath(e.target.value)} disabled={running} />
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
              {!running ? (
                <button className="btn btn-primary" onClick={startTraining} style={{ flex: 1, justifyContent: "center" }}>
                  <Play size={14} /> Start Training Run
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={stopTraining} style={{ flex: 1, justifyContent: "center", color: "var(--accent-brick)" }}>
                  <Square size={14} /> Interrupt Run
                </button>
              )}
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

          {/* Telemetry & Loss Curves */}
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {/* Live Metric Row */}
            <div className="grid-2" style={{ gap: 14 }}>
              <div className="metric-card sage">
                <div className="card-title">Training Loss (MSE)</div>
                <div className="card-value">
                  {latestEpoch ? latestEpoch.trainLoss.toExponential(4) : "—"}
                </div>
                <div className="card-sub">{latestEpoch ? `Epoch ${latestEpoch.epoch}` : "Awaiting data"}</div>
              </div>
              <div className="metric-card terracotta">
                <div className="card-title">Validation Loss (MSE)</div>
                <div className="card-value">
                  {latestEpoch ? latestEpoch.valLoss.toExponential(4) : "—"}
                </div>
                <div className="card-sub">{latestEpoch ? `Epoch ${latestEpoch.epoch}` : "Awaiting data"}</div>
              </div>
            </div>

            {/* Convergence Chart */}
            <div className="chart-container">
              <div className="chart-title">Loss Convergence Curves</div>
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={epochData} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EDE6DC" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#9C8E80", fontFamily: "JetBrains Mono" }} label={{ value: "Epoch", position: "insideBottom", offset: -5, fontSize: 11, fill: "#9C8E80" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#9C8E80", fontFamily: "JetBrains Mono" }} />
                  <Tooltip contentStyle={{ background: "#FFFFFF", border: "1px solid #D9CCBB", borderRadius: 4, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="trainLoss" stroke="#38704A" strokeWidth={2} dot={false} name="Train MSE" />
                  <Line type="monotone" dataKey="valLoss" stroke="#B85D36" strokeWidth={2} dot={false} name="Val MSE" strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Terminal log */}
            <div className="card" style={{ padding: "16px 20px" }}>
              <div className="card-title" style={{ marginBottom: 10 }}>Live Telemetry Console</div>
              <div className="terminal" style={{ height: 180 }}>
                {logs.length === 0 ? (
                  <span style={{ color: "var(--sand-500)" }}>[System standby. Click &ldquo;Start Training Run&rdquo; to launch child process.]</span>
                ) : (
                  logs.map((line, idx) => (
                    <div key={idx} style={{ color: line.includes("val MSE") ? "#E6C280" : line.includes("[stderr]") ? "#E57373" : "#D1E4D4" }}>
                      {line}
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
