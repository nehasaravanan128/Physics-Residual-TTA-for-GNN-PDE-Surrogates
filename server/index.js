/**
 * server/index.js
 * Express API server — bridges the Next.js UI to the Python ML scripts.
 *
 * Endpoints:
 *   POST /api/infer     — run model inference on a data slice
 *   POST /api/train     — launch a training job (SSE stream for live logs)
 *   POST /api/evaluate  — evaluate a checkpoint and return metrics
 *   GET  /api/dataset/info — return basic info about the loaded dataset
 */

const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(cors());
app.use(express.json());

const ROOT = path.resolve(__dirname, "..");
const SCRIPTS_DIR = path.join(ROOT, "scripts");
const FILES_DIR = path.join(ROOT, "files (5)");

// ─── Helpers ────────────────────────────────────────────────────────────────

function getPythonExe() {
  const candidates = [
    process.env.PYTHON_PATH,
    "C:\\Users\\nehas\\miniconda3\\python.exe",
    path.join(ROOT, "files (5)", ".venv", "Scripts", "python.exe"),
    "python",
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return "python";
}

function runPython(scriptName, args, res, onData) {
  const scriptPath = path.join(SCRIPTS_DIR, scriptName);
  const pythonExe = getPythonExe();

  const proc = spawn(pythonExe, [scriptPath, ...args], {
    cwd: FILES_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  let stdout = "";
  let stderr = "";

  proc.stdout.on("data", (d) => {
    const chunk = d.toString();
    stdout += chunk;
    if (onData) onData(chunk, res);
  });

  proc.stderr.on("data", (d) => {
    stderr += d.toString();
  });

  return new Promise((resolve, reject) => {
    proc.on("close", (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(stderr || `Process exited with code ${code}`));
    });
  });
}

// ─── POST /api/infer ─────────────────────────────────────────────────────────
app.post("/api/infer", async (req, res) => {
  const {
    dataset = "burgers",
    dataPath,
    checkpointPath,
    trajectoryIdx = 0,
    timestepIdx = 0,
    numSteps = 5,
  } = req.body;

  if (!dataPath || !checkpointPath) {
    return res.status(400).json({ error: "dataPath and checkpointPath are required." });
  }

  try {
    const { stdout } = await runPython("infer.py", [
      "--dataset", dataset,
      "--data_path", dataPath,
      "--checkpoint_path", checkpointPath,
      "--traj", String(trajectoryIdx),
      "--timestep", String(timestepIdx),
      "--num_steps", String(numSteps),
    ]);

    const jsonStart = stdout.indexOf("{");
    const jsonStr = jsonStart >= 0 ? stdout.slice(jsonStart) : stdout;
    res.json(JSON.parse(jsonStr));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── POST /api/tta (Physics-Residual Test-Time Adaptation) ───────────────────
app.post("/api/tta", async (req, res) => {
  const {
    dataset = "burgers",
    dataPath,
    checkpointPath,
    trajectoryIdx = 0,
    trajIdx,
    timestepIdx = 10,
    ttaSteps = 10,
    ttaLr = 0.00008,
    nu = 0.001,
  } = req.body;

  if (!dataPath || !checkpointPath) {
    return res.status(400).json({ error: "dataPath and checkpointPath are required." });
  }

  const effectiveTraj = trajIdx !== undefined ? trajIdx : trajectoryIdx;

  try {
    const { stdout } = await runPython("tta.py", [
      "--data_path", dataPath,
      "--checkpoint_path", checkpointPath,
      "--traj", String(effectiveTraj),
      "--timestep", String(timestepIdx),
      "--tta_steps", String(ttaSteps),
      "--tta_lr", String(ttaLr),
      "--nu", String(nu),
      "--json",
    ]);

    const jsonStart = stdout.indexOf("{");
    const jsonStr = jsonStart >= 0 ? stdout.slice(jsonStart) : stdout;
    res.json(JSON.parse(jsonStr));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── POST /api/train (SSE streaming) ────────────────────────────────────────
app.post("/api/train", (req, res) => {
  const {
    dataset = "burgers",
    dataPath,
    metaPath,
    epochs = 50,
    batchSize = 16,
    lr = 0.0001,
    checkpointPath = "baseline_model.pt",
    seed = 0,
  } = req.body;

  if (!dataPath) {
    return res.status(400).json({ error: "dataPath is required." });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const args = [
    "--dataset", dataset,
    "--data_path", dataPath,
    "--epochs", String(epochs),
    "--batch_size", String(batchSize),
    "--lr", String(lr),
    "--checkpoint_path", checkpointPath,
    "--seed", String(seed),
  ];
  if (metaPath) args.push("--meta_path", metaPath);

  const pythonExe = getPythonExe();
  const trainScript = path.join(FILES_DIR, "train.py");

  const proc = spawn(pythonExe, [trainScript, ...args], {
    cwd: FILES_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  proc.stdout.on("data", (d) => {
    const lines = d.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      res.write(`data: ${JSON.stringify({ log: line })}\n\n`);
    }
  });

  proc.stderr.on("data", (d) => {
    res.write(`data: ${JSON.stringify({ error: d.toString() })}\n\n`);
  });

  proc.on("close", (code) => {
    res.write(`data: ${JSON.stringify({ done: true, code })}\n\n`);
    res.end();
  });

  req.on("close", () => proc.kill());
});

// ─── POST /api/evaluate ──────────────────────────────────────────────────────
app.post("/api/evaluate", async (req, res) => {
  const { dataset = "burgers", dataPath, checkpointPath, metaPath } = req.body;

  if (!dataPath || !checkpointPath) {
    return res.status(400).json({ error: "dataPath and checkpointPath are required." });
  }

  const args = [
    "--dataset", dataset,
    "--data_path", dataPath,
    "--checkpoint_path", checkpointPath,
    "--output_json",
  ];
  if (metaPath) args.push("--meta_path", metaPath);

  try {
    const { stdout } = await runPython("evaluate_api.py", args);
    const jsonStart = stdout.indexOf("{");
    const jsonStr = jsonStart >= 0 ? stdout.slice(jsonStart) : stdout;
    res.json(JSON.parse(jsonStr));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /api/dataset/info ───────────────────────────────────────────────────
app.get("/api/dataset/info", async (req, res) => {
  const { dataset = "burgers", dataPath } = req.query;

  if (!dataPath) {
    return res.status(400).json({ error: "dataPath query param is required." });
  }

  try {
    const { stdout } = await runPython("dataset_info.py", [
      "--dataset", dataset,
      "--data_path", dataPath,
    ]);
    const jsonStart = stdout.indexOf("{");
    res.json(JSON.parse(stdout.slice(jsonStart)));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── POST /api/evaluate-geometric (Geometric Distribution Shift Benchmark) ──
app.post("/api/evaluate-geometric", async (req, res) => {
  const {
    dataPath = "files (5)/1D_Burgers_Sols_Nu0.001.hdf5",
    checkpointPath = "real_burgers_model.pt",
    numSamples = 5,
    ttaSteps = 10,
    ttaLr = 0.0001,
    shiftType = "all",
  } = req.body;

  try {
    const jsonPath = path.join(SCRIPTS_DIR, "geometric_benchmark_results.json");
    const args = [
      "--data_path", dataPath,
      "--checkpoint_path", checkpointPath,
      "--num_samples", String(numSamples),
      "--tta_steps", String(ttaSteps),
      "--tta_lr", String(ttaLr),
      "--shift_type", shiftType,
      "--output_json", jsonPath,
    ];

    await runPython("evaluate_geometric_shift.py", args);
    if (fs.existsSync(jsonPath)) {
      const data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
      return res.json({ success: true, results: data });
    }
    res.json({ success: false, error: "Failed to generate benchmark output." });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /api/evaluate-geometric/cached ──────────────────────────────────────
app.get("/api/evaluate-geometric/cached", (req, res) => {
  const jsonPath = path.join(SCRIPTS_DIR, "geometric_benchmark_results.json");
  if (fs.existsSync(jsonPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
      return res.json({ success: true, results: data });
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }
  return res.json({ success: false, message: "No cached geometric benchmark results yet." });
});

// ─── Health check ────────────────────────────────────────────────────────────
app.get("/api/health", (_, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`🚀 API server running on http://localhost:${PORT}`));
