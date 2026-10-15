"use client";
import React, { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, ChevronLeft, ChevronRight, Layers, Eye, RefreshCw, Zap } from "lucide-react";

interface Flow2DSlice {
  shape: string;
  equation: string;
  trajectory: number;
  timestep: number;
  total_timesteps: number;
  grid: [number, number];
  min_val: number;
  max_val: number;
  mask: boolean[][];
  ground_truth: number[][];
  baseline: number[][];
  adapted: number[][];
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
}

export default function FlowVisualizer2D() {
  const [shape, setShape] = useState<string>("cylinder");
  const [timestep, setTimestep] = useState<number>(2);
  const [viewVariant, setViewVariant] = useState<"ground_truth" | "baseline" | "adapted" | "diff">("adapted");
  const [data, setData] = useState<Flow2DSlice | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [showParticles, setShowParticles] = useState<boolean>(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animFrameRef = useRef<number | null>(null);

  // Fetch 2D slice from API
  const fetchSlice = useCallback(async (s: string, t: number) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:3001/api/flow-2d/slice?shape=${s}&timestep=${t}`);
      const json = await res.json();
      if (!json.error) {
        setData(json);
      }
    } catch (e) {
      console.error("Failed to fetch 2D slice:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSlice(shape, timestep);
  }, [shape, timestep, fetchSlice]);

  // Auto-play animation loop through timesteps
  useEffect(() => {
    if (!isPlaying || !data) return;
    const interval = setInterval(() => {
      setTimestep((prev) => (prev + 1) % (data.total_timesteps || 13));
    }, 600);
    return () => clearInterval(interval);
  }, [isPlaying, data]);

  // Colormap helper: converts normalized value [0, 1] to RGB
  function getThermalColor(val: number, isDiff = false): [number, number, number] {
    val = Math.max(0, Math.min(1, val));
    if (isDiff) {
      // Hot error map: white -> yellow -> red -> dark violet
      const r = Math.round(255 * Math.min(1, val * 2));
      const g = Math.round(255 * Math.max(0, 1 - val * 1.5));
      const b = Math.round(255 * Math.max(0, 1 - val * 3));
      return [r, g, b];
    }
    // Deep thermal oceanic / fluid colormap: Dark Indigo -> Teal -> Amber -> Cream
    if (val < 0.25) {
      const f = val / 0.25;
      return [Math.round(24 + 18 * f), Math.round(32 + 56 * f), Math.round(64 + 70 * f)];
    } else if (val < 0.5) {
      const f = (val - 0.25) / 0.25;
      return [Math.round(42 + 20 * f), Math.round(88 + 80 * f), Math.round(134 + 20 * f)];
    } else if (val < 0.75) {
      const f = (val - 0.5) / 0.25;
      return [Math.round(62 + 130 * f), Math.round(168 + 12 * f), Math.round(154 - 60 * f)];
    } else {
      const f = (val - 0.75) / 0.25;
      return [Math.round(192 + 55 * f), Math.round(180 + 60 * f), Math.round(94 + 140 * f)];
    }
  }

  // Canvas drawing & particle simulation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const [gridNy, gridNx] = data.grid;
    const cellW = width / gridNx;
    const cellH = height / gridNy;

    // Pick active field
    let field: number[][];
    const isDiff = viewVariant === "diff";
    if (viewVariant === "ground_truth") {
      field = data.ground_truth;
    } else if (viewVariant === "baseline") {
      field = data.baseline;
    } else if (viewVariant === "adapted") {
      field = data.adapted;
    } else {
      // Difference: |baseline - ground_truth| vs |adapted - ground_truth|
      field = data.ground_truth.map((row, r) =>
        row.map((val, c) => Math.abs(data.baseline[r][c] - val) * 4.0)
      );
    }

    // Initialize particles if needed
    if (particlesRef.current.length < 80) {
      particlesRef.current = Array.from({ length: 80 }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: 1.5 + Math.random() * 2.0,
        vy: (Math.random() - 0.5) * 0.8,
        life: Math.random() * 60,
        maxLife: 60 + Math.random() * 40,
      }));
    }

    const render = () => {
      // 1. Draw 2D Field Heatmap
      const imgData = ctx.createImageData(width, height);
      const buf = imgData.data;

      for (let y = 0; y < height; y++) {
        const gy = Math.min(gridNy - 1, Math.floor((y / height) * gridNy));
        for (let x = 0; x < width; x++) {
          const gx = Math.min(gridNx - 1, Math.floor((x / width) * gridNx));
          const idx = (y * width + x) * 4;

          const isSolid = data.mask[gy] && data.mask[gy][gx];
          if (isSolid) {
            // Dark obstacle body styling
            buf[idx] = 35;
            buf[idx + 1] = 28;
            buf[idx + 2] = 22;
            buf[idx + 3] = 255;
          } else {
            const val = field[gy] ? field[gy][gx] : 0;
            const [r, g, b] = getThermalColor(val, isDiff);
            buf[idx] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = 255;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);

      // 2. Draw Solid Obstacle Borders
      for (let gy = 0; gy < gridNy; gy++) {
        for (let gx = 0; gx < gridNx; gx++) {
          if (data.mask[gy] && data.mask[gy][gx]) {
            ctx.fillStyle = "rgba(22, 18, 14, 0.95)";
            ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
      }

      // 3. Draw Streamline Advection Particles
      if (showParticles && !isDiff) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
        ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
        ctx.shadowBlur = 4;

        particlesRef.current.forEach((p) => {
          const gy = Math.min(gridNy - 1, Math.max(0, Math.floor((p.y / height) * gridNy)));
          const gx = Math.min(gridNx - 1, Math.max(0, Math.floor((p.x / width) * gridNx)));
          const isSolid = data.mask[gy] && data.mask[gy][gx];

          if (isSolid) {
            p.x = 0;
            p.y = Math.random() * height;
          }

          const localSpeed = field[gy] ? field[gy][gx] : 0.5;
          p.x += p.vx * (0.8 + localSpeed * 1.5);
          p.y += p.vy;

          p.life++;
          if (p.x > width || p.y < 0 || p.y > height || p.life > p.maxLife) {
            p.x = 0;
            p.y = Math.random() * height;
            p.life = 0;
          }

          ctx.beginPath();
          ctx.arc(p.x, p.y, 1.2, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.shadowBlur = 0;
      }

      // Legend overlay text
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillText(
        `${data.equation} | t=${timestep}/${data.total_timesteps - 1} | Grid: ${gridNy}x${gridNx}`,
        12,
        height - 12
      );
    };

    render();
  }, [data, viewVariant, showParticles, timestep]);

  return (
    <div className="card" style={{ padding: "18px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Header & Controls Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
            <Layers size={14} color="var(--accent-terracotta)" />
            Interactive 2D Aerodynamic &amp; Obstacle Flow Field
          </div>
          <p style={{ fontSize: 11.5, color: "var(--text-secondary)", margin: "2px 0 0" }}>
            Real-time D2Q9 Navier-Stokes velocity colormap &amp; particle streamlines advecting past solid bodies.
          </p>
        </div>

        {/* Shape Selector Buttons */}
        <div style={{ display: "flex", gap: 4, background: "var(--bg-card-subtle)", padding: 3, borderRadius: 5, border: "1px solid var(--border)" }}>
          {[
            { id: "cylinder", label: "Cylinder" },
            { id: "airfoil", label: "NACA Airfoil" },
            { id: "square", label: "Square Body" },
            { id: "step", label: "Step Channel" },
            { id: "multi_cylinder", label: "Dual Array" },
          ].map((s) => (
            <button
              key={s.id}
              onClick={() => {
                setShape(s.id);
                setTimestep(2);
              }}
              style={{
                padding: "4px 10px",
                fontSize: 11,
                borderRadius: 4,
                fontWeight: shape === s.id ? 600 : 400,
                border: "none",
                background: shape === s.id ? "#FFFFFF" : "transparent",
                boxShadow: shape === s.id ? "0 1px 2px rgba(0,0,0,0.06)" : "none",
                color: shape === s.id ? "var(--accent-terracotta)" : "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Canvas Viewport */}
      <div
        style={{
          position: "relative",
          width: "100%",
          borderRadius: 6,
          overflow: "hidden",
          border: "1px solid var(--border)",
          background: "#181410",
        }}
      >
        <canvas
          ref={canvasRef}
          width={768}
          height={384}
          style={{ width: "100%", height: "340px", display: "block" }}
        />

        {/* View Mode Pill Toggle on Top of Canvas */}
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            display: "flex",
            gap: 4,
            background: "rgba(22, 18, 14, 0.8)",
            backdropFilter: "blur(4px)",
            padding: 3,
            borderRadius: 5,
            border: "1px solid rgba(255,255,255,0.15)",
          }}
        >
          {[
            { id: "adapted", label: "Post-TTA Adapted" },
            { id: "baseline", label: "Pre-TTA Baseline" },
            { id: "ground_truth", label: "Ground Truth" },
            { id: "diff", label: "Error Field" },
          ].map((v) => (
            <button
              key={v.id}
              onClick={() => setViewVariant(v.id as any)}
              style={{
                fontSize: 10.5,
                padding: "3px 8px",
                borderRadius: 3,
                border: "none",
                background: viewVariant === v.id ? "var(--accent-terracotta)" : "transparent",
                color: "#FFFFFF",
                fontWeight: viewVariant === v.id ? 600 : 400,
                cursor: "pointer",
              }}
            >
              {v.label}
            </button>
          ))}
        </div>

        {/* Particle Streamline Toggle */}
        <button
          onClick={() => setShowParticles(!showParticles)}
          style={{
            position: "absolute",
            top: 10,
            left: 10,
            fontSize: 10,
            padding: "3px 8px",
            borderRadius: 4,
            background: showParticles ? "rgba(40, 167, 69, 0.8)" : "rgba(30, 25, 20, 0.75)",
            color: "#FFFFFF",
            border: "1px solid rgba(255,255,255,0.2)",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <Zap size={10} />
          {showParticles ? "Particles: ON" : "Particles: OFF"}
        </button>
      </div>

      {/* Scrub Controls & Animation Bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, paddingTop: 4 }}>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="btn btn-secondary"
          style={{ padding: "5px 12px", height: 28, fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}
        >
          {isPlaying ? <Pause size={11} /> : <Play size={11} />}
          {isPlaying ? "Pause" : "Play"}
        </button>

        <button
          onClick={() => setTimestep((prev) => Math.max(0, prev - 1))}
          style={{ border: "1px solid var(--border)", background: "#FFF", borderRadius: 4, padding: "4px 8px", cursor: "pointer" }}
        >
          <ChevronLeft size={12} />
        </button>

        <input
          type="range"
          min={0}
          max={(data?.total_timesteps || 13) - 1}
          value={timestep}
          onChange={(e) => {
            setIsPlaying(false);
            setTimestep(Number(e.target.value));
          }}
          style={{ flex: 1, accentColor: "var(--accent-terracotta)", cursor: "pointer" }}
        />

        <button
          onClick={() => setTimestep((prev) => Math.min((data?.total_timesteps || 13) - 1, prev + 1))}
          style={{ border: "1px solid var(--border)", background: "#FFF", borderRadius: 4, padding: "4px 8px", cursor: "pointer" }}
        >
          <ChevronRight size={12} />
        </button>

        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "var(--text-primary)", fontWeight: 600, minWidth: 90 }}>
          Step {timestep + 1}/{data?.total_timesteps || 13}
        </span>
      </div>
    </div>
  );
}
