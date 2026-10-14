"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { Waves, ArrowRight, Play, Pause, Sliders, Cpu, Activity, BarChart2, Database } from "lucide-react";

const stats = [
  { label: "Surrogate Architecture", value: "MeshGraphNet", accent: "terracotta", sub: "Pfaff et al. inductive bias" },
  { label: "Message Passing Depth", value: "10 Blocks", accent: "dark", sub: "Residual edge + node MLPs" },
  { label: "Latent Dimension", value: "128 Channels", accent: "sage", sub: "Shared feature manifold" },
  { label: "Kinematic Viscosity (ν)", value: "0.001 m²/s", accent: "amber", sub: "High Reynolds shock regime" },
];

const features = [
  {
    href: "/inference",
    title: "Flow Inference & Dynamics",
    icon: Activity,
    description: "Execute multi-step surrogate rollouts with live fluid particle advection, space-time waterfalls, and pointwise error envelope plots.",
  },
  {
    href: "/explore",
    title: "Dataset Explorer",
    icon: Database,
    description: "Browse 10,000 PDEBench trajectories, inspect boundary conditions, and scrub temporal shockwave formations.",
  },
  {
    href: "/train",
    title: "Surrogate Training",
    icon: Cpu,
    description: "Tune learning rates and hidden layers. Stream live residual MSE training logs directly from the Python backend.",
  },
  {
    href: "/evaluate",
    title: "Benchmark & Verification",
    icon: BarChart2,
    description: "Evaluate held-out checkpoints for relative MSE, conservation of kinetic energy, and spatial L1/L2 error.",
  },
];

interface SandboxParticle {
  x: number;
  y: number;
  size: number;
  alpha: number;
  trail: Array<{ x: number; y: number }>;
  speedMult: number;
}

export default function HomePage() {
  const [shockSteepness, setShockSteepness] = useState(14);
  const [displayTime, setDisplayTime] = useState(0.4);
  const [isPlaying, setIsPlaying] = useState(true);
  const sandboxCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<SandboxParticle[]>([]);
  const waveTimeRef = useRef(0.4);
  const isPlayingRef = useRef(true);
  const steepnessRef = useRef(14);
  const animIdRef = useRef<number>(0);
  const lastTsRef = useRef(0);

  // Sync refs
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { steepnessRef.current = shockSteepness; }, [shockSteepness]);

  // Initialize particles with trails
  useEffect(() => {
    const pts: SandboxParticle[] = [];
    for (let i = 0; i < 120; i++) {
      pts.push({
        x: Math.random(),
        y: Math.random() * 0.7 + 0.15,
        size: Math.random() * 1.8 + 0.8,
        alpha: Math.random() * 0.45 + 0.3,
        trail: [],
        speedMult: Math.random() * 0.4 + 0.8,
      });
    }
    particlesRef.current = pts;
  }, []);

  // High-DPI canvas setup
  const setupCanvas = useCallback((canvas: HTMLCanvasElement) => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const tw = Math.round(rect.width * dpr);
    const th = Math.round(rect.height * dpr);
    if (canvas.width !== tw || canvas.height !== th) {
      canvas.width = tw;
      canvas.height = th;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.resetTransform?.();
    ctx.scale(dpr, dpr);
    return { ctx, width: rect.width, height: rect.height };
  }, []);   // Main 60fps animation loop — all rendering in RAF, no React state in hot path
  useEffect(() => {
    const loop = (timestamp: number) => {
      const delta = Math.min(0.05, (timestamp - (lastTsRef.current || timestamp)) / 1000);
      lastTsRef.current = timestamp;

      // Smooth time advance — fast enough to see the wave propagate visibly
      if (isPlayingRef.current) {
        waveTimeRef.current += 1.6 * delta;
        if (waveTimeRef.current >= 4.0) waveTimeRef.current -= 4.0; // seamless wrap
        if (Math.floor(timestamp / 100) !== Math.floor((timestamp - delta * 1000) / 100)) {
          setDisplayTime(waveTimeRef.current % 1.2);
        }
      }

      const canvas = sandboxCanvasRef.current;
      if (!canvas) { animIdRef.current = requestAnimationFrame(loop); return; }

      const res = setupCanvas(canvas);
      if (!res) { animIdRef.current = requestAnimationFrame(loop); return; }
      const { ctx, width, height } = res;

      const margin = { top: 18, right: 24, bottom: 28, left: 42 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;

      const t = waveTimeRef.current;
      const steep = steepnessRef.current;

      // Background
      ctx.fillStyle = "#FAF7F2";
      ctx.fillRect(0, 0, width, height);

      // Grid lines
      ctx.strokeStyle = "#EDE6DC";
      ctx.lineWidth = 0.7;
      for (let i = 0; i <= 4; i++) {
        const y = margin.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
      }
      for (let i = 0; i <= 6; i++) {
        const x = margin.left + (plotW / 6) * i;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, height - margin.bottom);
        ctx.stroke();
      }

      // Compute wave profile — multi-harmonic traveling wave for realistic fluid motion
      const nPoints = 300;
      const shockX = 0.3 + (t * 0.18) % 0.55;
      const pts: number[] = [];
      for (let i = 0; i < nPoints; i++) {
        const x = i / (nPoints - 1);
        // Traveling wave with 3 harmonics — propagates rightward
        const wave1 = 0.38 * Math.sin(2 * Math.PI * (x - t * 0.4));
        const wave2 = 0.15 * Math.sin(4 * Math.PI * (x - t * 0.55) + 0.8);
        const wave3 = 0.06 * Math.sin(6 * Math.PI * (x - t * 0.7) + 1.6);
        // Shock discontinuity that travels
        const shock = 0.42 * (1 - Math.tanh(steep * (x - shockX)));
        const u = shock + wave1 + wave2 + wave3;
        pts.push(u);
      }

      const getXPix = (x: number) => margin.left + x * plotW;
      const getYPix = (u: number) => margin.top + ((1.2 - u) / 2.0) * plotH;

      // Particle simulation with trails — faster advection
      ctx.save();
      ctx.beginPath();
      ctx.rect(margin.left, margin.top, plotW, plotH);
      ctx.clip();

      for (const p of particlesRef.current) {
        const idx = Math.min(nPoints - 1, Math.max(0, Math.floor(p.x * (nPoints - 1))));
        const uLocal = pts[idx];
        p.x += uLocal * 0.006 * p.speedMult;
        if (p.x > 1) { p.x = 0; p.trail = []; }
        if (p.x < 0) { p.x = 1; p.trail = []; }

        const px = getXPix(p.x);
        const py = margin.top + p.y * plotH;
        p.trail.push({ x: px, y: py });
        if (p.trail.length > 14) p.trail.shift();

        // Render fading trail
        if (p.trail.length > 1) {
          for (let ti = 1; ti < p.trail.length; ti++) {
            const fade = (ti / p.trail.length) * p.alpha * 0.5;
            const lw = (ti / p.trail.length) * 1.5;
            ctx.strokeStyle = `rgba(184, 93, 54, ${fade})`;
            ctx.lineWidth = lw;
            ctx.beginPath();
            ctx.moveTo(p.trail[ti - 1].x, p.trail[ti - 1].y);
            ctx.lineTo(p.trail[ti].x, p.trail[ti].y);
            ctx.stroke();
          }
        }

        // Particle head with velocity-dependent opacity
        const velAlpha = Math.min(0.85, Math.abs(uLocal) * 0.9 + 0.15);
        ctx.fillStyle = `rgba(184, 93, 54, ${velAlpha * p.alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Gradient fill beneath curve
      const grad = ctx.createLinearGradient(0, margin.top, 0, margin.top + plotH);
      grad.addColorStop(0, "rgba(184, 93, 54, 0.14)");
      grad.addColorStop(0.6, "rgba(184, 93, 54, 0.05)");
      grad.addColorStop(1, "rgba(184, 93, 54, 0.0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(getXPix(0), getYPix(pts[0]));
      for (let i = 1; i < nPoints; i++) {
        ctx.lineTo(getXPix(i / (nPoints - 1)), getYPix(pts[i]));
      }
      ctx.lineTo(getXPix(1), margin.top + plotH);
      ctx.lineTo(getXPix(0), margin.top + plotH);
      ctx.closePath();
      ctx.fill();

      // Glow under main curve
      ctx.save();
      ctx.shadowColor = "rgba(184, 93, 54, 0.35)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 2;
      ctx.strokeStyle = "#B85D36";
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let i = 0; i < nPoints; i++) {
        const px = getXPix(i / (nPoints - 1));
        const py = getYPix(pts[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();

      // Crisp main curve on top (no shadow)
      ctx.strokeStyle = "#B85D36";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < nPoints; i++) {
        const px = getXPix(i / (nPoints - 1));
        const py = getYPix(pts[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Shock front dashed vertical marker
      if (shockX >= 0.02 && shockX <= 0.98) {
        const sx = getXPix(shockX);
        ctx.strokeStyle = "rgba(56, 112, 74, 0.6)";
        ctx.lineWidth = 1.3;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(sx, margin.top);
        ctx.lineTo(sx, margin.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Shock label
        ctx.fillStyle = "#2E6343";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`x=${shockX.toFixed(2)}`, sx, margin.top - 5);
      }

      // Axis labels
      ctx.fillStyle = "#8F8274";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      for (let i = 0; i <= 6; i++) {
        ctx.fillText((i / 6).toFixed(2), margin.left + (plotW / 6) * i, height - margin.bottom + 13);
      }
      ctx.textAlign = "right";
      for (let i = 0; i <= 4; i++) {
        const val = 1.2 - (2.0 / 4) * i;
        ctx.fillText(val.toFixed(1), margin.left - 6, margin.top + (plotH / 4) * i + 3.5);
      }

      animIdRef.current = requestAnimationFrame(loop);
    };

    animIdRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animIdRef.current);
  }, [setupCanvas]);

  // Allow manual slider to override waveTime ref
  const handleTimeSlider = (val: number) => {
    waveTimeRef.current = val;
    setDisplayTime(val);
  };

  return (
    <div>
      <div className="page-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2>Physics Surrogate Laboratory</h2>
            <p>Graph Neural Network Surrogate for Burgers&apos; Equation &amp; Non-Linear Continuum Mechanics.</p>
          </div>
          <span className="badge badge-sage" style={{ padding: "5px 12px", fontSize: 12 }}>
            ● Model Status: Operational
          </span>
        </div>
      </div>

      <div className="page-body">
        {/* Metric cards */}
        <div className="grid-4" style={{ marginBottom: 24 }}>
          {stats.map((s) => (
            <div key={s.label} className={`metric-card ${s.accent}`}>
              <div className="card-title">{s.label}</div>
              <div className="card-value" style={{ fontSize: 20 }}>{s.value}</div>
              <div className="card-sub">{s.sub}</div>
            </div>
          ))}
        </div>

        {/* Interactive Fluid Playground Widget */}
        <div className="card" style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 14 }}>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                <Waves size={17} color="var(--accent-terracotta)" />
                Continuum Flow Dynamics Sandbox
              </h3>
              <p style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>
                Real-time Burgers&apos; wave profile evolution and fluid particle compressibility.
              </p>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="btn btn-secondary"
                style={{ padding: "5px 12px", fontSize: 12 }}
              >
                {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                {isPlaying ? " Pause Flow" : " Animate"}
              </button>
            </div>
          </div>

          <div style={{ borderRadius: "var(--radius)", overflow: "hidden", border: "1px solid var(--border)", marginBottom: 14 }}>
            <canvas
              ref={sandboxCanvasRef}
              style={{ width: "100%", height: "220px", display: "block" }}
            />
          </div>

          <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap", fontSize: 12, color: "var(--text-secondary)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 200 }}>
              <span style={{ fontWeight: 600 }}>Temporal Evolution (t):</span>
              <input
                type="range"
                min={0}
                max={1.2}
                step={0.005}
                value={displayTime}
                onChange={(e) => handleTimeSlider(Number(e.target.value))}
                style={{ flex: 1, accentColor: "var(--accent-terracotta)" }}
              />
              <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{displayTime.toFixed(2)}s</span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 200 }}>
              <span style={{ fontWeight: 600 }}>Gradient Steepness:</span>
              <input
                type="range"
                min={6}
                max={28}
                value={shockSteepness}
                onChange={(e) => setShockSteepness(Number(e.target.value))}
                style={{ flex: 1, accentColor: "var(--accent-terracotta)" }}
              />
              <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{shockSteepness}</span>
            </div>
          </div>
        </div>

        {/* Architecture flow block */}
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", marginBottom: 12 }}>
            Surrogate Computation Graph
          </h3>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12.5,
            }}
          >
            {["Input State (x_t, mesh)", "→", "Encoder MLP", "→", "10× Message-Passing Processor", "→", "Decoder MLP", "→", "Predicted State (x_t+1)"].map((s, i) => (
              <span
                key={i}
                style={{
                  padding: s === "→" ? "0 4px" : "6px 14px",
                  borderRadius: s === "→" ? 0 : "var(--radius-sm)",
                  background: s === "→" ? "transparent" : "var(--sand-100)",
                  border: s === "→" ? "none" : "1px solid var(--sand-200)",
                  color: s === "→" ? "var(--sand-500)" : "var(--sand-800)",
                  fontWeight: s === "→" ? 400 : 600,
                }}
              >
                {s}
              </span>
            ))}
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: 13, marginTop: 12, lineHeight: 1.6, maxWidth: 700 }}>
            The <strong>MeshGraphNet</strong> models physical dynamics by iteratively passing message vectors along mesh edges. 
            Residual node and edge blocks preserve high-frequency shock gradient propagation across 10 message-passing rounds without vanishing gradients.
          </p>
        </div>

        {/* Feature quick links */}
        <div className="grid-2" style={{ gap: 18 }}>
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <Link key={f.href} href={f.href} style={{ textDecoration: "none", color: "inherit" }}>
                <div
                  className="card"
                  style={{
                    cursor: "pointer",
                    height: "100%",
                    transition: "transform 0.12s, border-color 0.12s",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <Icon size={18} color="var(--accent-terracotta)" />
                    <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)" }}>
                      {f.title}
                    </h3>
                  </div>
                  <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: 14 }}>
                    {f.description}
                  </p>
                  <span style={{ fontSize: 12.5, fontWeight: 600, color: "var(--accent-terracotta)", display: "inline-flex", alignItems: "center", gap: 5 }}>
                    Open Module <ArrowRight size={13} />
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}

