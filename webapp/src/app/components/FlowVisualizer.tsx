"use client";
import React, { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, Sliders, Waves, Activity, BarChart2, Volume2, Sparkles } from "lucide-react";

export interface StepResult {
  t: number;
  input?: number[];
  true: number[];
  predicted: number[];
  mse: number;
  relative_mse: number;
}

interface FlowVisualizerProps {
  xCoord: number[];
  steps: StepResult[];
  activeStep?: number;
  onStepChange?: (index: number) => void;
  title?: string;
}

type ColorMapType = "thermal" | "viridis" | "diverging";

interface TracerParticle {
  x: number;
  y: number;
  age: number;
  maxAge: number;
  trail: Array<{ x: number; y: number }>;
  speedMult: number;
  size: number;
}

export default function FlowVisualizer({
  xCoord,
  steps,
  activeStep = 0,
  onStepChange,
  title = "Continuum Velocity Dynamics",
}: FlowVisualizerProps) {
  // Continuous temporal time: fractional step float [0, steps.length - 1]
  const [continuousTime, setContinuousTime] = useState<number>(activeStep);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<"wave" | "heatmap" | "metrics">("wave");
  const [heatmapView, setHeatmapView] = useState<"pred" | "true" | "error">("pred");
  const [colorMap, setColorMap] = useState<ColorMapType>("thermal");

  // Display toggles
  const [showTrue, setShowTrue] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [showError, setShowError] = useState(true);
  const [showTrails, setShowTrails] = useState(true);
  const [showShockFront, setShowShockFront] = useState(true);

  // Hover state ref for 60fps crosshairs without triggering React re-renders
  const hoverDataRef = useRef<{
    x: number;
    trueVal: number;
    predVal: number;
    error: number;
    grad: number;
    canvasX: number;
  } | null>(null);

  // Interactive wave perturbation on user click
  const rippleRef = useRef<{ x: number; amp: number; decay: number } | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hudRef = useRef<HTMLDivElement | null>(null);
  const sliderRef = useRef<HTMLInputElement | null>(null);
  const timeLabelRef = useRef<HTMLSpanElement | null>(null);
  const timeBadgeRef = useRef<HTMLSpanElement | null>(null);

  const continuousTimeRef = useRef<number>(activeStep);
  const playDirectionRef = useRef<number>(1);
  const frameTimeRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(isPlaying);
  const playbackSpeedRef = useRef<number>(playbackSpeed);

  // Canvas size and DPR cache to eliminate getBoundingClientRect layout thrashing
  const canvasSizeRef = useRef<{ width: number; height: number; dpr: number; left: number; top: number } | null>(null);

  // Render flags ref to prevent tearing down the RAF loop on toggle
  const flagsRef = useRef({
    showTrue,
    showPred,
    showError,
    showTrails,
    showShockFront,
  });
  useEffect(() => {
    flagsRef.current = { showTrue, showPred, showError, showTrails, showShockFront };
  }, [showTrue, showPred, showError, showTrails, showShockFront]);

  // Sync refs
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);
  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
  }, [playbackSpeed]);

  // Sync external activeStep changes (e.g. from step buttons)
  useEffect(() => {
    continuousTimeRef.current = activeStep;
    if (sliderRef.current) sliderRef.current.value = String(activeStep);
    if (timeLabelRef.current) timeLabelRef.current.textContent = `t = ${activeStep.toFixed(2)} / ${Math.max(1, steps.length - 1)}`;
    if (timeBadgeRef.current) timeBadgeRef.current.textContent = `60-120 FPS Ultra-Smooth Physical Dynamics · Frame ${activeStep.toFixed(2)}`;
    setContinuousTime(activeStep);
  }, [activeStep, steps.length]);

  // Pre-allocated typed arrays and point buffers to eliminate Garbage Collection micro-stutters
  const trueValsBufRef = useRef<Float32Array | null>(null);
  const predValsBufRef = useRef<Float32Array | null>(null);
  const predPointsBufRef = useRef<Array<{ x: number; y: number }>>([]);
  const truePointsBufRef = useRef<Array<{ x: number; y: number }>>([]);

  // Tracer particles with motion trails
  const particlesRef = useRef<TracerParticle[]>([]);
  // Low-frequency React state sync (every ~150ms) to keep non-canvas UI updated without dropped frames
  const displayUpdateRef = useRef<number>(0);

  useEffect(() => {
    const pts: TracerParticle[] = [];
    const count = 75;
    for (let i = 0; i < count; i++) {
      const maxAge = 60 + Math.floor(Math.random() * 80);
      pts.push({
        x: Math.random(),
        y: Math.random(),
        age: Math.floor(Math.random() * maxAge),
        maxAge,
        trail: [],
        speedMult: Math.random() * 0.4 + 0.8,
        size: 1.1 + Math.random() * 1.3,
      });
    }
    particlesRef.current = pts;
  }, []);

  // Compute global bounds
  const { minVal, maxVal, maxErr } = useMemo(() => {
    let minV = Infinity;
    let maxV = -Infinity;
    let maxE = 0;
    for (const s of steps) {
      for (let i = 0; i < s.true.length; i++) {
        const tv = s.true[i] ?? 0;
        const pv = s.predicted[i] ?? 0;
        if (tv < minV) minV = tv;
        if (tv > maxV) maxV = tv;
        if (pv < minV) minV = pv;
        if (pv > maxV) maxV = pv;
        const err = Math.abs(tv - pv);
        if (err > maxE) maxE = err;
      }
    }
    const pad = (maxV - minV) * 0.08 || 0.1;
    return { minVal: minV - pad, maxVal: maxV + pad, maxErr: maxE || 0.05 };
  }, [steps]);

  // Color map calculation
  const getColor = useCallback((tNorm: number, map: ColorMapType): [number, number, number] => {
    const v = Math.max(0, Math.min(1, tNorm));
    if (map === "thermal") {
      if (v < 0.35) {
        const f = v / 0.35;
        return [251 - 45 * f, 247 - 80 * f, 240 - 120 * f];
      } else if (v < 0.7) {
        const f = (v - 0.35) / 0.35;
        return [206 - 22 * f, 167 - 85 * f, 120 - 82 * f];
      } else {
        const f = (v - 0.7) / 0.3;
        return [184 - 158 * f, 82 - 59 * f, 38 - 15 * f];
      }
    } else if (map === "viridis") {
      const r = Math.round(68 + v * (253 - 68));
      const g = Math.round(1 + v * (231 - 1));
      const b = Math.round(84 + (1 - Math.abs(v - 0.5) * 2) * 110);
      return [r, g, b];
    } else {
      if (v < 0.5) {
        const f = v / 0.5;
        return [
          Math.round(46 + (251 - 46) * f),
          Math.round(99 + (249 - 99) * f),
          Math.round(67 + (245 - 67) * f),
        ];
      } else {
        const f = (v - 0.5) / 0.5;
        return [
          Math.round(251 - (251 - 184) * f),
          Math.round(249 - (249 - 82) * f),
          Math.round(245 - (245 - 38) * f),
        ];
      }
    }
  }, []);

  // High-DPI setup helper for static canvases (e.g. Space-Time Waterfall)
  const setupCanvas = useCallback((canvas: HTMLCanvasElement) => {
    const dpr = typeof window !== "undefined" ? Math.min(2, window.devicePixelRatio || 1) : 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;

    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.resetTransform?.();
    ctx.scale(dpr, dpr);
    return { ctx, width: rect.width, height: rect.height };
  }, []);

  // ResizeObserver for zero-overhead size caching
  useEffect(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas || activeTab !== "wave") return;

    const measure = () => {
      const dpr = typeof window !== "undefined" ? Math.min(2, window.devicePixelRatio || 1) : 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      const targetW = Math.round(rect.width * dpr);
      const targetH = Math.round(rect.height * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      canvasSizeRef.current = {
        width: rect.width,
        height: rect.height,
        dpr,
        left: rect.left,
        top: rect.top,
      };
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(canvas);
    window.addEventListener("resize", measure, { passive: true });
    window.addEventListener("scroll", measure, { passive: true });

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure);
    };
  }, [activeTab]);

  // Compute smooth time-interpolated velocity array at continuous time t without heap allocations
  const getInterpolatedFieldFast = useCallback((tFloat: number) => {
    const nSteps = steps.length;
    if (nSteps === 0) return { nPoints: 0, shockX: 0.5, maxGrad: 0 };

    const clampedT = Math.max(0, Math.min(nSteps - 1, tFloat));
    const k0 = Math.floor(clampedT);
    const k1 = Math.min(nSteps - 1, k0 + 1);
    const alpha = clampedT - k0; // [0, 1)

    const step0 = steps[k0];
    const step1 = steps[k1];
    const n = step0.true.length;

    // Ensure preallocated buffers are appropriately sized
    if (!trueValsBufRef.current || trueValsBufRef.current.length !== n) {
      trueValsBufRef.current = new Float32Array(n);
      predValsBufRef.current = new Float32Array(n);
    }
    const trueVals = trueValsBufRef.current!;
    const predVals = predValsBufRef.current!;

    let maxGrad = 0;
    let shockX = 0.5;
    const ripple = rippleRef.current;

    // Cosine ease alpha for buttery wave continuity
    const easeAlpha = 0.5 * (1 - Math.cos(Math.PI * alpha));

    for (let i = 0; i < n; i++) {
      const tv0 = step0.true[i];
      const tv1 = step1.true[i];
      const pv0 = step0.predicted[i];
      const pv1 = step1.predicted[i];

      let tv = tv0 + (tv1 - tv0) * easeAlpha;
      let pv = pv0 + (pv1 - pv0) * easeAlpha;

      if (ripple) {
        const xNorm = i / (n - 1);
        const dist = Math.abs(xNorm - ripple.x);
        const pert = ripple.amp * Math.exp(-dist * dist * 120) * Math.sin(dist * 35);
        tv += pert;
        pv += pert;
      }

      trueVals[i] = tv;
      predVals[i] = pv;

      if (i > 0) {
        const grad = Math.abs(pv - predVals[i - 1]);
        if (grad > maxGrad) {
          maxGrad = grad;
          shockX = (i - 0.5) / (n - 1);
        }
      }
    }

    return { nPoints: n, shockX, maxGrad: maxGrad * n };
  }, [steps]);

  // Main 60-120 FPS Wave Canvas Renderer
  const renderWaveCanvas = useCallback(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;

    const size = canvasSizeRef.current;
    if (!size || size.width === 0 || size.height === 0) return;

    const { width, height, dpr } = size;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const margin = { top: 22, right: 30, bottom: 44, left: 54 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    // Background
    ctx.fillStyle = "#FCFAF7";
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = "#EBE5DC";
    ctx.lineWidth = 1;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = margin.top + (plotH / yTicks) * i;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();

      const val = maxVal - ((maxVal - minVal) / yTicks) * i;
      ctx.fillStyle = "#8F8274";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(val.toFixed(2), margin.left - 8, y + 3.5);
    }

    const xTicks = 8;
    for (let i = 0; i <= xTicks; i++) {
      const x = margin.left + (plotW / xTicks) * i;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, height - margin.bottom);
      ctx.stroke();

      const xVal = (i / xTicks).toFixed(2);
      ctx.fillStyle = "#8F8274";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(xVal, x, height - margin.bottom + 15);
    }

    // Zero baseline
    const zeroY = margin.top + ((maxVal - 0) / (maxVal - minVal)) * plotH;
    if (zeroY >= margin.top && zeroY <= height - margin.bottom) {
      ctx.strokeStyle = "#DCD5C9";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(margin.left, zeroY);
      ctx.lineTo(width - margin.right, zeroY);
      ctx.stroke();
    }

    // Get continuous interpolated field into preallocated buffers
    const { nPoints, shockX } = getInterpolatedFieldFast(continuousTimeRef.current);
    if (nPoints === 0) return;

    const trueVals = trueValsBufRef.current!;
    const predVals = predValsBufRef.current!;

    const getXPix = (idx: number) => margin.left + (idx / (nPoints - 1)) * plotW;
    const getYPix = (val: number) => margin.top + ((maxVal - val) / (maxVal - minVal)) * plotH;

    // Helper to draw smooth Bézier curves through points for silky fluid rendering
    const drawSmooth = (points: { x: number; y: number }[]) => {
      if (points.length === 0) return;
      if (points.length === 1) { ctx.moveTo(points[0].x, points[0].y); return; }
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 0; i < points.length - 1; i++) {
        const xc = (points[i].x + points[i + 1].x) * 0.5;
        const yc = (points[i].y + points[i + 1].y) * 0.5;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
      }
      ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    };

    // Re-use points buffers without GC churn
    let predPoints = predPointsBufRef.current;
    let truePoints = truePointsBufRef.current;
    if (predPoints.length !== nPoints) {
      predPoints = new Array(nPoints);
      truePoints = new Array(nPoints);
      for (let i = 0; i < nPoints; i++) {
        predPoints[i] = { x: 0, y: 0 };
        truePoints[i] = { x: 0, y: 0 };
      }
      predPointsBufRef.current = predPoints;
      truePointsBufRef.current = truePoints;
    }

    for (let i = 0; i < nPoints; i++) {
      const px = getXPix(i);
      predPoints[i].x = px;
      predPoints[i].y = getYPix(predVals[i]);
      truePoints[i].x = px;
      truePoints[i].y = getYPix(trueVals[i]);
    }

    const { showTrue: fShowTrue, showPred: fShowPred, showError: fShowError, showTrails: fShowTrails, showShockFront: fShowShock } = flagsRef.current;

    // 1. Shaded Error Ribbon
    if (fShowError && fShowTrue && fShowPred) {
      ctx.fillStyle = "rgba(181, 53, 38, 0.09)";
      ctx.beginPath();
      drawSmooth(truePoints);
      for (let i = nPoints - 1; i >= 0; i--) {
        ctx.lineTo(predPoints[i].x, predPoints[i].y);
      }
      ctx.closePath();
      ctx.fill();
    }

    // 2. Smooth Fluid Body Gradient beneath prediction
    if (fShowPred) {
      const fluidGrad = ctx.createLinearGradient(0, margin.top, 0, height - margin.bottom);
      fluidGrad.addColorStop(0, "rgba(184, 82, 38, 0.16)");
      fluidGrad.addColorStop(0.6, "rgba(184, 82, 38, 0.05)");
      fluidGrad.addColorStop(1, "rgba(184, 82, 38, 0.00)");

      ctx.fillStyle = fluidGrad;
      ctx.beginPath();
      drawSmooth(predPoints);
      ctx.lineTo(getXPix(nPoints - 1), height - margin.bottom);
      ctx.lineTo(getXPix(0), height - margin.bottom);
      ctx.closePath();
      ctx.fill();
    }

    // 3. Fluid Streamline Streaklines (Laminar Flow Particles with Motion Ribbons)
    if (fShowTrails) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(margin.left, margin.top, plotW, plotH);
      ctx.clip();

      const pts = particlesRef.current;
      const bottomY = height - margin.bottom;

      for (let pi = 0; pi < pts.length; pi++) {
        const p = pts[pi];
        p.age++;
        if (p.age >= p.maxAge || p.x > 1.02 || p.x < -0.02) {
          p.x = -0.02 + Math.random() * 0.15;
          p.y = Math.random();
          p.age = 0;
          p.trail = [];
          p.maxAge = 65 + Math.floor(Math.random() * 80);
        }

        const idx = Math.min(nPoints - 1, Math.max(0, Math.floor(Math.max(0, p.x) * (nPoints - 1))));
        const uLocal = predVals[idx] ?? 0.5;

        const uBulk = 0.44;
        const uEffective = Math.max(0.14, uBulk + uLocal * 0.55);
        const moveDist = uEffective * 0.0048 * playbackSpeedRef.current * p.speedMult;
        p.x += moveDist;

        const px = margin.left + p.x * plotW;
        const surfY = getYPix(uLocal);
        const py = surfY + 4 + p.y * Math.max(12, bottomY - surfY - 8);

        p.trail.push({ x: px, y: py });
        if (p.trail.length > 5) p.trail.shift();

        if (p.trail.length > 1) {
          ctx.beginPath();
          ctx.moveTo(p.trail[0].x, p.trail[0].y);
          for (let ti = 1; ti < p.trail.length; ti++) {
            ctx.lineTo(p.trail[ti].x, p.trail[ti].y);
          }
          const speedRatio = Math.min(1, Math.max(0, (uEffective - 0.2) / 0.8));
          const trailAlpha = (0.18 + speedRatio * 0.22) * (1 - p.age / p.maxAge);
          ctx.strokeStyle = `rgba(184, 82, 38, ${trailAlpha})`;
          ctx.lineWidth = 1.2;
          ctx.lineCap = "round";
          ctx.stroke();
        }

        const headRadius = p.size * (0.8 + Math.min(0.4, uEffective * 0.4));
        const headAlpha = Math.min(0.85, 0.4 + (1 - p.age / p.maxAge) * 0.45);
        ctx.fillStyle = `rgba(184, 82, 38, ${headAlpha})`;
        ctx.beginPath();
        ctx.arc(px, py, headRadius, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // 4. Dynamic Shock Front Discontinuity Indicator
    if (fShowShock && shockX >= 0.02 && shockX <= 0.98) {
      const sx = margin.left + shockX * plotW;
      ctx.save();
      ctx.strokeStyle = "rgba(46, 99, 67, 0.65)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx, margin.top);
      ctx.lineTo(sx, height - margin.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#2E6343";
      ctx.font = "9.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`Shock Front (x=${shockX.toFixed(2)})`, sx, margin.top - 6);
      ctx.restore();
    }

    // 5. Ground Truth Curve (Fast Hardware Glow Layer instead of slow shadowBlur)
    if (fShowTrue) {
      // Wide soft bloom
      ctx.strokeStyle = "rgba(46, 99, 67, 0.18)";
      ctx.lineWidth = 7.0;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      drawSmooth(truePoints);
      ctx.stroke();

      // Core glow
      ctx.strokeStyle = "rgba(46, 99, 67, 0.4)";
      ctx.lineWidth = 3.8;
      ctx.beginPath();
      drawSmooth(truePoints);
      ctx.stroke();

      // Crisp foreground stroke
      ctx.strokeStyle = "#2E6343";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      drawSmooth(truePoints);
      ctx.stroke();
    }

    // 6. Surrogate / TTA Prediction Curve (Fast Hardware Glow Layer)
    if (fShowPred) {
      // Wide soft bloom
      ctx.strokeStyle = "rgba(184, 82, 38, 0.20)";
      ctx.lineWidth = 8.0;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      drawSmooth(predPoints);
      ctx.stroke();

      // Core glow
      ctx.strokeStyle = "rgba(184, 82, 38, 0.45)";
      ctx.lineWidth = 4.0;
      ctx.beginPath();
      drawSmooth(predPoints);
      ctx.stroke();

      // Crisp foreground stroke
      ctx.strokeStyle = "#B85226";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      drawSmooth(predPoints);
      ctx.stroke();
    }

    // 7. Interactive Crosshair & Cursor HUD (Read directly from ref for 60-120fps tracking)
    const hData = hoverDataRef.current;
    if (hData) {
      ctx.strokeStyle = "#8F8274";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hData.canvasX, margin.top);
      ctx.lineTo(hData.canvasX, height - margin.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      if (fShowTrue) {
        ctx.fillStyle = "#2E6343";
        ctx.beginPath();
        ctx.arc(hData.canvasX, getYPix(hData.trueVal), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      if (fShowPred) {
        ctx.fillStyle = "#B85226";
        ctx.beginPath();
        ctx.arc(hData.canvasX, getYPix(hData.predVal), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // Axis Titles
    ctx.fillStyle = "#5C5247";
    ctx.font = "11px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Spatial Coordinate x ∈ [0, 1] — Click anywhere to perturb fluid wave", margin.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Fluid Velocity u(x, t)", 0, 0);
    ctx.restore();
  }, [getInterpolatedFieldFast, minVal, maxVal]);

  // Continuous 60-120 FPS High-Performance Animation Loop
  useEffect(() => {
    let animId: number;
    let lastTimestamp = performance.now();

    const loop = (timestamp: number) => {
      // Sub-millisecond delta time calculation with 120Hz/144Hz monitor support
      const deltaSec = Math.min(0.033, (timestamp - lastTimestamp) / 1000);
      lastTimestamp = timestamp;
      frameTimeRef.current += deltaSec;

      if (isPlayingRef.current && steps.length > 1) {
        const speedRate = 0.95 * playbackSpeedRef.current; // steps per second
        let nextTime = continuousTimeRef.current + speedRate * deltaSec * playDirectionRef.current;
        if (nextTime >= steps.length - 1) {
          nextTime = steps.length - 1;
          playDirectionRef.current = -1; // smoothly reverse / relax
        } else if (nextTime <= 0) {
          nextTime = 0;
          playDirectionRef.current = 1; // smoothly steepen shock
        }
        continuousTimeRef.current = nextTime;

        // Directly update DOM elements without causing React re-renders for buttery 60-120 FPS
        if (sliderRef.current && document.activeElement !== sliderRef.current) {
          sliderRef.current.value = String(nextTime);
        }
        if (timeLabelRef.current) {
          timeLabelRef.current.textContent = `t = ${nextTime.toFixed(2)} / ${steps.length - 1}`;
        }
        if (timeBadgeRef.current) {
          timeBadgeRef.current.textContent = `60-120 FPS Ultra-Smooth Physical Dynamics · Frame ${nextTime.toFixed(2)}`;
        }

        // Low-frequency React state sync (every ~160ms) to keep waterfall or external listeners happy
        if (timestamp - displayUpdateRef.current > 160) {
          displayUpdateRef.current = timestamp;
          setContinuousTime(nextTime);
        }
      }

      // Decay interactive ripple perturbation
      if (rippleRef.current) {
        rippleRef.current.amp *= 0.96;
        if (rippleRef.current.amp < 0.001) rippleRef.current = null;
      }

      // Render 1D Wave canvas if active
      if (activeTab === "wave") {
        renderWaveCanvas();
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [activeTab, steps.length, renderWaveCanvas]);

  // Handle Interactive Mouse Hover & Click Ripple with 0ms overhead
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const size = canvasSizeRef.current;
    if (!size) return;

    const clientX = e.clientX - size.left;
    const margin = { top: 22, right: 30, bottom: 44, left: 54 };
    const plotW = size.width - margin.left - margin.right;

    if (clientX >= margin.left && clientX <= size.width - margin.right) {
      const normX = (clientX - margin.left) / plotW;
      getInterpolatedFieldFast(continuousTimeRef.current);
      const trueVals = trueValsBufRef.current;
      const predVals = predValsBufRef.current;
      if (!trueVals || !predVals) return;

      const nPoints = trueVals.length;
      const idx = Math.min(nPoints - 1, Math.max(0, Math.round(normX * (nPoints - 1))));
      const tv = trueVals[idx];
      const pv = predVals[idx];
      const nextVal = predVals[Math.min(nPoints - 1, idx + 1)];
      const prevVal = predVals[Math.max(0, idx - 1)];
      const grad = (nextVal - prevVal) / (2.0 / (nPoints - 1));

      const hData = {
        x: normX,
        trueVal: tv,
        predVal: pv,
        error: Math.abs(tv - pv),
        grad: grad,
        canvasX: clientX,
      };
      hoverDataRef.current = hData;

      // Update HUD directly in DOM without React re-rendering
      if (hudRef.current) {
        hudRef.current.innerHTML = `
          <span>x: <strong>${normX.toFixed(3)}</strong></span>
          <span style="color: var(--accent-sage)">True: <strong>${tv.toFixed(4)}</strong></span>
          <span style="color: var(--accent-terracotta)">Pred: <strong>${pv.toFixed(4)}</strong></span>
          <span style="color: var(--accent-brick)">Δu: <strong>${Math.abs(tv - pv).toExponential(2)}</strong></span>
          <span style="color: var(--text-muted)">∂u/∂x: <strong>${grad.toFixed(1)}</strong></span>
        `;
      }
    } else {
      hoverDataRef.current = null;
      if (hudRef.current) {
        hudRef.current.innerHTML = `<span style="font-size: 11px; color: var(--text-muted)">Click canvas to trigger pressure ripple</span>`;
      }
    }
  };

  const handleCanvasMouseLeave = () => {
    hoverDataRef.current = null;
    if (hudRef.current) {
      hudRef.current.innerHTML = `<span style="font-size: 11px; color: var(--text-muted)">Click canvas to trigger pressure ripple</span>`;
    }
  };

  // Inject interactive physical wave ripple on click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const size = canvasSizeRef.current;
    if (!size) return;
    const margin = { top: 22, right: 30, bottom: 44, left: 54 };
    const plotW = size.width - margin.left - margin.right;
    const normX = Math.max(0, Math.min(1, (e.clientX - size.left - margin.left) / plotW));

    rippleRef.current = {
      x: normX,
      amp: 0.18,
      decay: 0.95,
    };
  };

  // Render Space-Time Waterfall Canvas
  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas || activeTab !== "heatmap" || steps.length === 0) return;

    const res = setupCanvas(canvas);
    if (!res) return;
    const { ctx, width, height } = res;

    const colorbarW = 16;
    const margin = { top: 18, right: 65, bottom: 38, left: 54 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    ctx.fillStyle = "#FCFAF7";
    ctx.fillRect(0, 0, width, height);

    const T = steps.length;
    const X = steps[0].true.length;
    const cellW = plotW / X;
    const cellH = plotH / T;

    for (let t = 0; t < T; t++) {
      const s = steps[t];
      for (let x = 0; x < X; x++) {
        let val = 0;
        let norm = 0;
        if (heatmapView === "true") {
          val = s.true[x] ?? 0;
          norm = (val - minVal) / (maxVal - minVal);
        } else if (heatmapView === "pred") {
          val = s.predicted[x] ?? 0;
          norm = (val - minVal) / (maxVal - minVal);
        } else {
          val = Math.abs((s.true[x] ?? 0) - (s.predicted[x] ?? 0));
          norm = val / maxErr;
        }

        const [r, g, b] = getColor(norm, colorMap);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(
          margin.left + x * cellW,
          margin.top + t * cellH,
          cellW + 0.5,
          cellH + 0.5
        );
      }
    }

    // Active Continuous Timestep Line
    const curY = margin.top + (continuousTime / (T - 1 || 1)) * plotH;
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(margin.left, curY);
    ctx.lineTo(width - margin.right, curY);
    ctx.stroke();

    ctx.strokeStyle = "#1A1715";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(margin.left, curY);
    ctx.lineTo(width - margin.right, curY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Colorbar scale on the right
    const cbX = width - margin.right + 18;
    const cbH = plotH;
    for (let y = 0; y < cbH; y++) {
      const norm = 1 - y / cbH;
      const [r, g, b] = getColor(norm, colorMap);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(cbX, margin.top + y, colorbarW, 1);
    }
    ctx.strokeStyle = "#E5E0D8";
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, margin.top, colorbarW, cbH);

    // Colorbar labels
    ctx.fillStyle = "#8F8274";
    ctx.font = "9.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    const topVal = heatmapView === "error" ? maxErr : maxVal;
    const botVal = heatmapView === "error" ? 0 : minVal;
    ctx.fillText(topVal.toFixed(2), cbX + colorbarW + 6, margin.top + 8);
    ctx.fillText(((topVal + botVal) / 2).toFixed(2), cbX + colorbarW + 6, margin.top + cbH / 2 + 3);
    ctx.fillText(botVal.toFixed(2), cbX + colorbarW + 6, margin.top + cbH);

    // Axes
    ctx.fillStyle = "#5C5247";
    ctx.font = "11px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Space Coordinate x ∈ [0, 1]", margin.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Temporal Horizon t", 0, 0);
    ctx.restore();
  }, [steps, activeTab, heatmapView, colorMap, minVal, maxVal, maxErr, continuousTime, setupCanvas, getColor]);

  // Click on Heatmap to jump
  const handleHeatmapClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const yPix = e.clientY - rect.top;
    const margin = { top: 18, right: 65, bottom: 38, left: 54 };
    const plotH = rect.height - margin.top - margin.bottom;

    if (yPix >= margin.top && yPix <= rect.height - margin.bottom) {
      const frac = (yPix - margin.top) / plotH;
      const tFloat = Math.max(0, Math.min(steps.length - 1, frac * (steps.length - 1)));
      continuousTimeRef.current = tFloat;
      setContinuousTime(tFloat);
      const rounded = Math.round(tFloat);
      if (onStepChange) onStepChange(rounded);
    }
  };

  const activeFloorStep = Math.round(continuousTime);
  const activeStepData = steps[activeFloorStep] || steps[0];

  return (
    <div className="card" ref={containerRef} style={{ padding: "18px 22px" }}>
      {/* Top Header Bar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
          paddingBottom: 14,
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Waves size={16} color="var(--accent-terracotta)" />
            <h3 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>
              {title}
            </h3>
            <span ref={timeBadgeRef} className="badge badge-gray" style={{ fontSize: 10.5 }}>
              60-120 FPS Ultra-Smooth Physical Dynamics · Frame {continuousTime.toFixed(2)}
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
            Smooth temporal continuum interpolation, fluid advection streaklines, and shock discontinuity tracking.
          </p>
        </div>

        {/* View Segmented Control */}
        <div style={{ display: "flex", gap: 3, background: "var(--bg-card-subtle)", padding: 3, borderRadius: 5, border: "1px solid var(--border)" }}>
          <button
            onClick={() => setActiveTab("wave")}
            className={`btn-subtle ${activeTab === "wave" ? "active" : ""}`}
            style={{ fontSize: 11.5, padding: "4px 10px" }}
          >
            <Activity size={12} style={{ marginRight: 5 }} />
            Continuous 60-120 FPS Wave
          </button>
          <button
            onClick={() => setActiveTab("heatmap")}
            className={`btn-subtle ${activeTab === "heatmap" ? "active" : ""}`}
            style={{ fontSize: 11.5, padding: "4px 10px" }}
          >
            <BarChart2 size={12} style={{ marginRight: 5 }} />
            Space-Time Waterfall
          </button>
        </div>
      </div>

      {/* Main Plot Area */}
      <div style={{ marginTop: 14 }}>
        {activeTab === "wave" && (
          <div>
            {/* Display Toggles & Floating Readout */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 12,
                marginBottom: 10,
                fontSize: 11.5,
              }}
            >
              <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox"
                    checked={showTrue}
                    onChange={(e) => setShowTrue(e.target.checked)}
                    style={{ accentColor: "var(--accent-sage)" }}
                  />
                  <span style={{ display: "inline-block", width: 10, height: 2.5, background: "var(--accent-sage)", borderRadius: 1 }} />
                  Ground Truth (u)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox"
                    checked={showPred}
                    onChange={(e) => setShowPred(e.target.checked)}
                    style={{ accentColor: "var(--accent-terracotta)" }}
                  />
                  <span style={{ display: "inline-block", width: 10, height: 2.5, background: "var(--accent-terracotta)", borderRadius: 1 }} />
                  Surrogate (û)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox"
                    checked={showError}
                    onChange={(e) => setShowError(e.target.checked)}
                    style={{ accentColor: "var(--accent-brick)" }}
                  />
                  <span style={{ display: "inline-block", width: 8, height: 8, background: "rgba(181, 53, 38, 0.2)", border: "1px solid var(--accent-brick)", borderRadius: 1 }} />
                  Error Envelope
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox"
                    checked={showTrails}
                    onChange={(e) => setShowTrails(e.target.checked)}
                    style={{ accentColor: "var(--accent-terracotta)" }}
                  />
                  Fluid Streaklines
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox"
                    checked={showShockFront}
                    onChange={(e) => setShowShockFront(e.target.checked)}
                    style={{ accentColor: "var(--accent-sage)" }}
                  />
                  Shock Marker
                </label>
              </div>

              {/* Precision HUD readout — Updated directly via DOM ref for 60-120fps responsiveness */}
              <div
                ref={hudRef}
                style={{
                  background: "var(--bg-card-subtle)",
                  border: "1px solid var(--border)",
                  padding: "3px 10px",
                  borderRadius: 4,
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 10.5,
                  display: "flex",
                  gap: 10,
                  minHeight: 24,
                  alignItems: "center",
                }}
              >
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Hover crosshairs to probe · Click wave to perturb</span>
              </div>
            </div>

            {/* Canvas Viewport */}
            <div
              style={{
                position: "relative",
                borderRadius: "var(--radius-sm)",
                overflow: "hidden",
                border: "1px solid var(--border)",
                background: "#FCFAF7",
              }}
            >
              <canvas
                ref={waveCanvasRef}
                style={{ width: "100%", height: "290px", display: "block", cursor: "crosshair" }}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
                onClick={handleCanvasClick}
              />
            </div>
          </div>
        )}

        {activeTab === "heatmap" && (
          <div>
            {/* Heatmap Controls */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 12,
                marginBottom: 10,
                fontSize: 11.5,
              }}
            >
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ color: "var(--text-muted)", fontWeight: 600, fontSize: 10.5 }}>FIELD:</span>
                <button
                  onClick={() => setHeatmapView("pred")}
                  className={`btn-subtle ${heatmapView === "pred" ? "active" : ""}`}
                  style={{ fontSize: 11, padding: "3px 9px" }}
                >
                  Surrogate Prediction (û)
                </button>
                <button
                  onClick={() => setHeatmapView("true")}
                  className={`btn-subtle ${heatmapView === "true" ? "active" : ""}`}
                  style={{ fontSize: 11, padding: "3px 9px" }}
                >
                  Ground Truth (u)
                </button>
                <button
                  onClick={() => setHeatmapView("error")}
                  className={`btn-subtle ${heatmapView === "error" ? "active" : ""}`}
                  style={{ fontSize: 11, padding: "3px 9px" }}
                >
                  Absolute Error |Δu|
                </button>
              </div>

              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ color: "var(--text-muted)", fontWeight: 600, fontSize: 10.5 }}>COLOR SCALE:</span>
                <select
                  value={colorMap}
                  onChange={(e) => setColorMap(e.target.value as ColorMapType)}
                  className="form-select"
                  style={{ fontSize: 11, padding: "3px 8px", width: "auto" }}
                >
                  <option value="thermal">Thermal (Bone → Terracotta → Espresso)</option>
                  <option value="viridis">Perceptually Uniform Viridis</option>
                  <option value="diverging">Cool-to-Warm (Pine → Bone → Terracotta)</option>
                </select>
              </div>
            </div>

            {/* Waterfall Canvas */}
            <div
              style={{
                position: "relative",
                borderRadius: "var(--radius-sm)",
                overflow: "hidden",
                border: "1px solid var(--border)",
                background: "#FCFAF7",
              }}
            >
              <canvas
                ref={heatmapCanvasRef}
                style={{ width: "100%", height: "290px", display: "block", cursor: "pointer" }}
                onClick={handleHeatmapClick}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: 8,
                  left: 60,
                  background: "rgba(255, 255, 255, 0.8)",
                  padding: "2px 6px",
                  borderRadius: 3,
                  fontSize: 10,
                  color: "var(--text-muted)",
                  fontFamily: "JetBrains Mono, monospace",
                }}
              >
                Click or drag across waterfall to scrub continuous timeline
              </div>
            </div>
          </div>
        )}

        {/* Timeline Navigation Bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginTop: 14,
            paddingTop: 12,
            borderTop: "1px solid var(--border)",
          }}
        >
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="btn btn-primary"
            style={{ padding: "6px 14px", fontSize: 12 }}
          >
            {isPlaying ? (
              <>
                <Pause size={12} style={{ marginRight: 4 }} /> Pause Flow
              </>
            ) : (
              <>
                <Play size={12} style={{ marginRight: 4 }} /> Animate 60 FPS
              </>
            )}
          </button>

          <button
            onClick={() => {
              continuousTimeRef.current = 0;
              if (sliderRef.current) sliderRef.current.value = "0";
              if (timeLabelRef.current) timeLabelRef.current.textContent = `t = 0.00 / ${steps.length - 1}`;
              if (timeBadgeRef.current) timeBadgeRef.current.textContent = `60-120 FPS Ultra-Smooth Physical Dynamics · Frame 0.00`;
              setContinuousTime(0);
              if (onStepChange) onStepChange(0);
            }}
            className="btn btn-secondary"
            style={{ padding: "6px 10px", fontSize: 12 }}
            title="Reset to t=0"
          >
            <RotateCcw size={12} />
          </button>

          {/* Smooth Continuous Scrubber Slider */}
          <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
            <input
              ref={sliderRef}
              type="range"
              min={0}
              max={steps.length - 1}
              step={0.01}
              defaultValue={continuousTime}
              onChange={(e) => {
                const val = Number(e.target.value);
                continuousTimeRef.current = val;
                if (timeLabelRef.current) timeLabelRef.current.textContent = `t = ${val.toFixed(2)} / ${steps.length - 1}`;
                if (timeBadgeRef.current) timeBadgeRef.current.textContent = `60-120 FPS Ultra-Smooth Physical Dynamics · Frame ${val.toFixed(2)}`;
                setContinuousTime(val);
                if (onStepChange) onStepChange(Math.round(val));
              }}
              style={{
                flex: 1,
                accentColor: "var(--accent-terracotta)",
                cursor: "pointer",
              }}
            />
            <span
              ref={timeLabelRef}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11.5,
                fontWeight: 500,
                color: "var(--text-primary)",
                minWidth: 80,
                textAlign: "right",
              }}
            >
              t = {continuousTime.toFixed(2)} / {steps.length - 1}
            </span>
          </div>

          {/* Speed Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
            {[0.5, 1, 1.5, 2].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaybackSpeed(spd)}
                className={`btn-subtle ${playbackSpeed === spd ? "active" : ""}`}
                style={{ padding: "3px 8px", fontSize: 10.5 }}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
