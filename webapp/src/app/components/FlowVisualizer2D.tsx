"use client";
import React, { useRef, useEffect, useState, useCallback } from "react";
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Compass,
  Columns,
  LineChart,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Info,
  Layers,
  Sparkles,
  Move,
  RotateCw,
  Sliders,
  RefreshCw,
  Cpu,
} from "lucide-react";

interface VectorPoint {
  ix: number;
  iy: number;
  x_norm: number;
  y_norm: number;
  is_obs: boolean;
  vx: number;
  vy: number;
  speed: number;
  base_vx: number;
  base_vy: number;
  base_speed: number;
}

interface CenterlinePoint {
  x: number;
  col: number;
  ground_truth: number;
  baseline: number;
  adapted: number;
  is_obstacle: boolean;
}

interface AnimationFrame {
  timestep: number;
  adapted: number[][];
  baseline: number[][];
  ground_truth: number[][];
}

interface Flow2DSlice {
  shape: string;
  equation?: string;
  trajectory?: number;
  timestep: number;
  total_timesteps: number;
  grid: [number, number];
  min_val?: number;
  max_val?: number;
  mask: boolean[][];
  ground_truth: number[][];
  baseline: number[][];
  adapted: number[][];
  centerline?: CenterlinePoint[];
  vectors?: VectorPoint[];
  physics_metrics?: {
    baseline_divergence: number;
    adapted_divergence: number;
    divergence_drop_pct: number;
    baseline_boundary_leak: number;
    adapted_boundary_leak: number;
    wake_mse_improvement_pct: number;
    obstacle_center?: [number, number];
    physics?: string;
    model?: string;
    truth_residual?: number;
    baseline_rel_l2?: number;
    adapted_rel_l2?: number;
    held_out_trajectory?: boolean;
  };
  frames?: AnimationFrame[];
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
  const [diagramMode, setDiagramMode] = useState<"quiver" | "dual" | "centerline">("quiver");
  const [viewVariant, setViewVariant] = useState<"adapted" | "baseline" | "ground_truth">("adapted");
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [showStreamlines, setShowStreamlines] = useState<boolean>(true);
  const [data, setData] = useState<Flow2DSlice | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [promptShape, setPromptShape] = useState<string | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);

  // ── Custom Obstacle Placement Sandbox State ──
  const [isSandboxMode, setIsSandboxMode] = useState<boolean>(false);
  const [customShape, setCustomShape] = useState<string>("circle");
  const [draggedShape, setDraggedShape] = useState<string | null>(null);
  const [customX, setCustomX] = useState<number>(0.28);
  const [customY, setCustomY] = useState<number>(0.50);
  const [customSize, setCustomSize] = useState<number>(1.0);
  const [customAngle, setCustomAngle] = useState<number>(0.0);
  const [customReynolds, setCustomReynolds] = useState<number>(120.0);
  const [isSimulatingCustom, setIsSimulatingCustom] = useState<boolean>(false);
  const [customAnimationFrames, setCustomAnimationFrames] = useState<AnimationFrame[] | null>(null);
  const [isDraggingObstacle, setIsDraggingObstacle] = useState<boolean>(false);

  // Hover Probe State
  const [hoverProbe, setHoverProbe] = useState<{
    xNorm: number;
    yNorm: number;
    canvasX: number;
    canvasY: number;
    speed: number;
    vx: number;
    vy: number;
    isObstacle: boolean;
    divBaseline: number;
    divAdapted: number;
  } | null>(null);

  const [chartHoverIdx, setChartHoverIdx] = useState<number | null>(null);
  const [showLegendDetails, setShowLegendDetails] = useState<boolean>(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dualCanvasLeftRef = useRef<HTMLCanvasElement | null>(null);
  const dualCanvasRightRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dualOffscreenLeftRef = useRef<HTMLCanvasElement | null>(null);
  const dualOffscreenRightRef = useRef<HTMLCanvasElement | null>(null);
  const dragThrottleRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const sliceCacheRef = useRef<Record<string, Flow2DSlice>>({});

  // Fetch standard predefined slice from API with in-memory caching & background prefetch
  const fetchSlice = useCallback(async (s: string, t: number) => {
    const cacheKey = `${s}_${t}`;
    if (sliceCacheRef.current[cacheKey]) {
      setData(sliceCacheRef.current[cacheKey]);
      setError(null);
      setPromptShape(null);
      setCustomAnimationFrames(null);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:3001/api/flow-2d/slice?shape=${s}&timestep=${t}`);
      const json = await res.json();
      if (json.error) {
        setError(json.error);
        setPromptShape(s);
        setData(null);
      } else {
        sliceCacheRef.current[cacheKey] = json;
        setData(json);
        setError(null);
        setPromptShape(null);
        setCustomAnimationFrames(null);

        // Background prefetch adjacent timesteps for seamless instant animation
        const total = json.total_timesteps || 13;
        for (let nextOffset = 1; nextOffset <= 3; nextOffset++) {
          const nextT = (t + nextOffset) % total;
          const nextKey = `${s}_${nextT}`;
          if (!sliceCacheRef.current[nextKey]) {
            fetch(`http://localhost:3001/api/flow-2d/slice?shape=${s}&timestep=${nextT}`)
              .then((r) => r.json())
              .then((preJson) => {
                if (!preJson.error) sliceCacheRef.current[nextKey] = preJson;
              })
              .catch(() => {});
          }
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch 2D slice");
      setPromptShape(s);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Run Custom Obstacle Simulation: LBM ground truth + trained FNO-2D + TTA
  const handleSimulateCustom = async (
    shapeParam?: string,
    xParam?: number,
    yParam?: number
  ) => {
    setIsSimulatingCustom(true);
    setError(null);
    const shapeToUse = shapeParam || customShape;
    const xToUse = xParam !== undefined ? xParam : customX;
    const yToUse = yParam !== undefined ? yParam : customY;
    try {
      const res = await fetch("http://localhost:3001/api/flow-2d/simulate-custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shape: shapeToUse === "circle" ? "cylinder" : shapeToUse,
          x: xToUse,
          y: yToUse,
          size: customSize,
          angle: customAngle,
          reynolds: customReynolds,
          tta_steps: 8,
          timesteps: 12,
        }),
      });

      const json = await res.json();
      if (json.error) {
        console.error("Simulation error: " + json.error);
        return;
      }

      // Store trajectory animation frames
      if (json.frames && json.frames.length > 0) {
        setCustomAnimationFrames(json.frames);
        const repFrame = json.frames[timestep % json.frames.length];
        setData({
          shape: shapeToUse,
          equation: `Custom ${shapeToUse.toUpperCase()} (x=${Math.round(xToUse * 100)}%, y=${Math.round(yToUse * 100)}%)`,
          timestep: timestep % json.frames.length,
          total_timesteps: json.frames.length,
          grid: json.grid,
          mask: json.mask,
          adapted: repFrame.adapted,
          baseline: repFrame.baseline,
          ground_truth: repFrame.ground_truth,
          centerline: json.centerline,
          vectors: json.vectors,
          physics_metrics: json.physics_metrics,
          frames: json.frames,
        });
      }
    } catch (err: unknown) {
      console.error("Simulation failed: " + (err instanceof Error ? err.message : "Unknown error"));
    } finally {
      setIsSimulatingCustom(false);
    }
  };

  const handleUserConfirmedGenerate = async () => {
    if (!promptShape) return;
    setGenerating(true);
    try {
      const res = await fetch("http://localhost:3001/api/generate-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: promptShape }),
      });
      const json = await res.json();
      if (json.success) {
        setError(null);
        setPromptShape(null);
        fetchSlice(shape, timestep);
      } else {
        alert(json.error || "Generation failed");
      }
    } catch (e: unknown) {
      alert("Error: " + (e instanceof Error ? e.message : "Network error"));
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    if (!isSandboxMode) {
      fetchSlice(shape, timestep);
    }
  }, [shape, timestep, fetchSlice, isSandboxMode]);

  // Update current frame from customAnimationFrames when scrubbing/playing
  useEffect(() => {
    if (isSandboxMode && customAnimationFrames && customAnimationFrames.length > 0) {
      const frameIdx = timestep % customAnimationFrames.length;
      const f = customAnimationFrames[frameIdx];
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          timestep: frameIdx,
          adapted: f.adapted,
          baseline: f.baseline,
          ground_truth: f.ground_truth,
        };
      });
    }
  }, [timestep, isSandboxMode, customAnimationFrames]);

  // Auto-play animation loop through timesteps — 180ms per timestep for fluid motion
  useEffect(() => {
    if (!isPlaying || !data) return;
    const interval = setInterval(() => {
      setTimestep((prev) => (prev + 1) % (data.total_timesteps || 13));
    }, 180);
    return () => clearInterval(interval);
  }, [isPlaying, data]);

  // Scientific Colormap: Deep Navy -> Teal -> Golden Amber -> Warm White
  const getVelocityColor = useCallback((val: number): [number, number, number] => {
    val = Math.max(0, Math.min(1, val));
    if (val < 0.25) {
      const f = val / 0.25;
      return [Math.round(18 + 10 * f), Math.round(30 + 55 * f), Math.round(75 + 50 * f)];
    } else if (val < 0.5) {
      const f = (val - 0.25) / 0.25;
      return [Math.round(28 + 22 * f), Math.round(85 + 75 * f), Math.round(125 + 20 * f)];
    } else if (val < 0.75) {
      const f = (val - 0.5) / 0.25;
      return [Math.round(50 + 155 * f), Math.round(160 + 35 * f), Math.round(145 - 75 * f)];
    } else {
      const f = (val - 0.75) / 0.25;
      return [Math.round(205 + 45 * f), Math.round(195 + 50 * f), Math.round(70 + 130 * f)];
    }
  }, []);

  // Pre-render the fluid velocity contour texture to an offscreen canvas once per timestep/data change
  useEffect(() => {
    if (!data) return;
    const [gridNy, gridNx] = data.grid;
    const field =
      viewVariant === "ground_truth"
        ? data.ground_truth
        : viewVariant === "baseline"
        ? data.baseline
        : data.adapted;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const off = offscreenCanvasRef.current;
    if (off.width !== gridNx || off.height !== gridNy) {
      off.width = gridNx;
      off.height = gridNy;
    }
    const octx = off.getContext("2d");
    if (!octx) return;

    const imgData = octx.createImageData(gridNx, gridNy);
    const buf = imgData.data;

    for (let gy = 0; gy < gridNy; gy++) {
      for (let gx = 0; gx < gridNx; gx++) {
        const idx = (gy * gridNx + gx) * 4;
        const isSolid = Boolean(data.mask[gy] && data.mask[gy][gx]);
        if (isSolid) {
          buf[idx] = 22;
          buf[idx + 1] = 20;
          buf[idx + 2] = 18;
          buf[idx + 3] = 255;
        } else {
          const val = field[gy] ? field[gy][gx] : 0;
          const [r, g, b] = getVelocityColor(val);
          buf[idx] = r;
          buf[idx + 1] = g;
          buf[idx + 2] = b;
          buf[idx + 3] = 255;
        }
      }
    }
    octx.putImageData(imgData, 0, 0);
  }, [data, viewVariant, getVelocityColor]);

  // Mode 1: Render Wind Tunnel Vector & Quiver Field
  useEffect(() => {
    if (diagramMode !== "quiver" || !data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const [gridNy, gridNx] = data.grid;

    const field =
      viewVariant === "ground_truth"
        ? data.ground_truth
        : viewVariant === "baseline"
        ? data.baseline
        : data.adapted;

    if (particlesRef.current.length < 90) {
      particlesRef.current = Array.from({ length: 90 }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: 1.8 + Math.random() * 2.2,
        vy: (Math.random() - 0.5) * 0.4,
        life: Math.random() * 80,
        maxLife: 80 + Math.random() * 50,
      }));
    }

    let isSubscribed = true;

    const render = () => {
      if (!isSubscribed) return;

      // 1. Hardware GPU Accelerated Velocity Contour & Obstacle Mask (runs in ~0.05ms)
      if (offscreenCanvasRef.current) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(offscreenCanvasRef.current, 0, 0, width, height);
      }

      // 2. Wind Tunnel Walls
      ctx.fillStyle = "rgba(15, 12, 10, 0.9)";
      ctx.fillRect(0, 0, width, 5);
      ctx.fillRect(0, height - 5, width, 5);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 18) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x + 8, 5);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, height - 5);
        ctx.lineTo(x + 8, height);
        ctx.stroke();
      }

      // Sandbox Mode: Real-time Shape Rendering, Guidelines & Drag Overlay
      if (isSandboxMode) {
        const targetPx = customX * width;
        const targetPy = customY * height;

        // Dotted placement guidelines across the tunnel
        ctx.strokeStyle = isDraggingObstacle ? "rgba(242, 169, 0, 0.9)" : "rgba(242, 169, 0, 0.4)";
        ctx.lineWidth = isDraggingObstacle ? 1.4 : 1.0;
        ctx.setLineDash([4, 4]);

        ctx.beginPath();
        ctx.moveTo(targetPx, 2);
        ctx.lineTo(targetPx, height - 2);
        ctx.moveTo(2, targetPy);
        ctx.lineTo(width - 2, targetPy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Render Geometric Obstacle Shape
        ctx.save();
        ctx.translate(targetPx, targetPy);
        ctx.rotate((customAngle * Math.PI) / 180);

        const base = 48 * customSize;
        ctx.fillStyle = isDraggingObstacle ? "rgba(242, 169, 0, 0.4)" : "#161311";
        ctx.strokeStyle = "#F2A900";
        ctx.lineWidth = isDraggingObstacle ? 3.0 : 2.2;
        if (isDraggingObstacle) {
          ctx.shadowColor = "#F2A900";
          ctx.shadowBlur = 18;
        }

        ctx.beginPath();
        if (customShape === "circle" || customShape === "cylinder") {
          ctx.arc(0, 0, base / 2, 0, Math.PI * 2);
        } else if (customShape === "square") {
          ctx.rect(-base / 2, -base / 2, base, base);
        } else if (customShape === "rectangle") {
          ctx.rect(-base * 0.75, -base * 0.35, base * 1.5, base * 0.7);
        } else if (customShape === "triangle") {
          const h = base * 0.85;
          ctx.moveTo(-h / 2, 0);
          ctx.lineTo(h / 2, -base / 2);
          ctx.lineTo(h / 2, base / 2);
          ctx.closePath();
        } else if (customShape === "diamond") {
          ctx.moveTo(-base * 0.65, 0);
          ctx.lineTo(0, -base * 0.5);
          ctx.lineTo(base * 0.65, 0);
          ctx.lineTo(0, base * 0.5);
          ctx.closePath();
        } else if (customShape === "hexagon") {
          const r = base / 2;
          for (let i = 0; i < 6; i++) {
            const a = (i * Math.PI) / 3;
            const x = r * Math.cos(a);
            const y = r * Math.sin(a);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
        } else {
          ctx.arc(0, 0, base / 2, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.stroke();

        // Center grip icon
        ctx.strokeStyle = isDraggingObstacle ? "#FFFFFF" : "rgba(242, 169, 0, 0.7)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(-6, 0);
        ctx.lineTo(6, 0);
        ctx.moveTo(0, -6);
        ctx.lineTo(0, 6);
        ctx.stroke();

        ctx.restore();

        // Position & Interaction Pill Badge
        const pillText = isDraggingObstacle
          ? `📍 Release to Drop (${Math.round(customX * 100)}%, ${Math.round(customY * 100)}%)`
          : `✋ Drag to Reposition (${Math.round(customX * 100)}%, ${Math.round(customY * 100)}%)`;

        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        const textWidth = ctx.measureText(pillText).width;
        const pillW = textWidth + 18;
        const pillH = 22;
        const pillX = Math.max(10, Math.min(width - pillW - 10, targetPx - pillW / 2));
        const pillY = Math.max(26, targetPy - (base / 2) - 20);

        ctx.fillStyle = isDraggingObstacle ? "#F2A900" : "rgba(22, 19, 17, 0.92)";
        ctx.strokeStyle = "#F2A900";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(pillX, pillY, pillW, pillH, 5);
        } else {
          ctx.rect(pillX, pillY, pillW, pillH);
        }
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isDraggingObstacle ? "#161311" : "#FFFFFF";
        ctx.fillText(pillText, pillX + 9, pillY + 15);
      }

      // 4. Streamlines
      if (showStreamlines) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
        ctx.shadowColor = "rgba(255, 255, 255, 0.9)";
        ctx.shadowBlur = 3;

        particlesRef.current.forEach((p) => {
          const gy = Math.min(gridNy - 1, Math.max(0, Math.floor((p.y / height) * gridNy)));
          const gx = Math.min(gridNx - 1, Math.max(0, Math.floor((p.x / width) * gridNx)));
          const isSolid = data.mask[gy] && data.mask[gy][gx];

          if (isSolid) {
            p.x = 0;
            p.y = Math.random() * height;
          }

          const localSpeed = field[gy] ? field[gy][gx] : 0.5;
          p.x += p.vx * (0.6 + localSpeed * 1.4);
          p.y += p.vy;

          p.life++;
          if (p.x > width || p.y < 5 || p.y > height - 5 || p.life > p.maxLife) {
            p.x = 0;
            p.y = 8 + Math.random() * (height - 16);
            p.life = 0;
          }

          ctx.beginPath();
          ctx.arc(p.x, p.y, 1.4, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.shadowBlur = 0;
      }

      // 5. Quiver Vector Arrows
      if (showVectors && data.vectors && data.vectors.length > 0) {
        data.vectors.forEach((vec) => {
          if (vec.is_obs) return;

          const px = vec.x_norm * width;
          const py = vec.y_norm * height;

          const useVx = viewVariant === "baseline" ? vec.base_vx : vec.vx;
          const useVy = viewVariant === "baseline" ? vec.base_vy : vec.vy;
          const useSpd = viewVariant === "baseline" ? vec.base_speed : vec.speed;

          const arrowLen = Math.max(7, Math.min(22, useSpd * 24));
          const angle = Math.atan2(useVy, useVx);

          const tipX = px + Math.cos(angle) * arrowLen;
          const tipY = py + Math.sin(angle) * arrowLen;

          const arrowColor =
            useSpd < 0.35
              ? "rgba(160, 215, 255, 0.75)"
              : useSpd < 0.7
              ? "rgba(255, 255, 255, 0.85)"
              : "rgba(255, 205, 90, 0.9)";

          ctx.strokeStyle = arrowColor;
          ctx.fillStyle = arrowColor;
          ctx.lineWidth = 1.3;

          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(tipX, tipY);
          ctx.stroke();

          const headLen = 4;
          const headAngle = 0.45;
          ctx.beginPath();
          ctx.moveTo(tipX, tipY);
          ctx.lineTo(
            tipX - headLen * Math.cos(angle - headAngle),
            tipY - headLen * Math.sin(angle - headAngle)
          );
          ctx.lineTo(
            tipX - headLen * Math.cos(angle + headAngle),
            tipY - headLen * Math.sin(angle + headAngle)
          );
          ctx.closePath();
          ctx.fill();
        });
      }

      // Annotations
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.font = "bold 10px 'JetBrains Mono', monospace";
      ctx.fillText("INFLOW U_∞ →", 12, height - 14);
      ctx.fillText("OUTFLOW WAKE →", width - 110, height - 14);

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      isSubscribed = false;
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [
    diagramMode,
    data,
    viewVariant,
    showVectors,
    showStreamlines,
    getVelocityColor,
    isSandboxMode,
    customShape,
    customX,
    customY,
    customSize,
    customAngle,
    isDraggingObstacle,
  ]);

  // Mode 2: Dual Canvas render
  useEffect(() => {
    if (diagramMode !== "dual" || !data) return;

    const renderPanel = (
      canvas: HTMLCanvasElement | null,
      field: number[][],
      label: string,
      isBaseline: boolean
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      const [gridNy, gridNx] = data.grid;

      const imgData = ctx.createImageData(w, h);
      const buf = imgData.data;

      for (let y = 0; y < h; y++) {
        const gyf = (y / h) * (gridNy - 1);
        const gy0 = Math.floor(gyf);
        const gy1 = Math.min(gridNy - 1, gy0 + 1);
        const wy = gyf - gy0;

        for (let x = 0; x < w; x++) {
          const gxf = (x / w) * (gridNx - 1);
          const gx0 = Math.floor(gxf);
          const gx1 = Math.min(gridNx - 1, gx0 + 1);
          const wx = gxf - gx0;

          const idx = (y * w + x) * 4;

          const isSolid = Boolean(data.mask[gy0] && data.mask[gy0][gx0]);
          if (isSolid) {
            if (isBaseline && field[gy0] && field[gy0][gx0] > 0.05) {
              buf[idx] = 160;
              buf[idx + 1] = 40;
              buf[idx + 2] = 30;
              buf[idx + 3] = 255;
            } else {
              buf[idx] = 20;
              buf[idx + 1] = 18;
              buf[idx + 2] = 16;
              buf[idx + 3] = 255;
            }
          } else {
            const row0 = field[gy0] || [];
            const row1 = field[gy1] || [];
            const v00 = row0[gx0] ?? 0;
            const v01 = row0[gx1] ?? 0;
            const v10 = row1[gx0] ?? 0;
            const v11 = row1[gx1] ?? 0;
            const val = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11);
            const [r, g, b] = getVelocityColor(val);
            buf[idx] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = 255;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);

      ctx.strokeStyle = isBaseline ? "#E53E3E" : "#38A169";
      ctx.lineWidth = 1.8;
      for (let gy = 0; gy < gridNy; gy++) {
        for (let gx = 0; gx < gridNx; gx++) {
          if (data.mask[gy] && data.mask[gy][gx]) {
            const isEdge =
              gy === 0 ||
              gy === gridNy - 1 ||
              gx === 0 ||
              gx === gridNx - 1 ||
              !data.mask[gy - 1][gx] ||
              !data.mask[gy + 1][gx] ||
              !data.mask[gy][gx - 1] ||
              !data.mask[gy][gx + 1];
            if (isEdge) {
              ctx.strokeRect((gx / gridNx) * w, (gy / gridNy) * h, w / gridNx, h / gridNy);
            }
          }
        }
      }
    };

    renderPanel(dualCanvasLeftRef.current, data.baseline, "Pre-TTA Baseline", true);
    renderPanel(dualCanvasRightRef.current, data.adapted, "Post-TTA Adapted", false);
  }, [diagramMode, data, getVelocityColor]);

  // Canvas Mouse Interactions: Drag/Place in Sandbox mode, Probe in normal mode
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSandboxMode || !canvasRef.current) return;
    setIsDraggingObstacle(true);
    updateObstacleFromMouse(e);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !data) return;

    if (isSandboxMode && isDraggingObstacle) {
      updateObstacleFromMouse(e);
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const xNorm = Math.max(0, Math.min(1, x / rect.width));
    const yNorm = Math.max(0, Math.min(1, y / rect.height));

    const [gridNy, gridNx] = data.grid;
    const gx = Math.min(gridNx - 1, Math.floor(xNorm * gridNx));
    const gy = Math.min(gridNy - 1, Math.floor(yNorm * gridNy));

    const isObstacle = Boolean(data.mask[gy] && data.mask[gy][gx]);
    const speed = data.adapted[gy] ? data.adapted[gy][gx] : 0;

    // Velocity from the nearest predicted arrow (real ux, uy, normalized by the max true speed)
    let vx = 0;
    let vy = 0;
    if (!isObstacle && data.vectors && data.vectors.length > 0) {
      let best = data.vectors[0];
      let bestDist = Infinity;
      for (const vec of data.vectors) {
        const dist = (vec.x_norm - xNorm) ** 2 + (vec.y_norm - yNorm) ** 2;
        if (dist < bestDist) {
          bestDist = dist;
          best = vec;
        }
      }
      vx = best.vx;
      vy = best.vy;
    }

    setHoverProbe({
      xNorm: Math.round(xNorm * 100) / 100,
      yNorm: Math.round(yNorm * 100) / 100,
      canvasX: x,
      canvasY: y,
      speed: Math.round(speed * 1000) / 1000,
      vx: Math.round(vx * 1000) / 1000,
      vy: Math.round(vy * 1000) / 1000,
      isObstacle,
      divBaseline: data.physics_metrics?.baseline_divergence ?? 0,
      divAdapted: data.physics_metrics?.adapted_divergence ?? 0,
    });
  };

  const handleCanvasMouseUp = () => {
    if (isSandboxMode && isDraggingObstacle) {
      setIsDraggingObstacle(false);
      handleSimulateCustom(customShape, customX, customY);
    } else {
      setIsDraggingObstacle(false);
    }
  };

  const handleCanvasDrop = (e: React.DragEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const droppedShape = e.dataTransfer.getData("text/plain");
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const xNorm = Math.max(0.15, Math.min(0.75, (e.clientX - rect.left) / rect.width));
      const yNorm = Math.max(0.20, Math.min(0.80, (e.clientY - rect.top) / rect.height));
      const finalX = Math.round(xNorm * 100) / 100;
      const finalY = Math.round(yNorm * 100) / 100;
      const shapeToUse = droppedShape || customShape;
      setCustomShape(shapeToUse);
      setCustomX(finalX);
      setCustomY(finalY);
      setIsDraggingObstacle(false);
      handleSimulateCustom(shapeToUse, finalX, finalY);
    }
    setDraggedShape(null);
  };

  const handleCanvasDragOver = (e: React.DragEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (dragThrottleRef.current) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const xNorm = Math.max(0.15, Math.min(0.75, (e.clientX - rect.left) / rect.width));
      const yNorm = Math.max(0.20, Math.min(0.80, (e.clientY - rect.top) / rect.height));
      dragThrottleRef.current = requestAnimationFrame(() => {
        setCustomX(Math.round(xNorm * 100) / 100);
        setCustomY(Math.round(yNorm * 100) / 100);
        dragThrottleRef.current = null;
      });
    }
  };

  const updateObstacleFromMouse = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const xNorm = Math.max(0.15, Math.min(0.75, (e.clientX - rect.left) / rect.width));
    const yNorm = Math.max(0.20, Math.min(0.80, (e.clientY - rect.top) / rect.height));
    if (dragThrottleRef.current) return;
    dragThrottleRef.current = requestAnimationFrame(() => {
      setCustomX(Math.round(xNorm * 100) / 100);
      setCustomY(Math.round(yNorm * 100) / 100);
      dragThrottleRef.current = null;
    });
  };

  return (
    <div className="card" style={{ padding: 18, position: "relative" }}>
      {/* ── Top Header & Controls Bar ────────────────────────────────────────── */}
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
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
              2D Obstacle Flow Dynamics &amp; Residual Adaptation
            </h3>
            <span className="badge badge-terracotta" style={{ fontSize: 10 }}>
              D2Q9 Navier-Stokes
            </span>
            <span className="badge badge-sage" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 4 }}>
              <Cpu size={10} />
              FNO-2D Surrogate: 1.19M params
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "3px 0 0" }}>
            Benchmark standardized obstacle geometries or switch to the <strong>Interactive Obstacle Sandbox</strong> to place any shape anywhere.
          </p>
        </div>

        {/* Mode Selector: Benchmark vs Sandbox */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <button
            onClick={() => {
              setIsSandboxMode(false);
              fetchSlice(shape, timestep);
            }}
            style={{
              padding: "5px 12px",
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: !isSandboxMode ? 600 : 500,
              border: !isSandboxMode ? "1px solid var(--accent-terracotta)" : "1px solid var(--border)",
              background: !isSandboxMode ? "var(--accent-terracotta)" : "#FFFFFF",
              color: !isSandboxMode ? "#FFFFFF" : "var(--text-secondary)",
              cursor: "pointer",
            }}
          >
            Standard Benchmarks
          </button>

          <button
            onClick={() => {
              setIsSandboxMode(true);
              handleSimulateCustom();
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              padding: "5px 12px",
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: isSandboxMode ? 600 : 500,
              border: isSandboxMode ? "1px solid #B85226" : "1px solid var(--border)",
              background: isSandboxMode ? "rgba(184, 82, 38, 0.12)" : "#FFFFFF",
              color: isSandboxMode ? "var(--accent-terracotta)" : "var(--text-primary)",
              cursor: "pointer",
            }}
          >
            <Sparkles size={12} color="var(--accent-terracotta)" />
            Custom Obstacle Sandbox
          </button>
        </div>
      </div>

      {/* ── Standard Benchmark Shapes or Custom Sandbox Drawer ───────────────── */}
      {!isSandboxMode ? (
        <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
          {[
            { id: "cylinder", label: "Cylinder" },
            { id: "airfoil", label: "NACA Airfoil" },
            { id: "square", label: "Square Bluff" },
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
                fontSize: 11,
                padding: "4px 12px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: shape === s.id ? "#1C1917" : "#FFFFFF",
                color: shape === s.id ? "#FFFFFF" : "var(--text-secondary)",
                fontWeight: shape === s.id ? 600 : 400,
                cursor: "pointer",
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      ) : (
        /* ── INTERACTIVE OBSTACLE SANDBOX — BIG VISUAL SHAPE PICKER ──────────── */
        <div
          style={{
            padding: "18px 20px",
            background: "linear-gradient(135deg, #FAF7F2 0%, #F5F0E8 100%)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            marginBottom: 14,
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          {/* Row 1: Title + Simulate Button */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                <Move size={16} color="var(--accent-terracotta)" />
                Pick &amp; Place Your Obstacle Shape
              </div>
              <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "3px 0 0" }}>
                Select a shape below, then drag it on the wind tunnel canvas to position it. Click <strong>Simulate</strong> to run the flow solver, the trained FNO-2D and physics TTA.
              </p>
            </div>
            <button
              onClick={() => handleSimulateCustom()}
              disabled={isSimulatingCustom}
              className="btn btn-primary"
              style={{
                fontSize: 13,
                padding: "10px 22px",
                display: "flex",
                alignItems: "center",
                gap: 8,
                boxShadow: "0 3px 12px rgba(184,82,38,0.3)",
                borderRadius: 8,
                fontWeight: 700,
                letterSpacing: "0.01em",
              }}
            >
              <Zap size={16} />
              {isSimulatingCustom ? "Simulating..." : "▶ Simulate Flow"}
            </button>
          </div>

          {/* Row 2: 6 PROPER GEOMETRIC SHAPE CARDS (DRAG & DROP) */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10 }}>
            {[
              {
                id: "circle",
                label: "Circle",
                desc: "Cylinder Bluff",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <circle cx="40" cy="30" r="16" fill="none" stroke="currentColor" strokeWidth="2.5" />
                    <circle cx="40" cy="30" r="4.5" fill="currentColor" opacity="0.25" />
                  </svg>
                ),
              },
              {
                id: "square",
                label: "Square",
                desc: "90° Bluff Body",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <rect x="26" y="16" width="28" height="28" rx="1" fill="none" stroke="currentColor" strokeWidth="2.5" />
                    <rect x="33" y="23" width="14" height="14" fill="currentColor" opacity="0.2" />
                  </svg>
                ),
              },
              {
                id: "rectangle",
                label: "Rectangle",
                desc: "Flat Plate Block",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <rect x="18" y="21" width="44" height="18" rx="1" fill="none" stroke="currentColor" strokeWidth="2.5" />
                    <rect x="26" y="25" width="28" height="10" fill="currentColor" opacity="0.2" />
                  </svg>
                ),
              },
              {
                id: "triangle",
                label: "Triangle",
                desc: "Equilateral Wedge",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <polygon points="22,30 58,13 58,47" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
                    <polygon points="32,30 52,19 52,41" fill="currentColor" opacity="0.2" strokeLinejoin="round" />
                  </svg>
                ),
              },
              {
                id: "diamond",
                label: "Diamond",
                desc: "45° Rhombus",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <polygon points="40,11 65,30 40,49 15,30" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
                    <polygon points="40,20 54,30 40,40 26,30" fill="currentColor" opacity="0.2" strokeLinejoin="round" />
                  </svg>
                ),
              },
              {
                id: "hexagon",
                label: "Hexagon",
                desc: "6-Sided Body",
                svg: (
                  <svg viewBox="0 0 80 60" width="76" height="52">
                    <polygon points="40,11 60,20.5 60,39.5 40,49 20,39.5 20,20.5" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
                    <polygon points="40,20 51,25.5 51,34.5 40,40 29,34.5 29,25.5" fill="currentColor" opacity="0.2" strokeLinejoin="round" />
                  </svg>
                ),
              },
            ].map((s) => {
              const isSelected = customShape === s.id;
              return (
                <div
                  key={s.id}
                  draggable={true}
                  onDragStart={(e) => {
                    e.dataTransfer.setData("text/plain", s.id);
                    setDraggedShape(s.id);
                  }}
                  onDragEnd={() => setDraggedShape(null)}
                  onClick={() => {
                    setCustomShape(s.id);
                    handleSimulateCustom(s.id, customX, customY);
                  }}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 6,
                    padding: "12px 6px 10px",
                    borderRadius: 10,
                    border: isSelected ? "2.5px solid var(--accent-terracotta)" : "1.5px solid var(--border)",
                    background: isSelected
                      ? "linear-gradient(145deg, rgba(184,82,38,0.08) 0%, rgba(184,82,38,0.03) 100%)"
                      : "#FFFFFF",
                    cursor: "grab",
                    transition: "all 0.18s ease",
                    boxShadow: isSelected
                      ? "0 4px 16px rgba(184,82,38,0.18), inset 0 1px 0 rgba(255,255,255,0.8)"
                      : "0 1px 4px rgba(0,0,0,0.04)",
                    transform: isSelected ? "scale(1.04)" : "scale(1)",
                    position: "relative",
                    overflow: "hidden",
                    minHeight: 105,
                    userSelect: "none",
                  }}
                  title={`Drag ${s.label} onto the wind tunnel canvas or click to select`}
                >
                  {/* Selection indicator dot */}
                  {isSelected && (
                    <div style={{
                      position: "absolute",
                      top: 6,
                      right: 6,
                      width: 9,
                      height: 9,
                      borderRadius: "50%",
                      background: "var(--accent-terracotta)",
                      boxShadow: "0 0 6px rgba(184,82,38,0.5)",
                    }} />
                  )}

                  {/* SVG Shape Preview */}
                  <div style={{
                    color: isSelected ? "var(--accent-terracotta)" : "#786E64",
                    transition: "color 0.15s ease",
                  }}>
                    {s.svg}
                  </div>

                  {/* Label */}
                  <div style={{
                    fontSize: 12,
                    fontWeight: isSelected ? 700 : 600,
                    color: isSelected ? "var(--accent-terracotta)" : "var(--text-primary)",
                    letterSpacing: "-0.01em",
                  }}>
                    {s.label}
                  </div>

                  {/* Description */}
                  <div style={{
                    fontSize: 9.5,
                    color: "var(--text-secondary)",
                    textAlign: "center",
                    lineHeight: 1.2,
                  }}>
                    {s.desc}
                  </div>

                  {/* Drag Prompt */}
                  <div style={{
                    fontSize: 8.5,
                    color: isSelected ? "var(--accent-terracotta)" : "#A0988E",
                    fontWeight: 600,
                    marginTop: 2,
                    textTransform: "uppercase",
                    letterSpacing: "0.03em",
                  }}>
                    Drag to place
                  </div>
                </div>
              );
            })}
          </div>

          {/* Row 3: Position & Parameter Controls */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 16,
            padding: "14px 16px",
            background: "#FFFFFF",
            borderRadius: 8,
            border: "1px solid var(--border)",
          }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11.5, color: "var(--text-secondary)" }}>
                <span style={{ fontWeight: 600 }}>↔ Streamwise X</span>
                <strong style={{ color: "var(--accent-terracotta)" }}>{Math.round(customX * 100)}%</strong>
              </div>
              <input
                type="range"
                min={0.15}
                max={0.75}
                step={0.01}
                value={customX}
                onChange={(e) => setCustomX(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-terracotta)", height: 6, cursor: "pointer" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11.5, color: "var(--text-secondary)" }}>
                <span style={{ fontWeight: 600 }}>↕ Cross-stream Y</span>
                <strong style={{ color: "var(--accent-terracotta)" }}>{Math.round(customY * 100)}%</strong>
              </div>
              <input
                type="range"
                min={0.20}
                max={0.80}
                step={0.01}
                value={customY}
                onChange={(e) => setCustomY(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-terracotta)", height: 6, cursor: "pointer" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11.5, color: "var(--text-secondary)" }}>
                <span style={{ fontWeight: 600 }}>⬡ Scale</span>
                <strong style={{ color: "var(--accent-terracotta)" }}>{customSize.toFixed(1)}×</strong>
              </div>
              <input
                type="range"
                min={0.6}
                max={1.8}
                step={0.05}
                value={customSize}
                onChange={(e) => setCustomSize(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-terracotta)", height: 6, cursor: "pointer" }}
              />
            </div>

            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 11.5, color: "var(--text-secondary)" }}>
                <span style={{ fontWeight: 600 }}>⟳ Angle</span>
                <strong style={{ color: "var(--accent-terracotta)" }}>{customAngle > 0 ? `+${customAngle}` : customAngle}°</strong>
              </div>
              <input
                type="range"
                min={-30}
                max={30}
                step={1}
                value={customAngle}
                onChange={(e) => setCustomAngle(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-terracotta)", height: 6, cursor: "pointer" }}
              />
            </div>
          </div>

          {/* Tip Banner */}
          <div style={{
            fontSize: 11.5,
            color: "var(--text-secondary)",
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "8px 12px",
            background: "rgba(184, 82, 38, 0.05)",
            borderRadius: 6,
            border: "1px dashed rgba(184, 82, 38, 0.25)",
          }}>
            <Move size={14} color="var(--accent-terracotta)" style={{ flexShrink: 0 }} />
            <span>
              <strong>Drag to place:</strong> Click and drag directly on the wind tunnel canvas below to reposition the obstacle, then click <strong style={{ color: "var(--accent-terracotta)" }}>▶ Simulate Flow</strong>.
            </span>
          </div>
        </div>
      )}

      {/* ── Diagram Type Switcher Tabs ───────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 10,
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => setDiagramMode("quiver")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 12px",
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: diagramMode === "quiver" ? 600 : 500,
              border: diagramMode === "quiver" ? "1px solid var(--accent-terracotta)" : "1px solid var(--border)",
              background: diagramMode === "quiver" ? "rgba(184, 82, 38, 0.08)" : "var(--bg-card)",
              color: diagramMode === "quiver" ? "var(--accent-terracotta)" : "var(--text-primary)",
              cursor: "pointer",
            }}
          >
            <Compass size={13} />
            Diagram 1: Wind Tunnel Quiver &amp; Vectors
          </button>

          <button
            onClick={() => setDiagramMode("dual")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 12px",
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: diagramMode === "dual" ? 600 : 500,
              border: diagramMode === "dual" ? "1px solid var(--accent-terracotta)" : "1px solid var(--border)",
              background: diagramMode === "dual" ? "rgba(184, 82, 38, 0.08)" : "var(--bg-card)",
              color: diagramMode === "dual" ? "var(--accent-terracotta)" : "var(--text-primary)",
              cursor: "pointer",
            }}
          >
            <Columns size={13} />
            Diagram 2: Synchronized Dual Comparison (Pre vs Post TTA)
          </button>

          <button
            onClick={() => setDiagramMode("centerline")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 12px",
              borderRadius: 6,
              fontSize: 11.5,
              fontWeight: diagramMode === "centerline" ? 600 : 500,
              border: diagramMode === "centerline" ? "1px solid var(--accent-terracotta)" : "1px solid var(--border)",
              background: diagramMode === "centerline" ? "rgba(184, 82, 38, 0.08)" : "var(--bg-card)",
              color: diagramMode === "centerline" ? "var(--accent-terracotta)" : "var(--text-primary)",
              cursor: "pointer",
            }}
          >
            <LineChart size={13} />
            Diagram 3: Centerline Wake Velocity Profile
          </button>
        </div>

        {/* Sub-toggles for mode 1 */}
        {diagramMode === "quiver" && (
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <button
              onClick={() => setShowVectors(!showVectors)}
              style={{
                fontSize: 10.5,
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: showVectors ? "var(--accent-terracotta)" : "#FFF",
                color: showVectors ? "#FFF" : "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              Vectors: {showVectors ? "ON" : "OFF"}
            </button>
            <button
              onClick={() => setShowStreamlines(!showStreamlines)}
              style={{
                fontSize: 10.5,
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid var(--border)",
                background: showStreamlines ? "#2E6343" : "#FFF",
                color: showStreamlines ? "#FFF" : "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              Streamlines: {showStreamlines ? "ON" : "OFF"}
            </button>

            <div style={{ display: "flex", gap: 2, background: "var(--bg-tag)", padding: 2, borderRadius: 4 }}>
              {[
                { id: "adapted", label: "Adapted (TTA)" },
                { id: "baseline", label: "Baseline (trained model)" },
                { id: "ground_truth", label: "Ground Truth" },
              ].map((v) => (
                <button
                  key={v.id}
                  onClick={() => setViewVariant(v.id as any)}
                  style={{
                    fontSize: 10,
                    padding: "2px 7px",
                    borderRadius: 3,
                    border: "none",
                    background: viewVariant === v.id ? "#1C1917" : "transparent",
                    color: viewVariant === v.id ? "#FFF" : "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  {v.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── User Confirmation Modal (Missing Dataset) ────────────────────────── */}
      {promptShape && (
        <div
          style={{
            padding: 16,
            background: "#FFF8F2",
            border: "1px solid var(--accent-terracotta)",
            borderRadius: 6,
            marginBottom: 16,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--accent-terracotta)" }}>
              Dataset Missing for &apos;{promptShape}&apos;
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
              The 2D fluid simulation dataset for this shape has not been generated yet. Would you like to generate it now?
            </div>
          </div>
          <button
            onClick={handleUserConfirmedGenerate}
            disabled={generating}
            className="btn btn-primary"
            style={{ fontSize: 11.5, padding: "6px 14px", whiteSpace: "nowrap" }}
          >
            {generating ? "Simulating D2Q9 LBM..." : "Generate Dataset"}
          </button>
        </div>
      )}

      {/* ── DIAGRAM TYPE 1: WIND TUNNEL QUIVER & VECTORS ─────────────────────── */}
      {diagramMode === "quiver" && (
        <div style={{ position: "relative" }}>
          <div
            style={{
              position: "relative",
              width: "100%",
              borderRadius: 6,
              overflow: "hidden",
              border: "1px solid var(--border)",
              background: "#12100E",
              boxShadow: "inset 0 2px 6px rgba(0,0,0,0.3)",
            }}
          >
            <canvas
              ref={canvasRef}
              width={800}
              height={380}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onDragOver={isSandboxMode ? handleCanvasDragOver : undefined}
              onDrop={isSandboxMode ? handleCanvasDrop : undefined}
              onMouseLeave={() => {
                handleCanvasMouseUp();
                setHoverProbe(null);
              }}
              style={{
                width: "100%",
                height: "350px",
                display: "block",
                cursor: isSandboxMode ? (isDraggingObstacle ? "grabbing" : "grab") : "crosshair",
              }}
            />

            {/* Visual Drop Zone Glow when dragging from palette */}
            {draggedShape && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  border: "2px dashed #F2A900",
                  backgroundColor: "rgba(242, 169, 0, 0.08)",
                  pointerEvents: "none",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  zIndex: 10,
                }}
              >
                <div
                  style={{
                    padding: "8px 18px",
                    background: "rgba(22, 19, 17, 0.92)",
                    border: "1px solid #F2A900",
                    borderRadius: 20,
                    color: "#F2A900",
                    fontSize: 13,
                    fontWeight: 700,
                    letterSpacing: "0.02em",
                    boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
                  }}
                >
                  📍 Drop {draggedShape.toUpperCase()} Anywhere in Wind Tunnel
                </div>
              </div>
            )}

            {/* Hover Inspection HUD Card */}
            {hoverProbe && !isDraggingObstacle && (
              <div
                style={{
                  position: "absolute",
                  left: Math.min(hoverProbe.canvasX + 14, 560),
                  top: Math.min(hoverProbe.canvasY + 14, 230),
                  background: "rgba(18, 14, 12, 0.92)",
                  border: "1px solid rgba(255, 255, 255, 0.2)",
                  backdropFilter: "blur(6px)",
                  borderRadius: 6,
                  padding: "8px 12px",
                  color: "#FFFFFF",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  pointerEvents: "none",
                  zIndex: 20,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
                  minWidth: 200,
                }}
              >
                <div style={{ fontWeight: 600, color: "var(--accent-terracotta)", marginBottom: 4 }}>
                  {hoverProbe.isObstacle ? "⛔ Solid Obstacle Body" : "📍 Fluid Field Probe"}
                </div>
                <div>Position: ({hoverProbe.xNorm}L, {hoverProbe.yNorm}H)</div>
                <div>Speed (normalized): {hoverProbe.speed}</div>
                <div>Velocity (normalized): ({hoverProbe.vx}, {hoverProbe.vy})</div>
                <div style={{ marginTop: 4, paddingTop: 4, borderTop: "1px solid rgba(255,255,255,0.15)", fontSize: 10 }}>
                  <div style={{ color: "#E53E3E" }}>Baseline PDE residual (frame): {hoverProbe.divBaseline.toExponential(2)}</div>
                  <div style={{ color: "#48BB78" }}>After TTA (frame): {hoverProbe.divAdapted.toExponential(2)}</div>
                </div>
              </div>
            )}

            {/* In-Canvas Explanatory Legend Overlay */}
            <div
              style={{
                position: "absolute",
                bottom: 8,
                left: 12,
                display: "flex",
                alignItems: "center",
                gap: 14,
                background: "rgba(12, 10, 8, 0.75)",
                backdropFilter: "blur(4px)",
                padding: "4px 10px",
                borderRadius: 4,
                border: "1px solid rgba(255,255,255,0.1)",
                fontSize: 10,
                color: "rgba(255, 255, 255, 0.8)",
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 8, height: 8, background: "#161311", border: "1px solid #F2A900", display: "inline-block" }}></span>
                Solid Body (u=0)
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 12, height: 2, background: "rgba(160, 215, 255, 0.9)", display: "inline-block" }}></span>
                Low Speed
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 12, height: 2, background: "rgba(255, 205, 90, 0.9)", display: "inline-block" }}></span>
                High Acceleration
              </div>
              <div style={{ color: "rgba(255,255,255,0.5)" }}>
                {isSandboxMode ? "• Click/drag to reposition obstacle" : "• Hover anywhere to inspect physical probe"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DIAGRAM TYPE 2: SYNCHRONIZED DUAL COMPARISON ────────────────────── */}
      {diagramMode === "dual" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {/* Panel A: Baseline (trained model) */}
            <div
              style={{
                border: "1px solid #FEB2B2",
                borderRadius: 6,
                overflow: "hidden",
                background: "#FFF5F5",
              }}
            >
              <div
                style={{
                  padding: "8px 12px",
                  background: "#FED7D7",
                  borderBottom: "1px solid #FEB2B2",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: 12, color: "#C53030" }}>
                  <AlertTriangle size={14} />
                  Pre-TTA Baseline (trained FNO-2D)
                </div>
                <span style={{ fontSize: 10, fontFamily: "monospace", color: "#9B2C2C" }}>
                  residual = {data?.physics_metrics ? data.physics_metrics.baseline_divergence.toExponential(2) : "n/a"}
                </span>
              </div>
              <div style={{ position: "relative", background: "#12100E" }}>
                <canvas
                  ref={dualCanvasLeftRef}
                  width={400}
                  height={220}
                  style={{ width: "100%", height: "200px", display: "block" }}
                />
              </div>
              <div style={{ padding: "8px 12px", fontSize: 11, color: "#742A2A", lineHeight: 1.4 }}>
                <strong>Trained model:</strong> one-step prediction, relative error{" "}
                {data?.physics_metrics?.baseline_rel_l2 !== undefined ? `${(data.physics_metrics.baseline_rel_l2 * 100).toFixed(2)}%` : "n/a"}.
              </div>
            </div>

            {/* Panel B: Post-TTA Adapted */}
            <div
              style={{
                border: "1px solid #C6F6D5",
                borderRadius: 6,
                overflow: "hidden",
                background: "#F0FFF4",
              }}
            >
              <div
                style={{
                  padding: "8px 12px",
                  background: "#C6F6D5",
                  borderBottom: "1px solid #9AE6B4",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: 12, color: "#22543D" }}>
                  <CheckCircle2 size={14} />
                  Post-TTA Adapted (Physics-Residual Enforced)
                </div>
                <span style={{ fontSize: 10, fontFamily: "monospace", color: "#22543D" }}>
                  residual = {data?.physics_metrics ? data.physics_metrics.adapted_divergence.toExponential(2) : "n/a"}
                </span>
              </div>
              <div style={{ position: "relative", background: "#12100E" }}>
                <canvas
                  ref={dualCanvasRightRef}
                  width={400}
                  height={220}
                  style={{ width: "100%", height: "200px", display: "block" }}
                />
              </div>
              <div style={{ padding: "8px 12px", fontSize: 11, color: "#22543D", lineHeight: 1.4 }}>
                <strong>After TTA:</strong> relative error{" "}
                {data?.physics_metrics?.adapted_rel_l2 !== undefined ? `${(data.physics_metrics.adapted_rel_l2 * 100).toFixed(2)}%` : "n/a"}
                {data?.physics_metrics?.truth_residual !== undefined && (
                  <> (the true flow&apos;s residual is {data.physics_metrics.truth_residual.toExponential(2)})</>
                )}.
              </div>
            </div>
          </div>

          {/* Quantifiable Summary Strip */}
          <div
            style={{
              marginTop: 12,
              padding: "10px 14px",
              background: "var(--bg-tag)",
              borderRadius: 6,
              display: "flex",
              justifyContent: "space-around",
              alignItems: "center",
              fontSize: 11.5,
              border: "1px solid var(--border)",
            }}
          >
            <div>
              <span style={{ color: "var(--text-secondary)" }}>PDE Residual Change: </span>
              <strong style={{ color: "var(--accent-terracotta)" }}>
                {data?.physics_metrics ? `${(-data.physics_metrics.divergence_drop_pct).toFixed(1)}%` : "n/a"}
              </strong>
            </div>
            <div style={{ borderLeft: "1px solid var(--border)", height: 16 }}></div>
            <div>
              <span style={{ color: "var(--text-secondary)" }}>Speed Inside Body: </span>
              <strong style={{ color: "var(--accent-sage)" }}>
                {data?.physics_metrics
                  ? `${data.physics_metrics.baseline_boundary_leak.toExponential(1)} → ${data.physics_metrics.adapted_boundary_leak.toExponential(1)}`
                  : "n/a"}
              </strong>
            </div>
            <div style={{ borderLeft: "1px solid var(--border)", height: 16 }}></div>
            <div>
              <span style={{ color: "var(--text-secondary)" }}>Prediction Error (MSE) Change: </span>
              <strong style={{ color: "var(--accent-terracotta)" }}>
                {data?.physics_metrics ? `${(-data.physics_metrics.wake_mse_improvement_pct).toFixed(1)}%` : "n/a"}
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* ── DIAGRAM TYPE 3: CENTERLINE WAKE VELOCITY PROFILE ─────────────────── */}
      {diagramMode === "centerline" && data?.centerline && (
        <div>
          <div
            style={{
              padding: 16,
              background: "#FAF7F2",
              borderRadius: 6,
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div>
                <h4 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                  Streamwise Centerline Velocity Deficit: u(x, y = y_center) / U_∞
                </h4>
                <p style={{ margin: "2px 0 0", fontSize: 11.5, color: "var(--text-secondary)" }}>
                  Standard wind-tunnel benchmark showing fluid deceleration, obstacle boundary wall, and downstream wake recovery.
                </p>
              </div>

              <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 14, height: 2.5, background: "#1C1917", display: "inline-block" }}></span>
                  <span>Ground Truth</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 14, height: 2, background: "#E53E3E", borderTop: "2px dashed #E53E3E", display: "inline-block" }}></span>
                  <span style={{ color: "#E53E3E" }}>Pre-TTA Baseline</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 14, height: 2.5, background: "#2E6343", display: "inline-block" }}></span>
                  <span style={{ color: "#2E6343", fontWeight: 600 }}>Post-TTA Adapted</span>
                </div>
              </div>
            </div>

            {/* Interactive SVG Line Graph */}
            <div style={{ width: "100%", height: "240px", position: "relative" }}>
              <svg
                viewBox="0 0 700 220"
                style={{ width: "100%", height: "100%", overflow: "visible" }}
                onMouseLeave={() => setChartHoverIdx(null)}
              >
                <line x1="50" y1="20" x2="50" y2="180" stroke="#D5CEBF" strokeWidth="1" />
                <line x1="50" y1="180" x2="680" y2="180" stroke="#D5CEBF" strokeWidth="1" />

                {[0.0, 0.25, 0.5, 0.75, 1.0].map((v, i) => {
                  const y = 180 - v * 150;
                  return (
                    <g key={i}>
                      <line x1="45" y1={y} x2="680" y2={y} stroke="#EBE5D8" strokeWidth="0.8" strokeDasharray="3 3" />
                      <text x="40" y={y + 3.5} textAnchor="end" fontSize="9.5" fill="#786E64" fontFamily="monospace">
                        {v.toFixed(2)}
                      </text>
                    </g>
                  );
                })}

                {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v, i) => {
                  const x = 50 + v * 620;
                  return (
                    <g key={i}>
                      <line x1={x} y1="180" x2={x} y2="185" stroke="#786E64" strokeWidth="1" />
                      <text x={x} y="198" textAnchor="middle" fontSize="9.5" fill="#786E64" fontFamily="monospace">
                        x = {v.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                {/* Shaded Obstacle Body Zone */}
                {(() => {
                  const obsCols = data.centerline.filter((pt) => pt.is_obstacle);
                  if (obsCols.length === 0) return null;
                  const minX = Math.min(...obsCols.map((p) => p.x));
                  const maxX = Math.max(...obsCols.map((p) => p.x));
                  const xStart = 50 + minX * 620;
                  const xEnd = 50 + maxX * 620;
                  const w = Math.max(16, xEnd - xStart);
                  return (
                    <g>
                      <rect x={xStart} y="30" width={w} height="150" fill="rgba(28, 25, 23, 0.12)" stroke="#1C1917" strokeDasharray="2 2" />
                      <text x={xStart + w / 2} y="22" textAnchor="middle" fontSize="9" fill="#1C1917" fontWeight="600">
                        SOLID OBSTACLE (u = 0)
                      </text>
                    </g>
                  );
                })()}

                <path
                  d={data.centerline
                    .map((pt, i) => {
                      const x = 50 + pt.x * 620;
                      const y = 180 - pt.ground_truth * 150;
                      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="#1C1917"
                  strokeWidth="2.2"
                />

                <path
                  d={data.centerline
                    .map((pt, i) => {
                      const x = 50 + pt.x * 620;
                      const y = 180 - pt.baseline * 150;
                      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="#E53E3E"
                  strokeWidth="1.8"
                  strokeDasharray="4 3"
                />

                <path
                  d={data.centerline
                    .map((pt, i) => {
                      const x = 50 + pt.x * 620;
                      const y = 180 - pt.adapted * 150;
                      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="#2E6343"
                  strokeWidth="2.4"
                />

                {data.centerline.map((pt, i) => {
                  const x = 50 + pt.x * 620;
                  return (
                    <rect
                      key={i}
                      x={x - 4}
                      y="20"
                      width="8"
                      height="160"
                      fill="transparent"
                      style={{ cursor: "pointer" }}
                      onMouseEnter={() => setChartHoverIdx(i)}
                    />
                  );
                })}

                {chartHoverIdx !== null && data.centerline[chartHoverIdx] && (
                  <g>
                    {(() => {
                      const pt = data.centerline[chartHoverIdx];
                      const x = 50 + pt.x * 620;
                      const yGt = 180 - pt.ground_truth * 150;
                      const yBase = 180 - pt.baseline * 150;
                      const yTta = 180 - pt.adapted * 150;
                      return (
                        <g>
                          <line x1={x} y1="20" x2={x} y2="180" stroke="#B85226" strokeWidth="1.2" strokeDasharray="2 2" />
                          <circle cx={x} cy={yGt} r="3.5" fill="#1C1917" />
                          <circle cx={x} cy={yBase} r="3.5" fill="#E53E3E" />
                          <circle cx={x} cy={yTta} r="4" fill="#2E6343" stroke="#FFF" strokeWidth="1.5" />
                        </g>
                      );
                    })()}
                  </g>
                )}
              </svg>

              {chartHoverIdx !== null && data.centerline[chartHoverIdx] && (
                <div
                  style={{
                    position: "absolute",
                    top: 10,
                    right: 20,
                    background: "rgba(255, 255, 255, 0.95)",
                    border: "1px solid var(--border)",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                    borderRadius: 5,
                    padding: "6px 10px",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                  }}
                >
                  <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    x/L = {data.centerline[chartHoverIdx].x.toFixed(3)}
                    {data.centerline[chartHoverIdx].is_obstacle && " (Inside Obstacle)"}
                  </div>
                  <div>Ground Truth: {data.centerline[chartHoverIdx].ground_truth.toFixed(4)}</div>
                  <div style={{ color: "#C53030" }}>Baseline: {data.centerline[chartHoverIdx].baseline.toFixed(4)}</div>
                  <div style={{ color: "#2E6343", fontWeight: 600 }}>Post-TTA Adapted: {data.centerline[chartHoverIdx].adapted.toFixed(4)}</div>
                </div>
              )}
            </div>

            <div
              style={{
                marginTop: 10,
                padding: "8px 12px",
                background: "#FFFFFF",
                borderRadius: 5,
                border: "1px solid var(--border)",
                fontSize: 11.5,
                color: "var(--text-secondary)",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <Info size={14} style={{ color: "var(--accent-terracotta)", flexShrink: 0 }} />
              <div>
                <strong>What this shows:</strong> the true next frame from the Lattice Boltzmann solver, the trained {data.physics_metrics?.model ?? "surrogate"}&apos;s prediction, and its prediction after physics-residual TTA, all on one colour scale.
                {data.physics_metrics && (
                  <> On this frame TTA changed the PDE residual by {(-data.physics_metrics.divergence_drop_pct).toFixed(1)}% and the prediction error (MSE) by {(-data.physics_metrics.wake_mse_improvement_pct).toFixed(1)}%. On these obstacle flows the trained model is already very accurate, so lowering the residual often does not lower the error.</>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Playback & Scrubber Controls ─────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          paddingTop: 14,
          marginTop: 10,
          borderTop: "1px solid var(--border)",
        }}
      >
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="btn btn-secondary"
          style={{ padding: "5px 12px", height: 28, fontSize: 11, display: "flex", alignItems: "center", gap: 5 }}
        >
          {isPlaying ? <Pause size={11} /> : <Play size={11} />}
          {isPlaying ? "Pause Flow" : "Simulate Flow"}
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
          max={(data?.total_timesteps || 12) - 1}
          value={timestep % (data?.total_timesteps || 12)}
          onChange={(e) => {
            setIsPlaying(false);
            setTimestep(Number(e.target.value));
          }}
          style={{ flex: 1, accentColor: "var(--accent-terracotta)", cursor: "pointer" }}
        />

        <button
          onClick={() => setTimestep((prev) => Math.min((data?.total_timesteps || 12) - 1, prev + 1))}
          style={{ border: "1px solid var(--border)", background: "#FFF", borderRadius: 4, padding: "4px 8px", cursor: "pointer" }}
        >
          <ChevronRight size={12} />
        </button>

        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: "var(--text-primary)",
            fontWeight: 600,
            minWidth: 100,
            textAlign: "right",
          }}
        >
          Step {(timestep % (data?.total_timesteps || 12)) + 1}/{data?.total_timesteps || 12}
        </span>
      </div>

      {/* ── COMPREHENSIVE COLOR SCALE & AERODYNAMIC FLOW PHYSICS LEGEND ──── */}
      <div
        style={{
          marginTop: 16,
          padding: "16px 18px",
          background: "#FAF7F2",
          border: "1px solid var(--border)",
          borderRadius: 8,
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
              Fluid Velocity Magnitude Colormap &amp; Physics Guide
            </span>
            <span className="badge badge-terracotta" style={{ fontSize: 10 }}>
              Normalized Speed |u| / U_∞
            </span>
          </div>
          <button
            onClick={() => setShowLegendDetails(!showLegendDetails)}
            style={{
              background: "none",
              border: "none",
              fontSize: 11,
              color: "var(--accent-terracotta)",
              cursor: "pointer",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            {showLegendDetails ? "Collapse Guide ▲" : "Expand Guide ▼"}
          </button>
        </div>

        {/* Continuous Colorbar Gradient */}
        <div style={{ marginBottom: 12 }}>
          <div
            style={{
              height: 14,
              borderRadius: 4,
              background: "linear-gradient(to right, rgb(18,30,75) 0%, rgb(28,85,125) 25%, rgb(50,160,145) 50%, rgb(205,195,70) 75%, rgb(250,245,200) 100%)",
              border: "1px solid rgba(0,0,0,0.15)",
              boxShadow: "inset 0 1px 3px rgba(0,0,0,0.2)",
              position: "relative",
            }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 10.5,
              fontFamily: "'JetBrains Mono', monospace",
              color: "var(--text-secondary)",
              marginTop: 4,
              fontWeight: 600,
            }}
          >
            <span>0.00 (Stagnation)</span>
            <span>0.25 (Wake Recirculation)</span>
            <span>0.50 (Shear Layer)</span>
            <span>0.75 (Nominal Inflow U_∞)</span>
            <span>1.00+ (Shoulder Acceleration)</span>
          </div>
        </div>

        {/* Detailed Physical Meaning Cards */}
        {showLegendDetails && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
              gap: 10,
              fontSize: 11.5,
              marginTop: 10,
            }}
          >
            {/* Card 1: Solid Obstacle */}
            <div style={{ padding: "10px 12px", background: "#FFFFFF", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontWeight: 700, color: "var(--text-primary)" }}>
                <span style={{ width: 12, height: 12, background: "#161311", border: "1.5px solid #F2A900", borderRadius: 2, display: "inline-block" }}></span>
                Black Body + Gold Border
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                <strong>Solid Obstacle (No-Slip Wall):</strong> Physical body placed in the tunnel. Viscous fluid adheres to the wall enforcing zero velocity (u = 0). TTA penalises any predicted velocity inside the body (no-slip term of the residual).
              </div>
            </div>

            {/* Card 2: Deep Navy */}
            <div style={{ padding: "10px 12px", background: "#FFFFFF", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontWeight: 700, color: "#121E4B" }}>
                <span style={{ width: 12, height: 12, background: "rgb(18,30,75)", borderRadius: 2, display: "inline-block" }}></span>
                Deep Navy Blue (|u| &lt; 0.25)
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                <strong>Stagnation &amp; Wake Core:</strong> Upstream front stagnation point where incoming flow decelerates to rest, and the low-velocity recirculating vortex core immediately behind the body.
              </div>
            </div>

            {/* Card 3: Teal */}
            <div style={{ padding: "10px 12px", background: "#FFFFFF", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontWeight: 700, color: "#1C557D" }}>
                <span style={{ width: 12, height: 12, background: "rgb(28,85,125)", borderRadius: 2, display: "inline-block" }}></span>
                Teal / Cyan (0.25 ≤ |u| &lt; 0.65)
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                <strong>Viscous Wake Deficit:</strong> Region downstream of the obstacle where fluid momentum has been dissipated by boundary layer separation and unsteady Karman vortex shedding.
              </div>
            </div>

            {/* Card 4: Golden Amber */}
            <div style={{ padding: "10px 12px", background: "#FFFFFF", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontWeight: 700, color: "#B85226" }}>
                <span style={{ width: 12, height: 12, background: "rgb(205,195,70)", borderRadius: 2, display: "inline-block" }}></span>
                Golden Amber (0.65 ≤ |u| &lt; 0.95)
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                <strong>Undisturbed Freestream Flow:</strong> The nominal incoming wind tunnel flow (U_∞) approaching from the left boundary, unaffected by the obstacle&apos;s presence.
              </div>
            </div>

            {/* Card 5: Warm Gold / White */}
            <div style={{ padding: "10px 12px", background: "#FFFFFF", borderRadius: 6, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontWeight: 700, color: "#8C7B10" }}>
                <span style={{ width: 12, height: 12, background: "rgb(250,245,200)", border: "1px solid #D8C97A", borderRadius: 2, display: "inline-block" }}></span>
                Warm Gold / White (|u| ≥ 1.0)
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                <strong>Shoulder Acceleration:</strong> Fluid diverted above and below the obstacle accelerates due to the Bernoulli effect and mass conservation (div u = 0), reaching peak velocity.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
