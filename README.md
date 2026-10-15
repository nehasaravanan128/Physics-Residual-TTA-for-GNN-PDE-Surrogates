# Physics-Residual TTA for GNN PDE Surrogates

[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.8-3C2179)](https://pyg.org/)
[![PDEBench](https://img.shields.io/badge/PDEBench-Burgers-blue)](https://github.com/pdebench/PDEBench)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org/)

An implementation and empirical evaluation of **Physics-Residual Test-Time Adaptation (TTA)** for Graph Neural Network PDE surrogates (MeshGraphNets), providing unsupervised test-time adaptation under geometric and parameter distribution shifts.

---

## 🎯 Problem Statement & Objective

Graph Neural Network (GNN) PDE surrogates trained via supervised learning often suffer sharp accuracy degradation when deployed under distribution shifts (e.g., unseen Reynolds numbers, altered boundary conditions, or out-of-distribution initial states). Retraining GNNs for every new physical regime is computationally prohibitive.

This project investigates and validates that an **unsupervised physics-residual loss** computed directly from the model's own predictions can guide **Test-Time Adaptation (TTA)** without any ground-truth labels, yielding up to **+93% error reduction** over non-adapted baselines.

---

## 📁 Repository Structure

```
aiml project/
├── files (5)/
│   ├── model.py            # MeshGraphNet GNN architecture (Pfaff et al.)
│   ├── data.py             # Dataset loaders (Burgers HDF5 & CylinderFlow)
│   ├── train.py            # Supervised baseline training loop
│   ├── evaluate.py         # Standard checkpoint evaluation (Pre-TTA)
│   └── 1D_Burgers_Sols_Nu0.001.hdf5  # PDEBench Burgers dataset (8.2 GB)
├── scripts/
│   ├── physics_loss.py     # Differentiable PINN-style Burgers PDE residual module
│   ├── physics_loss_2d.py  # Differentiable 2D Navier-Stokes, continuity & diffusion residuals
│   ├── tta.py              # Test-Time Adaptation engine (freezing + anchor optimization)
│   ├── models_2d.py        # Multi-model 2D surrogate suite (FNO-2D, UNet-2D, GAT, MeshGraphNet)
│   ├── generate_2d_shapes.py # D2Q9 Lattice Boltzmann simulator past 5 geometric obstacle shapes
│   ├── generate_synthetic_pde.py # Pseudo-spectral 1D/2D PDE dataset generator
│   ├── download_pde_datasets.py # 10-dataset automated catalog & synthesizer
│   ├── data_2d.py          # 2D Delaunay mesh graph dataset loader
│   ├── get_2d_slice.py     # 2D velocity field extractor for web canvas rendering
│   ├── evaluate_2d_benchmarks.py # Multi-shape & multi-model 2D benchmark runner
│   ├── evaluate_geometric_shift.py # Geometric mesh distortion benchmark
│   ├── evaluate_ablations.py # Systematic ablation suite (K, λ, η, scope)
│   ├── evaluate_tta.py     # Batch comparative benchmarking suite (Shift vs In-Dist)
│   ├── run_all_benchmarks.py # 1-click master reproduction runner
│   ├── infer.py            # Multi-step inference bridge for API
│   ├── evaluate_api.py     # Evaluation bridge for API
│   └── dataset_info.py     # Dataset metadata extractor (1D & 2D)
├── datasets/               # 10 HDF5 datasets across 1D/2D PDEs and obstacle bodies
├── server/
│   ├── index.js            # Express API server (bridges UI to Python ML engine)
│   └── package.json
├── webapp/                 # Next.js 16 interactive dashboard & flow visualizer
│   ├── src/app/
│   │   ├── page.tsx        # Overview dashboard with live fluid sandbox
│   │   ├── inference/      # TTA execution, multi-step rollout & 2D flow viewer
│   │   ├── explore/        # Space-time waterfall & 10-dataset catalog explorer
│   │   ├── train/          # Training telemetry & live loss streaming
│   │   ├── evaluate/       # Benchmark reporting (2D Shapes + Geometric + Ablations)
│   │   └── components/
│   │       ├── FlowVisualizer.tsx    # 1D Canvas fluid wave & particle simulator
│   │       ├── FlowVisualizer2D.tsx  # 2D obstacle flow heatmap & streamline renderer
│   │       └── Sidebar.tsx
│   └── src/app/globals.css # Warm brown/beige editorial design system
├── tests/
│   └── test_physics_loss.py # Smoke tests for physics loss, model & TTA engine
├── real_burgers_model.pt   # Pretrained MeshGraphNet checkpoint
├── PROJECT_REPORT.md       # Full academic project report
├── PRESENTATION_SLIDES.md  # 12-slide conference/defense deck
├── BENCHMARK_RESULTS_SUMMARY.md  # Consolidated results across all experiments
├── benchmark_summary.csv   # Machine-readable benchmark matrix
├── requirements.txt        # Python dependency specifications
├── LICENSE                 # MIT License
├── run_experiments.bat     # Windows 1-click reproduction script
└── README.md
```

---

## 🔬 Methodology & Formulations

### 1. GNN Surrogate (MeshGraphNet)
The surrogate operates on PyTorch Geometric `Data` objects representing the spatial mesh:
* **Encoder**: Independent MLPs projecting node features $u(x, t)$ and edge vectors $\Delta x$ into a $128$-dimensional latent manifold.
* **Processor**: $10$ consecutive message-passing blocks with residual connections:
  $$\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \text{MLP}_{\text{edge}}\left(\mathbf{v}_i^{(l)}, \mathbf{v}_j^{(l)}, \mathbf{e}_{ij}^{(l)}\right)$$
  $$\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \text{MLP}_{\text{node}}\left(\mathbf{v}_i^{(l)}, \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)}\right)$$
* **Decoder**: Un-normalized MLP mapping node latent states to predicted state $\hat{u}_{t+1}$.

### 2. Physics-Residual Loss Function (PINN-style)
For the 1D viscous Burgers' equation:
$$\mathcal{R}(u) = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$$

Given the input physical state $u_t$ and surrogate prediction $\hat{u}_{t+1}$:
1. **Temporal derivative**:
   $$\frac{\partial u}{\partial t} \approx \frac{\hat{u}_{t+1} - u_t}{\Delta t}$$
2. **Spatial derivatives (Central Difference with Periodic Boundaries)**:
   $$\frac{\partial \hat{u}_{t+1}}{\partial x} \approx \frac{\hat{u}_{t+1}[i+1] - \hat{u}_{t+1}[i-1]}{2 \Delta x}$$
   $$\frac{\partial^2 \hat{u}_{t+1}}{\partial x^2} \approx \frac{\hat{u}_{t+1}[i+1] - 2\hat{u}_{t+1}[i] + \hat{u}_{t+1}[i-1]}{\Delta x^2}$$
3. **Unsupervised PDE Loss**:
   $$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \left| \frac{\hat{u}_{t+1} - u_t}{\Delta t} + \hat{u}_{t+1} \frac{\partial \hat{u}_{t+1}}{\partial x} - \nu \frac{\partial^2 \hat{u}_{t+1}}{\partial x^2} \right|^2$$

### 3. Test-Time Adaptation (TTA) Algorithm
At test time on an unlabelled sample $(x_t, \text{mesh})$:
1. Compute the baseline surrogate prediction $\hat{u}_0$.
2. **Freeze** the GNN Encoder and all 10 ProcessorBlocks to preserve geometric message-passing representations.
3. Optimize only the **Decoder MLP** parameters $\theta_{\text{dec}}$ for $K = 10\text{--}15$ steps:
   $$\min_{\theta_{\text{dec}}} \mathcal{L}_{\text{physics}}\left(\hat{u}_{t+1}, u_t; \nu\right) + \lambda_{\text{anchor}} \|\hat{u}_{t+1} - \hat{u}_0\|^2$$
   where $\lambda_{\text{anchor}} = 0.1$ prevents representation drift away from the surrogate manifold.
4. **Best Physical State Retention**: Retain the model parameters corresponding to the minimum $\mathcal{L}_{\text{physics}}$.
5. Restore baseline model weights after adaptation so subsequent inferences remain clean.

---

## 📈 Empirical Benchmarking Results

Evaluated across held-out PDEBench trajectories using `scripts/evaluate_tta.py`:

### Experiment 1: In-Distribution Evaluation ($\nu = 0.001$)
| Test Trajectory | Timestep | Baseline MSE (Pre-TTA) | Adapted MSE (Post-TTA) | Accuracy Gain |
|:---:|:---:|:---:|:---:|:---:|
| Trajectory 0 | $t=10$ | $1.760 \times 10^{-3}$ | $1.182 \times 10^{-4}$ | **+93.28%** |
| Trajectory 3333 | $t=15$ | $4.413 \times 10^{-4}$ | $4.397 \times 10^{-4}$ | **+0.35%** |
| Trajectory 6666 | $t=20$ | $4.587 \times 10^{-5}$ | $2.873 \times 10^{-5}$ | **+37.35%** |
| **Mean Aggregate** | — | **$7.491 \times 10^{-4}$** | **$1.955 \times 10^{-4}$** | **+73.89% error reduction** |

### Experiment 2: Parameter Distribution Shift (ν = 0.002)
| Metric | Baseline Surrogate | Physics-Residual TTA | Change |
|---|:---:|:---:|:---:|
| **Mean MSE** | $7.491 \times 10^{-4}$ | $1.939 \times 10^{-4}$ | **+74.11% reduction** |
| **Mean Relative MSE** | $2.255 \times 10^{-3}$ | $1.070 \times 10^{-3}$ | **+52.52% reduction** |
| **Mean Physics Residual** | $47.41$ | $27.36$ | **-42.29% drop** |

### Experiment 3: Geometric Distribution Shift Benchmark (Unseen Mesh Discretizations)
Evaluates when spatial mesh coordinates are non-uniform, distorted, or scale-shifted:

| Geometric Shift Regime | Mesh Characteristics | Baseline GNN MSE | Physics-Residual TTA MSE | Error Reduction (%) | Residual Drop (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Stochastic Irregular Mesh** | Random node jitter $\Delta x \pm 35\%$ | $5.017 \times 10^{-2}$ | $4.203 \times 10^{-2}$ | **+16.22%** | **-23.54%** |
| **Boundary-Layer Warping** | Non-uniform refinement ($x \mapsto x + \alpha \sin(2\pi x)$) | $4.212 \times 10^{-2}$ | $3.443 \times 10^{-2}$ | **+18.26%** | **-12.55%** |
| **Resolution Scale Shift** | Subsampled mesh ($N=512$, $2\times$ edge distances) | $5.004 \times 10^{-2}$ | $3.815 \times 10^{-2}$ | **+23.77%** | **-17.65%** |

### Experiment 4: Systematic Ablation & Sensitivity Analysis

Evaluates the key architectural and optimization choices across held-out test trajectories:

1. **Adaptation Step Budget ($K$)**:
   * $K=0$ (Unadapted): MSE $3.670 \times 10^{-2}$, PDE Residual $497.8$ (Baseline)
   * $K=3$: MSE $1.538 \times 10^{-2}$ (**+58.10% Error Reduction**, Residual drops to $314.0$)
   * $K=12$: MSE $1.662 \times 10^{-2}$ (**+54.72% Error Reduction**, Residual drops to $267.1$)
   * $K=20$: MSE $1.550 \times 10^{-2}$ (**+57.78% Error Reduction**, Residual drops to $241.3$)
   * *Conclusion*: Rapid convergence within $K=3\text{--}12$ steps; PDE residual drops monotonically by $-51.5\%$.

2. **Parameter Adaptation Scope (Decoder-Only vs. Full GNN)**:
   * **Decoder Only (Ours)**: MSE $1.640 \times 10^{-2}$ (**+55.31% Error Reduction**), Latency **363.2 ms**
   * **Full GNN (All Layers)**: MSE $2.205 \times 10^{-2}$ (+39.92% Error Reduction), Latency **1432.5 ms** (3.9&times; slower)
   * *Conclusion*: Freezing GNN message passing blocks preserves geometric embeddings, yields higher generalization accuracy, and runs nearly $4\times$ faster.

3. **Test-Time Learning Rate ($\eta$)**:
   * Peak stability between $\eta = 4 \times 10^{-5}$ and $8 \times 10^{-5}$ (+57.8% gain). Excess step sizes ($\eta \ge 5 \times 10^{-4}$) cause gradient overshoot.

4. **Anchor Regularization Weight ($\lambda_{\text{anchor}}$)**:
   * $\lambda_{\text{anchor}} = 0.1$ prevents representation drift away from the learned surrogate manifold while allowing flexible physical compliance.

---

## 💻 CLI Usage

### 1. Run Test-Time Adaptation on a Sample
```bash
python scripts/tta.py \
  --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
  --checkpoint_path real_burgers_model.pt \
  --traj 0 \
  --timestep 10 \
  --tta_steps 12 \
  --tta_lr 0.00008
```

### 2. Run Batch TTA Benchmark (Parameter Shift)
```bash
# In-distribution benchmark across 10 test trajectories
python scripts/evaluate_tta.py \
  --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
  --checkpoint_path real_burgers_model.pt \
  --num_samples 10 \
  --tta_steps 12

# Under viscosity distribution shift (nu = 0.002)
python scripts/evaluate_tta.py \
  --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
  --checkpoint_path real_burgers_model.pt \
  --num_samples 10 \
  --nu_shift 0.002
```

### 3. Run Geometric Distribution Shift Benchmark (Irregular & Warped Meshes)
```bash
python scripts/evaluate_geometric_shift.py \
  --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
  --checkpoint_path real_burgers_model.pt \
  --num_samples 10 \
  --shift_type all
```

### 4. Run Systematic Ablation Suite
```bash
python scripts/evaluate_ablations.py \
  --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
  --checkpoint_path real_burgers_model.pt \
  --num_samples 6
```

### 5. Download / Synthesize All 10 Benchmark Datasets
```bash
python scripts/download_pde_datasets.py --force_synth
```

### 6. Run 2D Shapes & Multi-Model Benchmark Suite
```bash
python scripts/evaluate_2d_benchmarks.py
```

### 7. Start Interactive Web Application
```bash
# Terminal 1: Backend API
cd server
node index.js

# Terminal 2: Next.js Frontend
cd webapp
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser:
* **`/evaluate`**: 2D Shapes & Models tab (FNO-2D, UNet-2D, MeshGraphNet across Cylinder, Airfoil, Square, Step), Geometric Shift tab, and Systematic Ablation tab.
* **`/inference`**: Interactive TTA execution, side-by-side comparison plots, and live particle advection flows.
* **`/explore`**: 10-dataset catalog explorer with quick preset selection and space-time waterfalls.
* **`/train`**: Supervised baseline training telemetry.

---

## 🌊 2D Geometric Obstacle Shapes & Multi-Model Surrogates

The framework extends beyond 1D to 2D continuous fields and fluid flow past complex aerodynamic bodies:

| Benchmark Shape / Domain | Physical Regime | Grid Size | Pre-TTA MSE | Post-TTA MSE | TTA Error Reduction (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Circular Cylinder** | Karman Vortex Street ($Re \sim 85$) | $48 \times 96$ | $7.04 \times 10^{-3}$ | $6.05 \times 10^{-4}$ | **+91.41%** (FNO-2D) |
| **NACA 0012 Airfoil** | Boundary-layer wake separation | $48 \times 96$ | $2.78 \times 10^{-3}$ | $3.27 \times 10^{-4}$ | **+88.22%** (FNO-2D) |
| **Square Bluff Body** | Sharp-corner recirculation | $48 \times 96$ | $1.43 \times 10^{-2}$ | $4.64 \times 10^{-4}$ | **+96.77%** (UNet-2D) |
| **Backward-Facing Step** | Sudden expansion channel flow | $48 \times 96$ | $9.30 \times 10^{-4}$ | $2.82 \times 10^{-4}$ | **+69.70%** (FNO-2D) |
| **Multi-Cylinder Array** | Tandem wake interference | $48 \times 96$ | $1.58 \times 10^{-3}$ | $2.61 \times 10^{-4}$ | **+83.50%** (FNO-2D) |
| **2D Advection Flow** | Rotating scalar transport | $64 \times 64$ | $1.24 \times 10^{-2}$ | $6.35 \times 10^{-3}$ | **+48.79%** (FNO-2D) |
| **2D Heat Diffusion** | Continuous thermal dissipation | $64 \times 64$ | $2.20 \times 10^{-2}$ | $2.06 \times 10^{-2}$ | **+6.33%** (FNO-2D) |

### Supported Surrogate Architectures
* **Fourier Neural Operator (FNO-2D)**: Truncated 2D spectral convolution via Fast Fourier Transforms ($1,188,353$ parameters).
* **Physics U-Net (UNet-2D)**: Multi-scale convolutional encoder-decoder with residual skip connections ($467,233$ parameters).
* **MeshGraphNet**: 10-layer message-passing neural network operating on Delaunay-triangulated spatial meshes ($1,079,041$ parameters).
* **Graph Attention Network (GATSurrogate)**: Multi-head attention on mesh graphs ($125,473$ parameters).

---

## 📚 References
* Pfaff, T., et al. (2021). *Learning Mesh-Based Simulation with Graph Networks*. ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
* Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
* Takamoto, M., et al. (2022). *PDEBench: An Extensive Benchmark for Scientific Machine Learning*. NeurIPS 2022. [GitHub](https://github.com/pdebench/PDEBench)
* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics.
