# Physics-Residual Test-Time Adaptation (TTA) for GNN PDE Surrogates
## Academic Defense & Conference Presentation Slide Deck

**Author:** Neha Saravanan  
**Repository:** [github.com/nehasaravanan128/Physics-Residual-TTA-for-GNN-PDE-Surrogates](https://github.com/nehasaravanan128/Physics-Residual-TTA-for-GNN-PDE-Surrogates)  
**Keywords:** Scientific Machine Learning, Graph Neural Networks, Test-Time Adaptation, PINNs, PDEBench, Burgers' Equation  

---

<!-- slide -->

# Slide 1: Title & Executive Summary

### Physics-Residual Test-Time Adaptation for Graph Neural Network PDE Surrogates
*Unsupervised Online Adaptation Under Physical Parameter and Geometric Mesh Shifts*

* **The Problem:** Supervised GNN surrogates (MeshGraphNets) suffer from severe generalization failure when deployed under unseen Reynolds numbers or non-uniform spatial discretizations.
* **The Core Idea:** Exploit the governing PDE residual $\mathcal{R}(u) = 0$ as a zero-label loss function at inference time to adapt the surrogate directly on unlabelled test queries.
* **Key Achievements:**
  * **+73.9%** mean error reduction in-distribution (up to **+93.3%** on shock fronts).
  * **+74.1%** error reduction under parameter viscosity shift ($\nu = 0.002$).
  * **+16.2% to +23.8%** error reduction under non-uniform geometric mesh distortions.
  * **3.9&times; faster** and higher accuracy by adapting the Decoder MLP while freezing GNN message-passing blocks.

---

<!-- slide -->

# Slide 2: Motivation & Problem Statement

### The Generalization Dilemma in Scientific Machine Learning

```
+-------------------------------------------------------------+
| Traditional Numerical Solvers (FEM / FVM)                  |
| [Accurate] [Physics-Compliant] [Prohibitively Slow (O(N^3))]|
+-------------------------------------------------------------+
                              vs.
+-------------------------------------------------------------+
| Supervised GNN Surrogates (MeshGraphNets)                   |
| [Ultra-fast O(N)] [Rigid] [Fails Under Distribution Shift]  |
+-------------------------------------------------------------+
```

* **Why Surrogates Degrade:**
  1. **Parameter Shifts:** Changes in viscosity $\nu$ change the convective-diffusive balance.
  2. **Discretization Shifts:** Irregular or refined mesh coordinates perturb edge vectors $\Delta x$, corrupting message aggregations.
  3. **Multi-step Rollout Drift:** Compounding errors lead to numerical blow-up over long horizons.
* **Why Not Just Retrain?**
  * Generating high-fidelity training data requires running expensive numerical solvers, defeating the purpose of a fast surrogate!
  * **Challenge:** Can we adapt the surrogate at test time **without any ground-truth labels**?

---

<!-- slide -->

# Slide 3: Baseline Architecture: MeshGraphNets

### Message-Passing on Spatial Mesh Graphs

$$\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \text{MLP}_{\text{edge}}\left(\mathbf{v}_i^{(l)}, \mathbf{v}_j^{(l)}, \mathbf{e}_{ij}^{(l)}\right)$$
$$\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \text{MLP}_{\text{node}}\left(\mathbf{v}_i^{(l)}, \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)}\right)$$

* **Spatial Representation:** 1D/2D continuum discretized as graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.
* **Node Features:** Local physical state $u(x, t) \in \mathbb{R}^1$.
* **Edge Features:** Relative spatial displacement vectors $\Delta x_{ij} = x_j - x_i$.
* **Deep Processor:** 10 consecutive InteractionNetwork blocks with residual skips and $128$-dimensional latent channels.
* **Decoder:** Unnormalized MLP projecting latent representations back to physical velocity $\hat{u}_{t+1}$.

---

<!-- slide -->

# Slide 4: Proposed Methodology: Physics-Residual TTA

### Closing the Generalization Gap at Test Time

```
Query (u_t, Mesh) ──> [ Frozen Encoder ] ──> [ Frozen 10x Processor ] 
                                                        │
                                                        ▼
                    [ Differentiable PDE Residual Engine ] <── [ Adaptable Decoder ]
                                      │                                 ▲
                                      ▼                                 │
                   Physics Loss + Anchor Regularization ───────────────┘
                               (K = 10-12 steps)
```

1. **Initial Forward Pass:** Compute baseline prediction $\hat{u}_0 = \mathcal{M}(u_t; \Theta_0)$.
2. **Selective Freezing:** Freeze Encoder & Processor to preserve topological graph embeddings.
3. **Unsupervised Optimization:** Optimize decoder parameters $\theta_{\text{dec}}$ against the PDE residual.
4. **Anchor Regularization:** Prevent representation drift away from the surrogate manifold.
5. **Clean Restoration:** Model weights are reset after inference to prevent cross-sample contamination.

---

<!-- slide -->

# Slide 5: Non-Uniform Conservative Finite-Difference Stencils

### Differentiable Physics Engine for Arbitrary Graphs

For the viscous Burgers' equation $\partial_t u + u \partial_x u - \nu \partial_{xx} u = 0$:

* **Temporal Derivative:**
  $$\left.\frac{\partial u}{\partial t}\right|_i \approx \frac{\hat{u}_{i, t+1} - u_{i, t}}{\Delta t}$$

* **Non-Uniform Conservative Spatial Stencils ($h_i = x_{i+1} - x_i$):**
  $$\left.\frac{\partial \hat{u}}{\partial x}\right|_i \approx \frac{\hat{u}_{i+1} - \hat{u}_{i-1}}{h_i + h_{i-1}}$$
  $$\left.\frac{\partial^2 \hat{u}}{\partial x^2}\right|_i \approx \frac{2}{h_i + h_{i-1}}\left(\frac{\hat{u}_{i+1} - \hat{u}_i}{h_i} - \frac{\hat{u}_i - \hat{u}_{i-1}}{h_{i-1}}\right)$$

* **Full Unsupervised Objective:**
  $$\min_{\theta_{\text{dec}}} \frac{1}{N} \sum_{i=1}^N \left| \frac{\hat{u}_i - u_i}{\Delta t} + \hat{u}_i \left.\frac{\partial \hat{u}}{\partial x}\right|_i - \nu \left.\frac{\partial^2 \hat{u}}{\partial x^2}\right|_i \right|^2 + \lambda_{\text{anchor}} \|\hat{u} - \hat{u}_0\|^2$$

---

<!-- slide -->

# Slide 6: Experiment 1: In-Distribution Benchmark ($\nu = 0.001$)

### Quantitative Results Across Held-Out PDEBench Trajectories

| Evaluation Trajectory | Timestep | Baseline GNN MSE | Physics TTA MSE | Error Reduction (%) |
|:---:|:---:|:---:|:---:|:---:|
| Trajectory #0 | $t=10$ | $1.760 \times 10^{-3}$ | $1.182 \times 10^{-4}$ | **+93.28%** |
| Trajectory #3333 | $t=15$ | $4.413 \times 10^{-4}$ | $4.397 \times 10^{-4}$ | **+0.35%** |
| Trajectory #6666 | $t=20$ | $4.587 \times 10^{-5}$ | $2.873 \times 10^{-5}$ | **+37.35%** |
| **Mean Aggregate** | — | **$7.491 \times 10^{-4}$** | **$1.955 \times 10^{-4}$** | **+73.89%** |

* **Key Takeaway:** Peak accuracy improvements (+93.3%) occur in shock-front regions where standard GNNs suffer from high numerical diffusion.

---

<!-- slide -->

# Slide 7: Experiment 2: Physical Parameter Shift ($\nu = 0.002$)

### Evaluating Out-of-Distribution Physical Viscosity

```
Baseline MSE:     [██████████████████████████████] 7.491e-4
Physics-TTA MSE:  [███████                       ] 1.939e-4  (-74.11% Error)
```

| Metric | Supervised Baseline | Physics-Residual TTA | Improvement |
|---|:---:|:---:|:---:|
| **Mean MSE** | $7.491 \times 10^{-4}$ | $1.939 \times 10^{-4}$ | **+74.11% reduction** |
| **Relative L2 MSE** | $2.255 \times 10^{-3}$ | $1.070 \times 10^{-3}$ | **+52.52% reduction** |
| **Mean PDE Residual** | $47.41$ | $27.36$ | **-42.29% drop** |

* Zero training labels were provided for $\nu = 0.002$.
* The unsupervised residual loss actively pulls the prediction toward physically valid dissipation rates.

---

<!-- slide -->

# Slide 8: Experiment 3: Geometric Mesh Discretization Shift

### Robustness on Irregular, Warped, and Subsampled Spatial Graphs

| Shift Regime | Mesh Characteristics | Baseline MSE | Physics TTA MSE | Gain (%) | Residual Drop |
|---|---|:---:|:---:|:---:|:---:|
| **1. Stochastic Irregular** | Random node jitter $\Delta x \pm 35\%$ | $5.017 \times 10^{-2}$ | $4.203 \times 10^{-2}$ | **+16.22%** | **-23.54%** |
| **2. Boundary Warping** | Refinement ($x \mapsto x + \alpha \sin(2\pi x)$) | $4.212 \times 10^{-2}$ | $3.443 \times 10^{-2}$ | **+18.26%** | **-12.55%** |
| **3. Resolution Scale Shift**| Subsampled mesh ($N=512$, $2\times$ edges) | $5.004 \times 10^{-2}$ | $3.815 \times 10^{-2}$ | **+23.77%** | **-17.65%** |

* **Significance:** Standard GNN surrogates break down when deployed on unaligned meshes; non-uniform conservative finite differences restore physical fidelity.

---

<!-- slide -->

# Slide 9: Systematic Ablation Studies

### Validating Architectural & Optimization Choices

1. **Parameter Scope:**
   * **Decoder-Only (Proposed):** +55.3% Error Reduction | **363.2 ms** latency
   * **Full GNN (All Layers):** +39.9% Error Reduction | 1432.5 ms latency (3.9&times; slower)
   * *Conclusion:* Freezing message passing prevents manifold collapse on few-step adaptation.

2. **Adaptation Step Budget ($K$):**
   * Rapid convergence in $K = 3\text{--}12$ steps; PDE residual drops monotonically by **$-51.5\%$**.

3. **Anchor Weight ($\lambda_{\text{anchor}}$):**
   * $\lambda = 0.1$ maintains surrogate manifold integrity while allowing adaptation.

---

<!-- slide -->

# Slide 10: Interactive Web Application & Visualizer

### Editorial Laboratory Interface (Next.js 16 + Express)

* **Continuum Wave Rollout:** Custom 60 FPS HTML5 Canvas engine rendering particle advection and velocity shock fronts.
* **Side-by-Side Inspection:** Instant visual comparison of Unadapted Baseline, Adapted Physics TTA, and Numerical Ground Truth.
* **Multi-Tab Evaluation Suite:** Interactive exploration of In-Distribution, Geometric Shift, and Ablation benchmarks.
* **1-Click Reproduction:** Full REST API bridge triggering PyTorch benchmarking runs asynchronously.

---

<!-- slide -->

# Slide 11: Summary & Conclusions

### Key Contributions & Takeaways

1. **First-of-its-Kind GNN TTA Formulation:** Proved that unsupervised PDE residuals can guide test-time adaptation for graph neural network surrogates.
2. **Conservative Geometric Stencils:** Extended PINN residual formulations to arbitrary non-uniform 1D meshes.
3. **Rigorous Empirical Validation:** Demonstrated superior generalization across in-distribution (+73.9%), parameter shifts (+74.1%), and geometric shifts (+23.8%).
4. **Computational Efficiency:** Decoder-only adaptation achieves 3.9&times; speedup over full fine-tuning with higher accuracy.
5. **Open Reproducibility:** End-to-end open-source repository with automated CLI runners, web dashboard, and complete documentation.

---

<!-- slide -->

# Slide 12: Future Work & Discussion

### Roadmap for Production Deployment

* **2D Complex Geometries:** Extension to unstructured triangular meshes for 2D Navier-Stokes flow around obstacles (CylinderFlow).
* **Self-Paced Learning Rates:** Conditioning $\eta$ dynamically on the initial residual magnitude $\|\mathcal{R}(u_0)\|$.
* **Periodic Autoregressive TTA:** Applying 2-3 adaptation steps every $m$ steps during multi-step rollouts to eliminate compounding drift.

---
### Thank You! Q & A
* **Code & Report:** [github.com/nehasaravanan128/Physics-Residual-TTA-for-GNN-PDE-Surrogates](https://github.com/nehasaravanan128/Physics-Residual-TTA-for-GNN-PDE-Surrogates)
