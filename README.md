# CardioXAI — Feasibility-Guided Cardiovascular Risk Optimization

> **Major Project I · Semester VII**  
> A PhD-level research system combining Explainable AI (XAI), Survival Analysis, and Counterfactual Optimization for cardiovascular risk assessment and clinical decision support.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation & Problem Statement](#2-research-motivation--problem-statement)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [ML Pipeline — Phase-by-Phase Breakdown](#5-ml-pipeline--phase-by-phase-breakdown)
6. [Backend API](#6-backend-api)
7. [Frontend — Pages & UI](#7-frontend--pages--ui)
8. [Results & Key Findings](#8-results--key-findings)
9. [Project Structure](#9-project-structure)
10. [Setup & Running the App](#10-setup--running-the-app)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

**CardioXAI** is a full-stack clinical decision-support system built as a Major Project for Sem VII. It predicts a patient''s 10-year cardiovascular risk using an ensemble of survival and machine-learning models, and then — uniquely — generates **counterfactual prescriptions**: personalized, clinically constrained lifestyle recommendations that, if followed, would meaningfully reduce that patient''s hazard score.

The system is served as a **FastAPI web application** with a rich, dark-themed UI consisting of five pages:

| Page | URL Route | Purpose |
|---|---|---|
| **Landing** | `/` | Hero page introducing CardioXAI |
| **Analysis Dashboard** | `/analysis` | Live patient risk assessment & XAI engine |
| **Methodology** | `/methodology` | Technical deep-dive into the research pipeline |
| **About / Research** | `/about` | Dataset background, Framingham Study context |
| **Legacy Dashboard** | `index.html` | Earlier Tailwind-based patient form (reference) |

---

## 2. Research Motivation & Problem Statement

Cardiovascular disease (CVD) is the leading cause of global mortality. Predictive models exist, but they typically answer only *"What is this patient''s risk?"* — not *"What can this patient do to lower it?"*

**CardioXAI addresses this gap** by framing the problem as a **counterfactual search**:

> *"What is the minimum, clinically plausible change to this patient''s modifiable risk factors that would result in a meaningfully lower predicted hazard?"*

This requires three things to work together:
1. A **reliable risk model** (Cox Proportional Hazards) as the objective function.
2. A **differentiable physiological manifold** (Split-Head Autoencoder) to ensure any generated counterfactual lies on the distribution of real patient data.
3. A **constrained gradient optimizer** that respects clinical reality (immutable features, directional constraints, absolute bounds).

---

## 3. Dataset

**Framingham Heart Study — `frmgham2.csv`**

- One of the most cited longitudinal cardiovascular cohort studies.
- **Period 1 (Baseline)** only is used — preventing temporal data leakage across study periods.
- **Target**: `ANYCHD` (binary: any coronary heart disease event) + `TIMECHD` (time-to-event in days, converted to years).
- **Class imbalance**: ~9.6% event rate — handled via class weighting + Isotonic Recalibration.

### Features Used

| Category | Features |
|---|---|
| **Immutable** | `SEX`, `PREVCHD`, `PREVAP`, `PREVMI`, `PREVSTRK`, `PREVHYP`, `educ` (one-hot) |
| **Directional** (age-only) | `AGE` |
| **Actionable** | `TOTCHOL`, `SYSBP`, `DIABP`, `CURSMOKE`, `CIGPDAY`, `BMI`, `DIABETES`, `BPMEDS`, `HEARTRTE`, `GLUCOSE` |

> **Note**: `HDLC` and `LDLC` were dropped due to high missingness. `SEX` was remapped from `{1,2}` to `{0,1}` for compatibility with BCE loss in the autoencoder.

---

## 4. System Architecture

```
+----------------------------------------------------------+
|                    FastAPI Backend                        |
|                      (src/api.py)                        |
|                                                          |
|  +------------+  +--------------+  +-----------------+  |
|  | Data       |  | ML Models    |  | Counterfactual  |  |
|  | Pipeline   |  | CoxPH / XGB  |  | Optimizer       |  |
|  | (Phase 1)  |  | LGB / RF     |  | (Phase 4)       |  |
|  +------------+  +--------------+  +-----------------+  |
|                        |                   |             |
|         REST API: /api/predict, /api/optimize            |
|                   /api/patients, /api/patient            |
+------------------------+---------------------------------+
                         |  Jinja2 Templates
          +--------------+-----------------------------+
          |          Frontend (5 pages)                |
          |  landing / analysis / methodology /        |
          |  about / index                             |
          +--------------------------------------------+
```

---

## 5. ML Pipeline — Phase-by-Phase Breakdown

### Phase 1 · Data Pipeline

**File**: `src/data_pipeline.py`

The pipeline performs a rigorous, **leakage-free** preprocessing sequence:

1. **Filter** to `PERIOD == 1` (baseline only).
2. **Train/Test split first** (80/20, stratified by `ANYCHD`) — imputation statistics are computed only on the training set, then applied to the test set.
3. **RANDID overlap assertion** — hard check to guarantee no patient appears in both splits.
4. **Imputation**:
   - Continuous features ? **median imputation**
   - Binary & categorical features ? **mode imputation**
   - Saved to `models/imputation_dict.joblib`
5. **One-hot encode** `educ` ? `educ_1.0`, `educ_2.0`, `educ_3.0`, `educ_4.0`
6. **StandardScaler** fit on train continuous features, saved to `models/scaler.joblib`

---

### Phase 2 · Model Comparison

**Notebook**: `notebooks/01_model_comparison.ipynb`

Five models were trained and benchmarked on the stratified hold-out test set:

| Model | AUC-ROC | PR-AUC | Optimal F1 | Brier Score |
|---|---|---|---|---|
| **CoxPH (10-yr)** | **0.7951** | **0.2458** | **0.3216** | **0.0759** |
| DNN | 0.7875 | 0.2077 | 0.3152 | 0.0797 |
| Random Forest | 0.7607 | 0.2043 | 0.2724 | 0.0775 |
| LightGBM | 0.7443 | 0.1838 | 0.1918 | 0.0794 |
| XGBoost | 0.7284 | 0.1797 | 0.2749 | 0.0789 |

**Key decisions:**
- Probability thresholds for F1/Accuracy were optimized on a **separate validation set**, applied blindly to the test set (no threshold leakage).
- Tree models used class weights to handle imbalance; raw probabilities were **post-hoc recalibrated via Isotonic Regression**.
- **CoxPH selected as primary risk engine** — best AUC, best calibration, interpretable coefficients.

---

### Phase 3 · Autoencoder Latent Space

**File**: `src/autoencoder.py`  
**Notebook**: `notebooks/02_latent_space_analysis.ipynb`

#### Architecture: `SplitHeadAutoencoder`

A custom PyTorch autoencoder handling **mixed-type clinical data** (continuous, binary, categorical):

```
Input (21-dim)
      |
  Encoder: Linear(21->16) -> ReLU -> BatchNorm -> Linear(16->latent_dim)
      |
  Latent Space (8-dim)
      |
  Decoder Body: Linear(8->16) -> ReLU -> BatchNorm
      |-- Head Cont:  Linear(16 -> 8)   <- MSE Loss
      |-- Head Bin:   Linear(16 -> 9)   <- BCEWithLogits Loss
      +-- Head Cat:   Linear(16 -> 4)   <- CrossEntropy Loss
```

#### Latent Dimension Selection

Sweep over `[2, 3, 4, 6, 8, 12, 16]` dimensions; **Kneedle Algorithm** identified elbow at **Dimension 8**:

| Latent Dim | Total Loss | Cont MSE | Bin Acc | Cat Acc |
|---|---|---|---|---|
| 2 | 0.6691 | 0.5294 | 94.81% | 99.85% |
| 4 | 0.3643 | 0.2740 | 96.61% | 100.0% |
| **8** | **0.1005** | **0.0467** | **98.36%** | **100.0%** |
| 12 | 0.0128 | 0.0049 | 99.84% | 100.0% |
| 16 | 0.0034 | 0.0016 | 99.98% | 100.0% |

#### Reconstruction Error in Clinical Units (at dim=8)

| Feature | RMSE |
|---|---|
| AGE | +/-1.45 years |
| TOTCHOL | +/-5.14 mg/dL |
| SYSBP | +/-7.05 mmHg (largest — see caveat) |
| DIABP | +/-4.39 mmHg |
| BMI | +/-0.71 kg/m2 |
| HEARTRTE | +/-2.22 bpm |
| GLUCOSE | +/-2.32 mg/dL |

> **SYSBP caveat**: Reconstruction noise of +/-7 mmHg is modest relative to the full range (83–295 mmHg) but can straddle hypertension staging thresholds (10 mmHg bands). Fine-grained BP counterfactuals near boundaries should be interpreted with caution.

---

### Phase 4 · Counterfactual Optimizer

**File**: `src/optimizer.py`  
**Notebook**: `notebooks/03_counterfactual_examples.ipynb`

#### How It Works

Gradient descent in the AE latent space minimizes CoxPH log-hazard, subject to clinical constraints.

**Objective Function:**

```
L = lambda_risk * L_risk  +  lambda_prox * L_prox  +  lambda_imm * L_imm
    + lambda_dir * L_dir  +  lambda_bound * L_bound
```

| Loss Term | Description |
|---|---|
| `L_risk` | CoxPH log-hazard of decoded counterfactual (minimize this) |
| `L_prox` | L2 distance in latent space from original z |
| `L_imm` | L2 anchor on immutable features (SEX, PREVCHD, DIABETES, educ…) |
| `L_dir` | Squared hinge loss enforcing directional monotonicity |
| `L_bound` | Squared hinge loss enforcing per-patient relative magnitude caps |

#### Clinical Constraints

| Feature | Direction | Cap | Absolute Floor | Justification |
|---|---|---|---|---|
| `TOTCHOL` | Decrease-only | 20% | 120 mg/dL | Cox coef +0.182 |
| `SYSBP` | Decrease-only | 20% | 90 mmHg | Cox coef +0.170 |
| `BMI` | Decrease-only | 20% | 18.5 | Cox coef +0.108 |
| `GLUCOSE` | Decrease-only | 20% | 65 mg/dL | Cox coef +0.068 |
| `CIGPDAY` | Decrease-only | 50% | 0 | Cox coef +0.063 |
| `DIABP` | Decrease-only | 20% | 60 mmHg | Cox coef +0.059 |
| `CURSMOKE` | Decrease-only | binary | 0 | Cox coef +0.077 |
| `HEARTRTE` | **Decrease-only** | 15% | 50 bpm | OVERRIDE: Cox coef is -0.056 but patients must not raise resting HR |
| `BPMEDS` | **Increase-only** | binary | — | OVERRIDE: Positive Cox coef from confounding; patients must not stop medication |
| `DIABETES` | **Immutable** | — | — | Permanent diagnosis |
| `AGE` | **Immutable** | — | — | Cannot change |
| All `PREV*`, `SEX`, `educ` | **Immutable** | — | — | Demographics & prior events |

#### 3-Tier Success Status

| Status | Condition |
|---|---|
| `strong` | log-hazard drop >= 0.05 |
| `partial` | log-hazard drop >= 0.01 |
| `none` | No meaningful improvement — system reports "No feasible recourse" |

---

## 6. Backend API

**File**: `src/api.py` — FastAPI + Jinja2 templates

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/analysis` | Patient risk dashboard |
| `GET` | `/methodology` | Methodology page |
| `GET` | `/about` | Research & dataset info |
| `GET` | `/api/patients` | List all available patient profiles |
| `GET` | `/api/patient/{type}/{idx}` | Fetch patient data (de-scaled to native units) |
| `POST` | `/api/predict` | Run full risk prediction |
| `POST` | `/api/optimize` | Generate counterfactual prescription |
| `POST` | `/api/patient` | Save a new custom patient to CSV |

### `/api/predict` Response Shape

```json
{
  "log_hazard": 0.312,
  "survival_times": [1.0, 2.0, "...", 10.0],
  "survival_probs": [0.98, 0.96, "...", 0.85],
  "prob_xgb": 0.21,
  "prob_lgb": 0.18,
  "prob_rf": 0.19,
  "ensemble_votes": 2,
  "xai_contributions": { "SYSBP": 0.053, "AGE": 0.041, "...": "..." }
}
```

### `/api/optimize` Response Shape

```json
{
  "success": true,
  "status": "strong",
  "relative_risk_reduction": 0.14,
  "prescription": [
    {"feature": "SYSBP", "original": 162, "recommended": 140, "change": -22},
    {"feature": "BMI", "original": 31.2, "recommended": 27.4, "change": -3.8}
  ],
  "survival_times": ["..."],
  "survival_probs": ["..."],
  "cf_xai_contributions": {},
  "trajectory": [{"step": 0, "SYSBP": 162, "BMI": 31.2, "TOTCHOL": 240}]
}
```

---

## 7. Frontend — Pages & UI

All pages share a **dark glassmorphic design system**:

- **Color palette**: `#03050d` background, `#3b7eff` accent blue, `#00d4ff` cyan, `#8b5cf6` violet, `#10b981` emerald
- **Typography**: `Inter` (body), `Space Grotesk` (headings), `JetBrains Mono` (code)
- **Effects**: `backdrop-filter: blur`, gradient text, glassmorphic cards, smooth scroll, micro-animations

---

### Landing Page (`/`)

**File**: `templates/landing.html`

A cinematic hero page with an interactive **Three.js 3D particle network** rendered in the background:

- **Hero section**: Project title, tagline, animated gradient headline, dual CTAs
- **Stats row**: Key project metrics (dataset size, best AUC, latent dims, etc.)
- **Feature cards**: 3-column grid showcasing the four core capabilities
- **Architecture diagram**: Visual 4-phase pipeline overview
- **Footer**: Research context & navigation links

---

### Analysis Dashboard (`/analysis`)

**File**: `templates/analysis.html`

The **core interactive tool** — a 3-panel clinical application layout:

#### Left Sidebar (272px)
- Patient profile loader dropdown (base test patients + custom patients)
- Input form for all 21 clinical features, grouped as: Demographics, Vitals, Biomarkers & History
- **"Analyse Risk"** button ? calls `/api/predict`
- **"Generate Prescription"** button ? calls `/api/optimize`
- Add Patient modal to save new profiles

#### Centre Panel — Risk Overview
- **10-Year Survival Curve**: Chart.js area chart (CoxPH output)
- **Risk Gauges**: Circular SVG progress rings for CoxPH log-hazard, XGBoost, LightGBM, RF probabilities + ensemble vote badge
- **XAI Feature Contributions**: Horizontal bar chart showing each feature''s partial-hazard contribution (`beta_i * x_i`)

#### Right Panel — Counterfactual Prescription
- **Prescription table**: Feature-by-feature diff (Original ? Recommended ? Change with delta arrows)
- **Intervention survival curve**: Overlaid Chart.js comparing original vs. counterfactual survival
- **Optimizer trajectory chart**: Line chart tracking SYSBP, BMI, TOTCHOL across 300 gradient steps
- **Counterfactual XAI**: Separate feature-contribution bar chart for the generated plan
- **Clinical risk badges**: Color-coded status (green/amber/red) per biomarker vs. clinical thresholds

---

### Methodology Page (`/methodology`)

**File**: `templates/methodology.html`

A scrollable research document with a **sticky left sidebar table-of-contents**. Sections cover:

1. Data Pipeline — preprocessing, imputation strategy, stratified splitting
2. Model Comparison — metrics table, calibration & recalibration discussion
3. Autoencoder Architecture — split-head design, dimension sweep, loss formulation
4. Counterfactual Optimizer — mathematical objective, constraint specification
5. XAI Engine — local feature contributions via Cox partial-hazard decomposition
6. Clinical Safety Guardrails — BPMEDS override, HEARTRTE override, immutable anchoring

---

### About / Research Page (`/about`)

**File**: `templates/about.html`

Academic context page with sections:

- **Framingham Heart Study** background — study design, why it is the gold standard for CVD research
- **Dataset statistics** — cohort size, event rate, feature descriptions
- **Research contributions** — novelty of the counterfactual + XAI approach
- **Limitations** — SYSBP reconstruction noise, period-1-only scope, cohort generalizability
- **References** — key academic citations

---

## 8. Results & Key Findings

### Model Performance

- **CoxPH achieved the best overall performance** (AUC 0.795, Brier 0.0759), validating it as the primary risk engine.
- Isotonic Recalibration corrected probability inflation in tree models without sacrificing ranking performance.
- DNN (AUC 0.787, F1 0.315) serves as an excellent differentiable surrogate.

### Autoencoder

- **Dimension 8** was objectively identified via the Kneedle elbow algorithm.
- Binary Acc: 98.36%, Categorical Acc: 100% — high fidelity reconstruction of mixed-type clinical data.
- The latent space organizes patients along discrete axes (smoking, sex) with a continuous physiological substructure (age).

### Counterfactual Optimizer

- Clinical constraints prevent manifold hallucinations — the optimizer never recommends raising BP or stopping medications.
- **Relative Risk Reduction (RRR)** = `1 - exp(-delta_log_hazard)`, giving a clinically interpretable percentage.
- Honest fallback: when no feasible recourse exists, the system reports `none` status rather than fabricating a recommendation.

---

## 9. Project Structure

```
Project_A2/
|-- data/
|   +-- frmgham2.csv                   # Framingham Heart Study dataset
|
|-- models/                            # Serialized models & artifacts
|   |-- cox_model.joblib               # Primary: CoxPH survival model
|   |-- cox_feature_cols.joblib        # Feature ordering for CoxPH
|   |-- xgb_model.joblib
|   |-- lgb_model.joblib
|   |-- rf_model.joblib
|   |-- dnn_model.pth
|   |-- dnn_config.joblib
|   |-- autoencoder.pth                # Production autoencoder (dim=8)
|   |-- autoencoder_dim8.pth           # Sweep candidate (dim=8)
|   |-- ae_config.joblib               # AE architecture config
|   |-- scaler.joblib
|   +-- imputation_dict.joblib
|
|-- notebooks/
|   |-- 01_model_comparison.ipynb      # Phase 2: Train & evaluate all models
|   |-- 02_latent_space_analysis.ipynb # Phase 3: AE + UMAP/PCA visualization
|   |-- 03_counterfactual_examples.ipynb # Phase 4: Optimizer case studies
|   |-- generate_notebook.py
|   |-- generate_ae_notebook.py
|   +-- generate_optimizer_notebook.py
|
|-- results/
|   |-- training_summary.md
|   |-- ae_summary.md
|   |-- counterfactual_summary.md
|   |-- calibration_plot.png
|   |-- ae_elbow_plot.png
|   +-- latent_space_projection.png
|
|-- src/
|   |-- api.py                         # FastAPI app & all endpoints
|   |-- data_pipeline.py               # Preprocessing, imputation, scaling
|   |-- autoencoder.py                 # SplitHeadAutoencoder + training loop
|   +-- optimizer.py                   # CounterfactualOptimizer
|
|-- templates/
|   |-- landing.html                   # Three.js hero landing page
|   |-- analysis.html                  # Patient risk dashboard (main app)
|   |-- methodology.html               # Technical methodology page
|   |-- about.html                     # Research & dataset background
|   +-- index.html                     # Legacy Tailwind dashboard
|
|-- static/                            # Static assets
|-- requirements.txt
+-- README.md
```

---

## 10. Setup & Running the App

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### 1. Install dependencies

```bash
cd "Project_A2"

# Create & activate venv
python -m venv venv
venv\Scripts\activate          # Windows PowerShell

# Install
pip install -r requirements.txt
pip install fastapi uvicorn jinja2 python-multipart
```

### 2. (Optional) Retrain models

Models are already serialized in `models/`. To retrain from scratch, run notebooks in order:

```
01_model_comparison.ipynb        --> trains CoxPH, XGB, LGB, RF, DNN
02_latent_space_analysis.ipynb   --> trains autoencoder, runs UMAP/PCA
03_counterfactual_examples.ipynb --> validates optimizer on test cases
```

### 3. Start the server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in browser

| Page | URL |
|---|---|
| Landing | http://localhost:8000/ |
| Risk Dashboard | http://localhost:8000/analysis |
| Methodology | http://localhost:8000/methodology |
| About | http://localhost:8000/about |
| Swagger API Docs | http://localhost:8000/docs |

---

## 11. Dependencies

| Package | Role |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, Random Forest, metrics, Isotonic Regression |
| `torch` | Autoencoder + counterfactual optimizer (PyTorch) |
| `lifelines` | Cox Proportional Hazards survival model |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `joblib` | Model serialization |
| `matplotlib`, `plotly` | Plotting (notebooks) |
| `statsmodels` | Statistical utilities |
| `umap-learn` | UMAP dimensionality reduction |
| `kneed` | Kneedle elbow algorithm for dimension selection |
| `nbformat`, `jupyter` | Notebook generation & execution |
| `fastapi` | Web framework (backend) |
| `uvicorn` | ASGI server |
| `jinja2` | HTML templating |
| `python-multipart` | File upload support |

---

## Research Notes

- **Immutable vs. Actionable split** is central to the ethical design — the optimizer never recommends changing someone''s sex, past diagnoses, or education level.
- **Cox coefficients as XAI** provides local additive feature attributions: `contribution_i = beta_i x x_i`, shown as a ranked horizontal bar chart.
- **Ensemble voting** (XGB + LGB + RF at threshold 0.15) gives a secondary sanity check alongside the primary Cox hazard score.
- **3D Three.js particle network** on the landing page renders a dynamically animated graph reflecting the concept of a patient manifold.
- All model serialization uses `joblib` for sklearn/lifelines objects and `torch.save` for PyTorch `.pth` files.

---

*CardioXAI — Sem VII Major Project I*
