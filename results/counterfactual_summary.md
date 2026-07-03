# Phase 4 (Layer 3) Summary: Counterfactual Optimizer

## Overview
The Counterfactual Optimizer is a PyTorch-native gradient descent engine that searches the Latent Autoencoder Manifold for biologically plausible, risk-reducing lifestyle interventions. It uses the Cox Proportional Hazards model's coefficients as the objective function to minimize.

## Structural Constraints
To prevent clinically implausible interventions caused by manifold entanglement, two classes of structural limits were implemented:

1. **Directional Cox Monotonicity**: Actionable features are programmatically locked to either `increase_only` or `decrease_only` strictly based on their Cox partial hazard coefficient sign (`cph_model.params_`). This guarantees the optimizer will never exploit a paradoxical manifold correlation (e.g., raising SYSBP) to lower the calculated risk.
2. **Relative Magnitude Caps**: A generalized relative-magnitude barrier penalty restricts how much any single continuous feature can change. Features are capped relative to the patient's baseline (e.g., `SYSBP` ±20%, `HEARTRTE` ±15%, `BMI` ±20%, `CIGPDAY` ±50%) preventing the manifold from passively pulling a feature to absurd extremes.

## Advanced Guardrails
- **Immutable Anchoring**: Diagnostic markers (e.g., `DIABETES`, `PREVCHD`) and demographics (`SEX`, `educ`) are heavily penalized from deviating via an $L2$ anchor, ensuring recommendations only target actionable lifestyle factors.
- **Honest Fallbacks**: If a patient is deemed un-salvageable under all clinical bounds (the generated vector fails to lower the baseline hazard after post-hoc sanitization), the system safely aborts and returns `No feasible recourse found`.

## Conclusion
The optimizer cleanly resolves manifold-induced hallucinations, delivering safe, modest, and actionable recommendations for high-risk patients.

---

## Final Constraint Specification

The table below serves as the single source of truth for the structural constraints applied within the Latent Autoencoder Counterfactual Search. It documents how the Cox Proportional Hazards model parameters were translated into programmatic constraints, the patient-relative magnitude bounds instituted to prevent manifold hallucinations, and the specific clinical overrides required for biological safety.

| Feature | Actionable? | Cox Coefficient | Enforced Direction | Magnitude Cap | Absolute Floor | Manual Override Justification |
|---------|-------------|-----------------|--------------------|---------------|----------------|-------------------------------|
| `TOTCHOL` | Yes | +0.182 | Decrease-only | 20% | 120.0 | - |
| `SYSBP` | Yes | +0.170 | Decrease-only | 20% | 90.0 | - |
| `BMI` | Yes | +0.108 | Decrease-only | 20% | 18.5 | - |
| `GLUCOSE` | Yes | +0.068 | Decrease-only | 20% | 65.0 | - |
| `CIGPDAY` | Yes | +0.063 | Decrease-only | 50% | 0.0 | - |
| `DIABP` | Yes | +0.059 | Decrease-only | 20% | 60.0 | - |
| `CURSMOKE`| Yes | +0.077 | Decrease-only | Binary (1 &rarr; 0) | 0.0 | - |
| `HEARTRTE`| Yes | -0.056 | **Decrease-only** | 15% | 50.0 | **Override**: The Autoencoder manifold creates a frailty confound (correlating lower HR with lower risk factors), while Cox natively assigns a negative coefficient. Clinically, patients should not be advised to *raise* resting heart rate to lower cardiovascular risk. |
| `BPMEDS` | Yes | +0.250 | **Increase-only** | Binary (0 &rarr; 1) | N/A | **Override**: The Cox model assigns a positive risk coefficient to BP meds (due to correlation with hypertension). Patients cannot be advised to *stop* medication to artificially lower their hazard score. |
| `DIABETES`| No | +0.572 | **Immutable** | N/A | N/A | **Override**: Diabetes is a permanent diagnosis. It cannot be "cured" or reversed as a point-in-time lifestyle recommendation. |
| `AGE` | No | +0.183 | **Immutable** | N/A | N/A | **Override**: While age strictly increases in reality, for a point-in-time counterfactual recommendation it must be locked symmetrically to prevent the AE manifold from artificially drifting age up or down. |

*Note: All demographic features (`SEX`, `educ`) and prior medical event indicators (`PREVSTRK`, `PREVHYP`, `PREVMI`, `PREVAP`, `PREVCHD`) are strictly treated as **Immutable** anchor points for the duration of the latent search.*
