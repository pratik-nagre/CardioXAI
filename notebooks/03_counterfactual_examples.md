# Phase 4: Counterfactual Search Optimizer
In this notebook, we utilize the Cox Proportional Hazards model and the Latent Manifold Autoencoder to find biologically plausible counterfactual paths for high-risk patients. We use a PyTorch-native gradient descent optimizer operating in the latent space.



```python
import torch
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append('../src')
from optimizer import CounterfactualOptimizer
from data_pipeline import load_and_preprocess_data

# Load Data
print("Loading data...")
train_df, test_df, imm, dir_f, act = load_and_preprocess_data(filepath='../data/frmgham2.csv', random_seed=42)

# Load configuration
ae_config = joblib.load('../models/ae_config.joblib')
feature_cols = ae_config['feature_cols']

# Initialize Optimizer
opt = CounterfactualOptimizer(models_dir='../models')
scaler = opt.scaler
cont_cols = opt.cont_cols
cont_idx = opt.cont_idx
bin_cols = opt.bin_cols
bin_idx = opt.bin_idx

```

    Loading data...
    Loading data from ../data/frmgham2.csv...
    
    Missingness before imputation (%):
    TIMECHD       0.000000
    RANDID        0.000000
    time_years    0.000000
    BPMEDS        1.250000
    ANYCHD        0.000000
    SYSBP         0.000000
    GLUCOSE       9.150943
    PREVMI        0.000000
    PREVSTRK      0.000000
    PREVAP        0.000000
    AGE           0.000000
    DIABP         0.000000
    SEX           0.000000
    TOTCHOL       1.179245
    CIGPDAY       0.683962
    PREVHYP       0.000000
    PREVCHD       0.000000
    HEARTRTE      0.023585
    DIABETES      0.000000
    educ          2.476415
    BMI           0.448113
    CURSMOKE      0.000000
    dtype: float64
    
    --- Imputation Impact (Train Set) ---
    AGE: mean shift=0.00%, var shift=0.00%
    TOTCHOL: mean shift=0.01%, var shift=1.14%
    SYSBP: mean shift=0.00%, var shift=0.00%
    DIABP: mean shift=0.00%, var shift=0.00%
    CIGPDAY: mean shift=0.65%, var shift=0.28%
    BMI: mean shift=0.01%, var shift=0.47%
    HEARTRTE: mean shift=0.00%, var shift=0.03%
    GLUCOSE: mean shift=0.41%, var shift=9.11%
    SEX: mean shift=0.00%, var shift=0.00%
    PREVCHD: mean shift=0.00%, var shift=0.00%
    PREVAP: mean shift=0.00%, var shift=0.00%
    PREVMI: mean shift=0.00%, var shift=0.00%
    PREVSTRK: mean shift=0.00%, var shift=0.00%
    PREVHYP: mean shift=0.00%, var shift=0.00%
    CURSMOKE: mean shift=0.00%, var shift=0.00%
    DIABETES: mean shift=0.00%, var shift=0.00%
    BPMEDS: mean shift=1.27%, var shift=1.23%
    educ: mean shift=1.33%, var shift=0.28%
    
    --- Final Feature Groups ---
    IMMUTABLE: ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'educ_1.0', 'educ_2.0', 'educ_3.0', 'educ_4.0']
    DIRECTIONAL: ['AGE']
    ACTIONABLE: ['TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']
    

## 1. Selecting High-Risk Patients
We identify a subset of high-risk patients from the test cohort (e.g., those who experienced a cardiac event) to demonstrate the counterfactual search.



```python
# Filter to those who had an event in the test set
test_events = test_df[test_df['ANYCHD'] == 1].copy()

# Pick 3 original patients as case studies
sample_patients = test_events.head(3)

# Find a patient with very low SYSBP to test the hard bound
sysbp_scale_idx = cont_cols.index('SYSBP')
sysbp_thresh = (100 - scaler.mean_[sysbp_scale_idx]) / scaler.scale_[sysbp_scale_idx]
low_sysbp_patient = test_df[test_df['SYSBP'] < sysbp_thresh].head(1)

# Find a patient with low TOTCHOL to test the new monotonicity rule
totchol_scale_idx = cont_cols.index('TOTCHOL')
totchol_thresh = (140 - scaler.mean_[totchol_scale_idx]) / scaler.scale_[totchol_scale_idx]
low_totchol_patient = test_df[test_df['TOTCHOL'] < totchol_thresh].head(1)

# Combine them all
sample_patients = pd.concat([sample_patients, low_sysbp_patient, low_totchol_patient])
patient_vectors = sample_patients[feature_cols].values
print(f"Selected {len(patient_vectors)} high-risk patients for demonstration.")

```

    Selected 5 high-risk patients for demonstration.
    

## 2. Running the Counterfactual Search
For each patient, the optimizer minimizes the Cox log partial hazard while enforcing proximity, immutability, and hard clinical bounds.



```python
results = []

for i, p_vec in enumerate(patient_vectors):
    res = opt.optimize(p_vec)
    
    # We want to format the output nicely. Let's unscale the continuous features.
    x_orig_cont = res['x_orig'][cont_idx]
    x_cf_cont = res['x_cf'][cont_idx]
    
    # Inverse transform requires 2D array
    x_orig_native = scaler.inverse_transform(x_orig_cont.reshape(1, -1))[0]
    x_cf_native = scaler.inverse_transform(x_cf_cont.reshape(1, -1))[0]
    
    # Format changes
    changes = []
    
    # Continuous changes
    for feat, orig_val, cf_val in zip(cont_cols, x_orig_native, x_cf_native):
        diff = cf_val - orig_val
        if abs(diff) > 0.5: # Only show meaningful changes
            changes.append(f"{feat}: {orig_val:.1f} -> {cf_val:.1f} ({diff:+.1f})")
            
    # Binary changes
    x_orig_bin = res['x_orig'][bin_idx]
    x_cf_bin = np.round(res['x_cf'][bin_idx])
    for feat, orig_val, cf_val in zip(bin_cols, x_orig_bin, x_cf_bin):
        diff = cf_val - orig_val
        if diff != 0:
            changes.append(f"{feat}: {int(orig_val)} -> {int(cf_val)}")
            
    results.append({
        'Patient ID': f"Patient {i+1}",
        'Baseline Log Hazard': f"{res['baseline_hazard']:.3f}",
        'Post-Recourse Log Hazard': f"{res['final_hazard']:.3f}",
        'Hazard Drop': f"{res['hazard_drop']:.3f}",
        'Success': res['success'],
        'Suggested Interventions': " | ".join(changes) if (changes and res['success']) else "No feasible recourse found."
    })

pd.set_option('display.max_colwidth', None)
res_df = pd.DataFrame(results)
display(res_df)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient ID</th>
      <th>Baseline Log Hazard</th>
      <th>Post-Recourse Log Hazard</th>
      <th>Hazard Drop</th>
      <th>Success</th>
      <th>Suggested Interventions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Patient 1</td>
      <td>0.208</td>
      <td>0.153</td>
      <td>0.055</td>
      <td>True</td>
      <td>TOTCHOL: 241.0 -&gt; 237.2 (-3.8) | SYSBP: 126.5 -&gt; 118.0 (-8.5) | DIABP: 78.5 -&gt; 76.1 (-2.4) | CIGPDAY: 30.0 -&gt; 28.9 (-1.1) | BMI: 29.9 -&gt; 28.0 (-2.0) | HEARTRTE: 85.0 -&gt; 82.4 (-2.6)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patient 2</td>
      <td>0.445</td>
      <td>0.975</td>
      <td>-0.530</td>
      <td>False</td>
      <td>No feasible recourse found.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Patient 3</td>
      <td>0.363</td>
      <td>0.296</td>
      <td>0.067</td>
      <td>True</td>
      <td>TOTCHOL: 272.0 -&gt; 266.4 (-5.6) | SYSBP: 127.5 -&gt; 121.3 (-6.2) | DIABP: 79.5 -&gt; 75.3 (-4.2) | BMI: 26.4 -&gt; 24.0 (-2.3) | HEARTRTE: 78.0 -&gt; 73.9 (-4.1)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Patient 4</td>
      <td>-0.835</td>
      <td>-0.927</td>
      <td>0.092</td>
      <td>True</td>
      <td>SYSBP: 98.0 -&gt; 104.4 (+6.4) | DIABP: 75.0 -&gt; 69.0 (-6.0) | BMI: 26.6 -&gt; 25.3 (-1.3) | HEARTRTE: 70.0 -&gt; 68.6 (-1.4) | GLUCOSE: 78.0 -&gt; 72.1 (-5.9)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Patient 5</td>
      <td>-0.643</td>
      <td>-0.728</td>
      <td>0.085</td>
      <td>True</td>
      <td>SYSBP: 117.0 -&gt; 112.3 (-4.7) | DIABP: 78.5 -&gt; 72.1 (-6.4) | BMI: 26.4 -&gt; 24.3 (-2.1) | HEARTRTE: 75.0 -&gt; 73.8 (-1.2) | GLUCOSE: 78.0 -&gt; 73.0 (-5.0)</td>
    </tr>
  </tbody>
</table>
</div>

