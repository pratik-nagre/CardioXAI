import os
import sys
import pandas as pd
import numpy as np
import joblib

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_markdown_cell(content):
    return new_markdown_cell(content)

def create_code_cell(content):
    return new_code_cell(content)

def build_notebook():
    nb = new_notebook()
    cells = []
    
    cells.append(create_markdown_cell("""\
# Phase 4: Counterfactual Search Optimizer
In this notebook, we utilize the Cox Proportional Hazards model and the Latent Manifold Autoencoder to find biologically plausible counterfactual paths for high-risk patients. We use a PyTorch-native gradient descent optimizer operating in the latent space.
"""))

    cells.append(create_code_cell("""\
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
"""))

    cells.append(create_markdown_cell("""\
## 1. Selecting High-Risk Patients
We identify a subset of high-risk patients from the test cohort (e.g., those who experienced a cardiac event) to demonstrate the counterfactual search.
"""))

    cells.append(create_code_cell("""\
# Filter to those who had an event in the test set
test_events = test_df[test_df['ANYCHD'] == 1].copy()

# Pick 10 original patients as case studies (plus the 2 edge cases below makes 12)
sample_patients = test_events.head(10)

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
"""))

    cells.append(create_markdown_cell("""\
## 2. Running the Counterfactual Search
For each patient, the optimizer minimizes the Cox log partial hazard while enforcing proximity, immutability, and hard clinical bounds.
"""))

    cells.append(create_code_cell("""\
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
            
    # Confusion Matrix Logic
    old_success = res['log_hazard_drop'] >= 0.05
    new_tier = res['status']
    
    confusion_status = "MATCH"
    if old_success and new_tier == 'strong':
        confusion_status = "MATCH (Old Success -> New Strong)"
    elif old_success and new_tier != 'strong':
        confusion_status = f"DEMOTED (Old Success -> {new_tier.upper()})"
    elif not old_success and new_tier == 'strong':
        confusion_status = "PROMOTED (Old Fail -> New Strong)"
    elif not old_success and new_tier == 'partial':
        confusion_status = "PROMOTED (Old Fail -> New Partial)"
    elif not old_success and new_tier == 'none':
        confusion_status = "MATCH (Old Fail -> New None)"
        
    results.append({
        'Patient ID': f"Patient {i+1}",
        'Baseline Hazard': f"{res['baseline_hazard']:.3f}",
        'Post-Recourse Hazard': f"{res['final_hazard']:.3f}",
        'Hazard Drop': f"{res['log_hazard_drop']:.3f}",
        'Relative Risk Drop': f"{res['relative_risk_reduction']*100:.1f}%",
        'Old Success (>=0.05)': old_success,
        'New Tier Status': new_tier.upper(),
        'Confusion Check': confusion_status,
        'Suggested Interventions': " | ".join(changes) if (changes and res['status'] != 'none') else "No feasible recourse."
    })

pd.set_option('display.max_colwidth', None)
res_df = pd.DataFrame(results)
display(res_df)

# Print summary confusion matrix
print("--- 3-Way Confusion Check Summary ---")
print(res_df['Confusion Check'].value_counts())
"""))

    nb['cells'] = cells
    with open('03_counterfactual_examples.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook '03_counterfactual_examples.ipynb' successfully generated.")

if __name__ == '__main__':
    build_notebook()
