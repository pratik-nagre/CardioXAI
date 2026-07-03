import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import sys

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from autoencoder import SplitHeadAutoencoder
from data_pipeline import load_and_preprocess_data

class CounterfactualOptimizer:
    def __init__(self, models_dir='../models'):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        
        # Load AE Config & Model
        ae_config = joblib.load(os.path.join(models_dir, 'ae_config.joblib'))
        self.ae_feature_cols = ae_config['feature_cols']
        self.latent_dim = ae_config['latent_dim']
        
        # Get indices for feature types to initialize AE correctly
        self.cont_cols = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
        self.bin_cols = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'CURSMOKE', 'DIABETES', 'BPMEDS']
        self.cat_cols = [c for c in self.ae_feature_cols if c.startswith('educ_')]
        
        # Filter for actuals present in config
        self.cont_cols = [c for c in self.cont_cols if c in self.ae_feature_cols]
        self.bin_cols = [c for c in self.bin_cols if c in self.ae_feature_cols]
        
        self.cont_idx = [self.ae_feature_cols.index(c) for c in self.cont_cols]
        self.bin_idx = [self.ae_feature_cols.index(c) for c in self.bin_cols]
        self.cat_idx = [self.ae_feature_cols.index(c) for c in self.cat_cols]
        
        self.ae = SplitHeadAutoencoder(len(self.ae_feature_cols), self.latent_dim, self.cont_idx, self.bin_idx, self.cat_idx)
        self.ae.load_state_dict(torch.load(os.path.join(models_dir, 'autoencoder.pth'), map_location=self.device))
        self.ae.to(self.device)
        self.ae.eval() # AE weights are frozen
        
        # Load Cox Model
        self.cph_model = joblib.load(os.path.join(models_dir, 'cox_model.joblib'))
        self.cox_feature_cols = joblib.load(os.path.join(models_dir, 'cox_feature_cols.joblib'))
        
        # Extract Cox Betas as a tensor, strictly ordered by cox_feature_cols
        betas = self.cph_model.params_[self.cox_feature_cols].values
        self.cox_betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        
        # Mapping AE output indices to Cox input indices
        # We need a slicing array to pluck and reorder AE outputs to match Cox inputs
        self.ae_to_cox_indices = [self.ae_feature_cols.index(c) for c in self.cox_feature_cols]
        
        # Categorize indices for penalties
        self.imm_cols = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP'] + self.cat_cols
        self.imm_idx = [self.ae_feature_cols.index(c) for c in self.imm_cols if c in self.ae_feature_cols]
        
        self.age_idx = self.ae_feature_cols.index('AGE') if 'AGE' in self.ae_feature_cols else -1
        
        # Smoker paradox and frailty confounding features (can only decrease)
        self.decrease_only_cols = ['CIGPDAY', 'CURSMOKE', 'TOTCHOL']
        self.decrease_only_idx = [self.ae_feature_cols.index(c) for c in self.decrease_only_cols if c in self.ae_feature_cols]
        
        # Medication confounding features (can only increase)
        self.increase_only_cols = ['BPMEDS']
        self.increase_only_idx = [self.ae_feature_cols.index(c) for c in self.increase_only_cols if c in self.ae_feature_cols]
        
        # Define Actionable Hard Bounds (in scaled space)
        # We must transform physiological limits to scaled space.
        def scale_val(feat, val):
            idx = self.cont_cols.index(feat)
            return (val - self.scaler.mean_[idx]) / self.scaler.scale_[idx]
            
        self.sysbp_min_scaled = scale_val('SYSBP', 90.0)
        self.diabp_min_scaled = scale_val('DIABP', 60.0)
        self.totchol_min_scaled = scale_val('TOTCHOL', 120.0)
        self.glucose_min_scaled = scale_val('GLUCOSE', 65.0)
        self.hr_min_scaled = scale_val('HEARTRTE', 50.0)
        self.bmi_abs_min_scaled = scale_val('BMI', 18.5)
        self.cigpday_min_scaled = scale_val('CIGPDAY', 0.0)
        
        self.sysbp_idx = self.ae_feature_cols.index('SYSBP')
        self.diabp_idx = self.ae_feature_cols.index('DIABP')
        self.totchol_idx = self.ae_feature_cols.index('TOTCHOL')
        self.glucose_idx = self.ae_feature_cols.index('GLUCOSE')
        self.hr_idx = self.ae_feature_cols.index('HEARTRTE')
        self.bmi_idx = self.ae_feature_cols.index('BMI')
        self.bmi_scale_idx = self.cont_cols.index('BMI')

    def construct_decoded_vector(self, out_cont, out_bin, out_cat):
        """Reassembles the split heads into a single 21-dim feature vector matching ae_feature_cols order."""
        batch_size = out_cont.shape[0]
        full_x = torch.zeros((batch_size, len(self.ae_feature_cols)), device=self.device)
        
        full_x[:, self.cont_idx] = out_cont
        full_x[:, self.bin_idx] = torch.sigmoid(out_bin) # Convert logits to probabilities
        
        # For categorical, we use softmax probabilities to maintain differentiability
        if len(self.cat_idx) > 0:
            full_x[:, self.cat_idx] = torch.softmax(out_cat, dim=1)
            
        return full_x

    def optimize(self, x_original_np, max_steps=300, lr=0.05, 
                 lambda_risk=1.0, lambda_prox=5.0, lambda_imm=100.0, 
                 lambda_dir=50.0, lambda_bound=50.0):
        
        x_0 = torch.tensor(x_original_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Find starting latent anchor
        with torch.no_grad():
            z_0, _, _, _ = self.ae(x_0)
            
        # z is the variable we optimize
        z = z_0.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)
        
        # Setup patient-specific relative bounds (e.g., BMI drop max 20%)
        bmi_native_orig = (x_0[0, self.bmi_idx].item() * self.scaler.scale_[self.bmi_scale_idx]) + self.scaler.mean_[self.bmi_scale_idx]
        bmi_native_min = max(18.5, bmi_native_orig * 0.80) # Max 20% drop, but never below 18.5
        bmi_min_scaled = (bmi_native_min - self.scaler.mean_[self.bmi_scale_idx]) / self.scaler.scale_[self.bmi_scale_idx]
        
        best_z = z_0.clone().detach()
        best_loss = float('inf')
        
        # Baseline hazard
        with torch.no_grad():
            _, o_cont, o_bin, o_cat = self.ae(x_0) # use AE reconstruction as baseline for fair comparison
            x_0_hat = self.construct_decoded_vector(o_cont, o_bin, o_cat)
            cox_x_0 = x_0_hat[0, self.ae_to_cox_indices]
            baseline_hazard = torch.dot(cox_x_0, self.cox_betas).item()
            
        # Optimization Loop
        patience = 30
        plateau_counter = 0
        min_risk_drop = 0.05 # log hazard must drop by at least 0.05
        
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Decode
            dec_shared = self.ae.decoder_body(z)
            out_cont = self.ae.head_cont(dec_shared)
            out_bin = self.ae.head_bin(dec_shared)
            out_cat = self.ae.head_cat(dec_shared)
            
            x_hat = self.construct_decoded_vector(out_cont, out_bin, out_cat)
            
            # 1. Risk Penalty
            cox_x_hat = x_hat[0, self.ae_to_cox_indices]
            hazard = torch.dot(cox_x_hat, self.cox_betas)
            loss_risk = hazard * lambda_risk
            
            # 2. Proximity Penalty
            loss_prox = torch.sum((z - z_0)**2) * lambda_prox
            
            # 3. Immutable Penalty (Anchor exactly to x_0's immutable values)
            loss_imm = torch.sum((x_hat[0, self.imm_idx] - x_0[0, self.imm_idx])**2) * lambda_imm
            
            # 4. Directional & Monotonic Penalties
            loss_dir = 0.0
            if self.age_idx != -1:
                # AGE cannot decrease -> penalize if x_hat < x_0
                age_diff = x_0[0, self.age_idx] - x_hat[0, self.age_idx]
                loss_dir += torch.relu(age_diff)**2 * lambda_dir
                
            for s_idx in self.decrease_only_idx:
                # CIGPDAY/CURSMOKE/TOTCHOL cannot increase -> penalize if x_hat > x_0
                diff = x_hat[0, s_idx] - x_0[0, s_idx]
                loss_dir += torch.relu(diff)**2 * lambda_dir
                
            for i_idx in self.increase_only_idx:
                # BPMEDS cannot decrease -> penalize if x_hat < x_0
                diff = x_0[0, i_idx] - x_hat[0, i_idx]
                loss_dir += torch.relu(diff)**2 * lambda_dir
                
            # 5. Barrier Penalties (Squared Hinge)
            loss_bound = 0.0
            loss_bound += torch.relu(self.sysbp_min_scaled - x_hat[0, self.sysbp_idx])**2 * lambda_bound
            loss_bound += torch.relu(self.diabp_min_scaled - x_hat[0, self.diabp_idx])**2 * lambda_bound
            loss_bound += torch.relu(self.totchol_min_scaled - x_hat[0, self.totchol_idx])**2 * lambda_bound
            loss_bound += torch.relu(self.glucose_min_scaled - x_hat[0, self.glucose_idx])**2 * lambda_bound
            loss_bound += torch.relu(self.hr_min_scaled - x_hat[0, self.hr_idx])**2 * lambda_bound
            loss_bound += torch.relu(bmi_min_scaled - x_hat[0, self.bmi_idx])**2 * lambda_bound
            
            cigpday_idx = self.ae_feature_cols.index('CIGPDAY') if 'CIGPDAY' in self.ae_feature_cols else -1
            if cigpday_idx != -1:
                loss_bound += torch.relu(self.cigpday_min_scaled - x_hat[0, cigpday_idx])**2 * lambda_bound
            
            # Total Loss
            loss = loss_risk + loss_prox + loss_imm + loss_dir + loss_bound
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone().detach()
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            if plateau_counter >= patience:
                break
                
        # Post-hoc evaluation
        with torch.no_grad():
            dec_shared = self.ae.decoder_body(best_z)
            out_cont = self.ae.head_cont(dec_shared)
            out_bin = self.ae.head_bin(dec_shared)
            out_cat = self.ae.head_cat(dec_shared)
            x_cf_final = self.construct_decoded_vector(out_cont, out_bin, out_cat)
            
            # Strict Post-Hoc Sanitization (Clipping)
            # 1. Force immutables back to absolute original
            x_cf_final[0, self.imm_idx] = x_0[0, self.imm_idx]
            
            # 2. Apply Hard bounds first
            x_cf_final[0, self.sysbp_idx] = torch.max(x_cf_final[0, self.sysbp_idx], torch.tensor(self.sysbp_min_scaled, device=self.device))
            x_cf_final[0, self.diabp_idx] = torch.max(x_cf_final[0, self.diabp_idx], torch.tensor(self.diabp_min_scaled, device=self.device))
            x_cf_final[0, self.totchol_idx] = torch.max(x_cf_final[0, self.totchol_idx], torch.tensor(self.totchol_min_scaled, device=self.device))
            x_cf_final[0, self.glucose_idx] = torch.max(x_cf_final[0, self.glucose_idx], torch.tensor(self.glucose_min_scaled, device=self.device))
            x_cf_final[0, self.hr_idx] = torch.max(x_cf_final[0, self.hr_idx], torch.tensor(self.hr_min_scaled, device=self.device))
            x_cf_final[0, self.bmi_idx] = torch.max(x_cf_final[0, self.bmi_idx], torch.tensor(bmi_min_scaled, device=self.device))
            
            if cigpday_idx != -1:
                x_cf_final[0, cigpday_idx] = torch.max(x_cf_final[0, cigpday_idx], torch.tensor(self.cigpday_min_scaled, device=self.device))
            
            # 3. Force directional rules strictly (has final say over hard bounds)
            if self.age_idx != -1:
                x_cf_final[0, self.age_idx] = torch.max(x_cf_final[0, self.age_idx], x_0[0, self.age_idx])
            for s_idx in self.decrease_only_idx:
                x_cf_final[0, s_idx] = torch.min(x_cf_final[0, s_idx], x_0[0, s_idx])
            for i_idx in self.increase_only_idx:
                x_cf_final[0, i_idx] = torch.max(x_cf_final[0, i_idx], x_0[0, i_idx])
            
            # Final hazard check
            cox_x_cf = x_cf_final[0, self.ae_to_cox_indices]
            final_hazard = torch.dot(cox_x_cf, self.cox_betas).item()
            
        success = (baseline_hazard - final_hazard) > min_risk_drop
        
        return {
            'success': success,
            'baseline_hazard': baseline_hazard,
            'final_hazard': final_hazard,
            'hazard_drop': baseline_hazard - final_hazard,
            'x_cf': x_cf_final[0].cpu().numpy(),
            'x_orig': x_0[0].cpu().numpy()
        }

if __name__ == '__main__':
    print("Testing Counterfactual Optimizer...")
    
    # 1. Load some test data
    train_df, test_df, imm, dir_f, act = load_and_preprocess_data(filepath='../data/frmgham2.csv', random_seed=42)
    ae_config = joblib.load('../models/ae_config.joblib')
    feature_cols = ae_config['feature_cols']
    
    # Take a high risk patient (e.g. index 0 from test set may or may not be high risk, let's grab one with a cardiac event)
    test_events = test_df[test_df['ANYCHD'] == 1]
    if len(test_events) == 0:
        test_events = test_df
    
    patient_high_risk = test_events[feature_cols].iloc[0].values
    
    # Take a low risk patient
    test_no_events = test_df[test_df['ANYCHD'] == 0]
    patient_low_risk = test_no_events[feature_cols].iloc[0].values
    
    opt = CounterfactualOptimizer()
    
    print("\\n--- Testing High Risk Patient ---")
    res_high = opt.optimize(patient_high_risk)
    print(f"Success: {res_high['success']}")
    print(f"Baseline Hazard: {res_high['baseline_hazard']:.4f}")
    print(f"Final Hazard: {res_high['final_hazard']:.4f}")
    
    print("\\n--- Testing Low Risk Patient ---")
    res_low = opt.optimize(patient_low_risk)
    print(f"Success: {res_low['success']}")
    print(f"Baseline Hazard: {res_low['baseline_hazard']:.4f}")
    print(f"Final Hazard: {res_low['final_hazard']:.4f}")
