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
        self.imm_cols = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'DIABETES'] + self.cat_cols
        self.imm_idx = [self.ae_feature_cols.index(c) for c in self.imm_cols if c in self.ae_feature_cols]
        
        self.decrease_only_cols = []
        self.increase_only_cols = []
        
        for col in self.ae_feature_cols:
            if col in self.imm_cols or col not in self.cox_feature_cols:
                continue
                
            # Manual Overrides for Clinical/Biological Safety
            if col == 'BPMEDS':
                # Confounding by indication: never recommend stopping BP meds
                self.increase_only_cols.append(col)
                continue
            elif col == 'HEARTRTE':
                # Confounding by frailty: Cox coef is negative, but clinically shouldn't recommend raising HR
                self.decrease_only_cols.append(col)
                continue
            elif col == 'AGE':
                # Biological reality: cannot get younger. 
                # For single-point counterfactual recommendations, we anchor it symmetrically to prevent upward manifold drift.
                self.imm_cols.append(col)
                self.imm_idx.append(self.ae_feature_cols.index(col))
                continue
                
            # Programmatic rule: enforce movement only in the hazard-reducing direction
            coef = self.cph_model.params_[col]
            if coef > 0:
                self.decrease_only_cols.append(col)
            elif coef < 0:
                self.increase_only_cols.append(col)
                
        self.decrease_only_idx = [self.ae_feature_cols.index(c) for c in self.decrease_only_cols]
        self.increase_only_idx = [self.ae_feature_cols.index(c) for c in self.increase_only_cols]
        
        

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
        
        # Setup patient-specific relative bounds
        rel_caps = {
            'SYSBP': 0.20,
            'DIABP': 0.20,
            'TOTCHOL': 0.20,
            'GLUCOSE': 0.20,
            'BMI': 0.20,
            'HEARTRTE': 0.15,
            'CIGPDAY': 0.50
        }
        abs_floors = {
            'SYSBP': 90.0,
            'DIABP': 60.0,
            'TOTCHOL': 120.0,
            'GLUCOSE': 65.0,
            'HEARTRTE': 50.0,
            'BMI': 18.5,
            'CIGPDAY': 0.0
        }
        
        self.patient_mins = {}
        for feat, cap in rel_caps.items():
            if feat not in self.ae_feature_cols:
                continue
            f_idx = self.ae_feature_cols.index(feat)
            c_idx = self.cont_cols.index(feat)
            
            orig_scaled = x_0[0, f_idx].item()
            orig_native = (orig_scaled * self.scaler.scale_[c_idx]) + self.scaler.mean_[c_idx]
            
            # Compute minimum allowed value (relative cap vs absolute floor)
            rel_min = orig_native * (1.0 - cap)
            final_native_min = max(abs_floors.get(feat, 0.0), rel_min)
            final_scaled_min = (final_native_min - self.scaler.mean_[c_idx]) / self.scaler.scale_[c_idx]
            
            self.patient_mins[f_idx] = final_scaled_min
        
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
        min_risk_drop = 0.01 # Relaxed to allow plans for high-risk patients
        
        trajectory = []
        sysbp_idx = self.ae_feature_cols.index('SYSBP')
        bmi_idx = self.ae_feature_cols.index('BMI')
        chol_idx = self.ae_feature_cols.index('TOTCHOL')
        
        sysbp_c_idx = self.cont_cols.index('SYSBP')
        bmi_c_idx = self.cont_cols.index('BMI')
        chol_c_idx = self.cont_cols.index('TOTCHOL')
        
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
            
            # 4. Directional Constraints (Monotonicity)
            loss_dir = 0.0
            
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
            for f_idx, min_scaled in self.patient_mins.items():
                loss_bound += torch.relu(min_scaled - x_hat[0, f_idx])**2 * lambda_bound
            
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
                
            if step % 10 == 0 or step == max_steps - 1:
                with torch.no_grad():
                    s_val = (x_hat[0, sysbp_idx].item() * self.scaler.scale_[sysbp_c_idx]) + self.scaler.mean_[sysbp_c_idx]
                    b_val = (x_hat[0, bmi_idx].item() * self.scaler.scale_[bmi_c_idx]) + self.scaler.mean_[bmi_c_idx]
                    c_val = (x_hat[0, chol_idx].item() * self.scaler.scale_[chol_c_idx]) + self.scaler.mean_[chol_c_idx]
                    trajectory.append({'step': step, 'SYSBP': float(s_val), 'BMI': float(b_val), 'TOTCHOL': float(c_val)})
                
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
            for f_idx, min_scaled in self.patient_mins.items():
                x_cf_final[0, f_idx] = torch.max(x_cf_final[0, f_idx], torch.tensor(min_scaled, device=self.device))
            
            # 3. Force directional rules strictly (has final say over hard bounds)
            for s_idx in self.decrease_only_idx:
                x_cf_final[0, s_idx] = torch.min(x_cf_final[0, s_idx], x_0[0, s_idx])
            for i_idx in self.increase_only_idx:
                x_cf_final[0, i_idx] = torch.max(x_cf_final[0, i_idx], x_0[0, i_idx])
            
            # Final hazard check
            cox_x_cf = x_cf_final[0, self.ae_to_cox_indices]
            final_hazard = torch.dot(cox_x_cf, self.cox_betas).item()
            
            if final_hazard >= baseline_hazard:
                x_cf_final = x_0.clone()
                final_hazard = baseline_hazard
            
        log_hazard_drop = baseline_hazard - final_hazard
        
        # 3-Tier Success Criteria
        if log_hazard_drop >= 0.05:
            status = 'strong'
        elif log_hazard_drop >= 0.01:
            status = 'partial'
        else:
            status = 'none'
            
        # Calculate Relative Risk Reduction (RRR)
        relative_risk_reduction = 1.0 - np.exp(-log_hazard_drop)
        
        return {
            'success': status != 'none', # Kept for backward compatibility, but UI should use status
            'status': status,
            'baseline_hazard': baseline_hazard,
            'final_hazard': final_hazard,
            'log_hazard_drop': log_hazard_drop,
            'relative_risk_reduction': relative_risk_reduction,
            'x_orig': x_0[0].cpu().numpy(),
            'x_cf': x_cf_final[0].cpu().numpy(),
            'trajectory': trajectory
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
