import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from data_pipeline import load_and_preprocess_data
except ImportError:
    from src.data_pipeline import load_and_preprocess_data

class SplitHeadAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, cont_idx, bin_idx, cat_idx):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.cont_idx = cont_idx
        self.bin_idx = bin_idx
        self.cat_idx = cat_idx
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, latent_dim)
        )
        
        self.decoder_body = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        # Split heads based on variable types
        self.head_cont = nn.Linear(16, len(cont_idx))
        self.head_bin = nn.Linear(16, len(bin_idx))
        self.head_cat = nn.Linear(16, len(cat_idx))

    def forward(self, x):
        latent = self.encoder(x)
        dec_shared = self.decoder_body(latent)
        
        out_cont = self.head_cont(dec_shared)
        out_bin = self.head_bin(dec_shared)
        out_cat = self.head_cat(dec_shared)
        
        return latent, out_cont, out_bin, out_cat

def train_autoencoder(X_train, X_val, feature_cols, latent_dim=8, patience=15, epochs=150, lr=1e-3, random_seed=42):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        
    # Determine indices for different feature types
    cont_cols = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
    bin_cols = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'CURSMOKE', 'DIABETES', 'BPMEDS']
    cat_cols = [c for c in feature_cols if c.startswith('educ_')]
    
    # Filter only those present in feature_cols
    cont_cols = [c for c in cont_cols if c in feature_cols]
    bin_cols = [c for c in bin_cols if c in feature_cols]
    
    cont_idx = [feature_cols.index(c) for c in cont_cols]
    bin_idx = [feature_cols.index(c) for c in bin_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    
    # Pre-flight check for SEX
    if 'SEX' in feature_cols:
        sex_col_idx = feature_cols.index('SEX')
        unique_sex = set(np.unique(X_train[:, sex_col_idx]))
        assert unique_sex.issubset({0.0, 1.0}), f"SEX column contains invalid values for BCE Loss: {unique_sex}"

    ae = SplitHeadAutoencoder(len(feature_cols), latent_dim, cont_idx, bin_idx, cat_idx)
    
    # Loss Functions
    criterion_cont = nn.MSELoss()
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_cat = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-4)
    
    # DataLoaders
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    val_metrics = {}
    
    for epoch in range(epochs):
        ae.train()
        for (X_batch,) in train_loader:
            optimizer.zero_grad()
            
            # Split targets
            t_cont = X_batch[:, cont_idx]
            t_bin = X_batch[:, bin_idx]
            t_cat = X_batch[:, cat_idx]
            
            # For cross entropy, we need class index (argmax over one-hot columns)
            t_cat_idx = torch.argmax(t_cat, dim=1)
            
            _, out_cont, out_bin, out_cat = ae(X_batch)
            
            # Normalized loss computation
            loss_cont = criterion_cont(out_cont, t_cont)
            loss_bin = criterion_bin(out_bin, t_bin)
            loss_cat = criterion_cat(out_cat, t_cat_idx)
            
            loss = loss_cont + loss_bin + loss_cat
            loss.backward()
            optimizer.step()
            
        # Validation
        ae.eval()
        with torch.no_grad():
            t_cont_v = X_val_t[:, cont_idx]
            t_bin_v = X_val_t[:, bin_idx]
            t_cat_v = X_val_t[:, cat_idx]
            t_cat_idx_v = torch.argmax(t_cat_v, dim=1)
            
            _, o_cont_v, o_bin_v, o_cat_v = ae(X_val_t)
            
            v_loss_cont = criterion_cont(o_cont_v, t_cont_v).item()
            v_loss_bin = criterion_bin(o_bin_v, t_bin_v).item()
            v_loss_cat = criterion_cat(o_cat_v, t_cat_idx_v).item()
            
            v_loss = v_loss_cont + v_loss_bin + v_loss_cat
            
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = ae.state_dict().copy()
            patience_counter = 0
            
            # Record component metrics for the best epoch
            pred_bin_classes = (torch.sigmoid(o_bin_v) > 0.5).float()
            bin_acc = (pred_bin_classes == t_bin_v).float().mean().item()
            
            pred_cat_classes = torch.argmax(o_cat_v, dim=1)
            cat_acc = (pred_cat_classes == t_cat_idx_v).float().mean().item()
            
            val_metrics = {
                'Total Loss': v_loss,
                'Cont MSE': v_loss_cont,
                'Bin BCE': v_loss_bin,
                'Cat CE': v_loss_cat,
                'Bin Acc': bin_acc,
                'Cat Acc': cat_acc
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    ae.load_state_dict(best_state)
    return ae, val_metrics

if __name__ == '__main__':
    # Default testing execution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'frmgham2.csv')
    
    print("Testing Autoencoder Architecture & Training Loop...")
    train_df, test_df, imm, dir_f, act = load_and_preprocess_data(filepath=data_path, random_seed=42)
    feature_cols = imm + dir_f + act
    
    # 80/20 train/val split without censoring filter
    X_train, X_val = train_test_split(train_df[feature_cols].values, test_size=0.2, random_state=42)
    
    print(f"Training shapes -> Train: {X_train.shape}, Val: {X_val.shape}")
    
    model, metrics = train_autoencoder(X_train, X_val, feature_cols, latent_dim=8)
    
    print("\\nValidation Metrics for latent_dim=8:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(models_dir, 'autoencoder_dim8.pth'))
    joblib.dump({'latent_dim': 8, 'feature_cols': feature_cols}, os.path.join(models_dir, 'ae_config.joblib'))
    
    print(f"\\nSaved test autoencoder to {models_dir}")
