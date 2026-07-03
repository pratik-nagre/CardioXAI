import sys
sys.path.append('../src')
from optimizer import CounterfactualOptimizer
from data_pipeline import load_and_preprocess_data
import joblib

print("Loading data...")
train_df, test_df, imm, dir_f, act = load_and_preprocess_data(filepath='../data/frmgham2.csv', random_seed=42)
test_events = test_df[test_df['ANYCHD'] == 1].copy()
ae_config = joblib.load('../models/ae_config.joblib')
feature_cols = ae_config['feature_cols']

patient_vectors = test_events.head(3)[feature_cols].values
opt = CounterfactualOptimizer(models_dir='../models')
scaler = opt.scaler

for i, p_vec in enumerate(patient_vectors):
    res = opt.optimize(p_vec)
    x_orig = scaler.inverse_transform(res['x_orig'][opt.cont_idx].reshape(1, -1))[0]
    x_cf = scaler.inverse_transform(res['x_cf'][opt.cont_idx].reshape(1, -1))[0]
    changes = []
    for feat, orig, cf in zip(opt.cont_cols, x_orig, x_cf):
        if abs(cf - orig) > 0.5:
            changes.append(f"{feat}: {orig:.1f} -> {cf:.1f}")
    print(f"Patient {i+1}: {' | '.join(changes)}")
