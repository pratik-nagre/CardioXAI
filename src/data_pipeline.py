import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define base features per phase 1 requirements
BASE_IMMUTABLE_FEATURES = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP']
BASE_DIRECTIONAL_FEATURES = ['AGE']
BASE_ACTIONABLE_FEATURES = ['TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE']
# educ will be one-hot encoded, and we'll dynamically construct the final feature lists.

def load_and_preprocess_data(filepath="data/frmgham2.csv", random_seed=42):
    """
    Loads data, filters to PERIOD==1, handles missingness via train-set imputation,
    scales continuous features, one-hot encodes categoricals, and returns splits.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter to PERIOD == 1 (Baseline only)
    df = df[df['PERIOD'] == 1].copy()
    
    # Remap SEX from 1=Male/2=Female to 0=Male/1=Female for BCE Loss
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].map({1: 0, 2: 1})
        
    # Drop HDLC and LDLC
    if 'HDLC' in df.columns:
        df = df.drop(columns=['HDLC'])
    if 'LDLC' in df.columns:
        df = df.drop(columns=['LDLC'])
        
    # Convert TIMECHD to years
    if 'TIMECHD' in df.columns:
        df['time_years'] = df['TIMECHD'] / 365.25
        # Filter zero durations out
        df = df[df['time_years'] > 0].copy()
        
    # Features of interest (pre-encoding)
    outcomes = ['ANYCHD', 'time_years', 'RANDID', 'TIMECHD']
    available_outcomes = [c for c in outcomes if c in df.columns]
    
    all_needed_cols = list(set(BASE_IMMUTABLE_FEATURES + BASE_DIRECTIONAL_FEATURES + BASE_ACTIONABLE_FEATURES + ['educ'] + available_outcomes))
    df = df[all_needed_cols].copy()
    
    # Missingness report
    print("\nMissingness before imputation (%):")
    print(df.isnull().mean() * 100)
    
    # Train/test split FIRST to avoid imputation leakage
    # Stratify by ANYCHD
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=random_seed, stratify=df['ANYCHD'])
    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    
    # Assert no RANDID overlap
    train_rands = set(df_train['RANDID'])
    test_rands = set(df_test['RANDID'])
    assert len(train_rands.intersection(test_rands)) == 0, "Leakage detected: overlapping RANDID between train and test!"
    
    # Imputation Strategy: Compute on train, apply to train/test
    continuous_feats = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
    binary_feats = ['SEX', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'CURSMOKE', 'DIABETES', 'BPMEDS']
    categorical_feats = ['educ']
    
    imputation_dict = {}
    
    print("\n--- Imputation Impact (Train Set) ---")
    for col in continuous_feats:
        if col in df_train.columns:
            median_val = df_train[col].median()
            imputation_dict[col] = median_val
            
            # Log shift
            pre_mean, pre_var = df_train[col].mean(), df_train[col].var()
            df_train[col] = df_train[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
            post_mean, post_var = df_train[col].mean(), df_train[col].var()
            
            mean_shift = abs(post_mean - pre_mean) / (abs(pre_mean) + 1e-9)
            var_shift = abs(post_var - pre_var) / (abs(pre_var) + 1e-9)
            print(f"{col}: mean shift={mean_shift:.2%}, var shift={var_shift:.2%}")
            
    for col in binary_feats + categorical_feats:
        if col in df_train.columns:
            mode_val = df_train[col].mode()[0]
            imputation_dict[col] = mode_val
            
            pre_mean, pre_var = df_train[col].mean(), df_train[col].var()
            df_train[col] = df_train[col].fillna(mode_val)
            df_test[col] = df_test[col].fillna(mode_val)
            post_mean, post_var = df_train[col].mean(), df_train[col].var()
            mean_shift = abs(post_mean - pre_mean) / (abs(pre_mean) + 1e-9)
            var_shift = abs(post_var - pre_var) / (abs(pre_var) + 1e-9)
            print(f"{col}: mean shift={mean_shift:.2%}, var shift={var_shift:.2%}")
            
    # Save imputation dict
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(imputation_dict, os.path.join(models_dir, 'imputation_dict.joblib'))
    
    # One-hot encoding for educ
    df_train = pd.get_dummies(df_train, columns=['educ'], prefix='educ', drop_first=False, dtype=int)
    df_test = pd.get_dummies(df_test, columns=['educ'], prefix='educ', drop_first=False, dtype=int)
    
    # Ensure test has all columns that train has
    missing_cols = set(df_train.columns) - set(df_test.columns)
    for c in missing_cols:
        df_test[c] = 0
    df_test = df_test[df_train.columns]
    
    educ_encoded_cols = [c for c in df_train.columns if c.startswith('educ_')]
    
    # Dynamic feature lists
    global IMMUTABLE_FEATURES, DIRECTIONAL_FEATURES, ACTIONABLE_FEATURES
    IMMUTABLE_FEATURES = BASE_IMMUTABLE_FEATURES + educ_encoded_cols
    DIRECTIONAL_FEATURES = BASE_DIRECTIONAL_FEATURES
    ACTIONABLE_FEATURES = BASE_ACTIONABLE_FEATURES
    
    print("\n--- Final Feature Groups ---")
    print(f"IMMUTABLE: {IMMUTABLE_FEATURES}")
    print(f"DIRECTIONAL: {DIRECTIONAL_FEATURES}")
    print(f"ACTIONABLE: {ACTIONABLE_FEATURES}")
    
    # Scaling continuous features
    scaler = StandardScaler()
    df_train[continuous_feats] = scaler.fit_transform(df_train[continuous_feats])
    df_test[continuous_feats] = scaler.transform(df_test[continuous_feats])
    
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    return df_train, df_test, IMMUTABLE_FEATURES, DIRECTIONAL_FEATURES, ACTIONABLE_FEATURES

if __name__ == "__main__":
    train, test, imm, dir_f, act = load_and_preprocess_data()
    print(f"\nTrain shape: {train.shape}, Test shape: {test.shape}")
