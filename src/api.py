from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import torch
import os
import csv

from src.data_pipeline import load_and_preprocess_data
from src.optimizer import CounterfactualOptimizer

app = FastAPI(title="Cardiovascular Risk XAI Engine")

# Setup templates and static
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global models
models = {}

@app.on_event("startup")
def startup_event():
    print("Loading models and data...")
    train_X, test_X, IMMUTABLE, DIRECTIONAL, ACTIONABLE = load_and_preprocess_data(filepath="data/frmgham2.csv", random_seed=42)
    
    models['cph'] = joblib.load('models/cox_model.joblib')
    models['xgb'] = joblib.load('models/xgb_model.joblib')
    models['lgb'] = joblib.load('models/lgb_model.joblib')
    models['rf'] = joblib.load('models/rf_model.joblib')
    models['scaler'] = joblib.load('models/scaler.joblib')
    models['opt'] = CounterfactualOptimizer(models_dir='models')
    
    models['test_X'] = test_X
    models['cont_cols'] = list(models['scaler'].feature_names_in_)
    models['ACTIONABLE'] = ACTIONABLE
    print("Models loaded successfully.")

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(name="landing.html", request=request)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    return templates.TemplateResponse(name="analysis.html", request=request)

@app.get("/methodology", response_class=HTMLResponse)
async def methodology(request: Request):
    return templates.TemplateResponse(name="methodology.html", request=request)

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(name="about.html", request=request)

@app.get("/api/patients")
async def get_patients():
    # Return available patient IDs from test_X and custom_patients.csv
    base_patients = [f"Sample Patient {i}" for i in range(len(models['test_X']))]
    custom_patients = []
    
    custom_file = 'data/custom_patients.csv'
    if os.path.exists(custom_file):
        df_custom = pd.read_csv(custom_file)
        custom_patients = [f"Custom Patient {i}" for i in range(len(df_custom))]
        
    return {"base": base_patients, "custom": custom_patients}

@app.get("/api/patient/{p_type}/{idx}")
async def get_patient_data(p_type: str, idx: int):
    if p_type == "base":
        if idx < 0 or idx >= len(models['test_X']):
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_scaled = models['test_X'].iloc[idx].copy()
        patient_native = patient_scaled.copy()
        scaler = models['scaler']
        cont_cols = models['cont_cols']
        patient_native[cont_cols] = scaler.inverse_transform(patient_scaled[cont_cols].values.reshape(1, -1))[0]
        
        return patient_native.to_dict()
    elif p_type == "custom":
        custom_file = 'data/custom_patients.csv'
        if not os.path.exists(custom_file):
             raise HTTPException(status_code=404, detail="No custom patients found")
        df_custom = pd.read_csv(custom_file)
        if idx < 0 or idx >= len(df_custom):
            raise HTTPException(status_code=404, detail="Custom patient not found")
        
        return df_custom.iloc[idx].to_dict()
    else:
        raise HTTPException(status_code=400, detail="Invalid patient type")

class PredictionRequest(BaseModel):
    patient_data: dict

@app.post("/api/predict")
async def predict_risk(req: PredictionRequest):
    # incoming patient data is native, must scale it
    native_data = req.patient_data
    
    # Check if all cols are present
    required_cols = models['test_X'].columns
    for c in required_cols:
        if c not in native_data:
            native_data[c] = models['test_X'][c].median()
            
    # Scale continuous
    scaler = models['scaler']
    cont_cols = models['cont_cols']
    
    native_arr = np.array([float(native_data.get(c, 0.0)) for c in cont_cols]).reshape(1, -1)
    scaled_arr = scaler.transform(native_arr)[0]
    
    scaled_data = {}
    for c in required_cols:
        if c in cont_cols:
            scaled_data[c] = float(scaled_arr[cont_cols.index(c)])
        else:
            scaled_data[c] = float(native_data.get(c, 0.0))
            
    df = pd.DataFrame([scaled_data])
    opt = models['opt']
    
    # 1. Cox Risk
    cox_input = df[opt.cox_feature_cols]
    cph = models['cph']
    hazard = cph.predict_partial_hazard(cox_input).iloc[0]
    log_hazard = np.log(hazard)
    
    survival_func = cph.predict_survival_function(cox_input)
    times = survival_func.index.tolist()
    probs = survival_func.iloc[:, 0].tolist()
    
    t_10 = [t for t in times if t <= 10.0]
    p_10 = [probs[times.index(t)] for t in t_10]
    
    # XAI: Local Feature Contributions (Cox log-hazard)
    # We multiply the coefficient by the patient's scaled feature value
    feature_contributions = (cph.params_ * cox_input.iloc[0]).to_dict()
    
    # Sort contributions by absolute magnitude
    sorted_contributions = dict(sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True))
    
    # 2. Ensemble Tree Risk
    tree_input = df[opt.ae_feature_cols]
    prob_xgb = float(models['xgb'].predict_proba(tree_input)[0, 1])
    prob_lgb = float(models['lgb'].predict_proba(tree_input)[0, 1])
    prob_rf = float(models['rf'].predict_proba(tree_input)[0, 1])
    
    threshold = 0.15
    votes = sum(1 for p in [prob_xgb, prob_lgb, prob_rf] if p > threshold)
    
    return {
        "log_hazard": float(log_hazard),
        "survival_times": t_10,
        "survival_probs": p_10,
        "prob_xgb": prob_xgb,
        "prob_lgb": prob_lgb,
        "prob_rf": prob_rf,
        "ensemble_votes": votes,
        "xai_contributions": sorted_contributions
    }

@app.post("/api/optimize")
async def optimize_patient(req: PredictionRequest):
    native_data = req.patient_data
    required_cols = models['test_X'].columns
    for c in required_cols:
        if c not in native_data:
            native_data[c] = models['test_X'][c].median()
            
    scaler = models['scaler']
    cont_cols = models['cont_cols']
    
    native_arr = np.array([float(native_data.get(c, 0.0)) for c in cont_cols]).reshape(1, -1)
    scaled_arr = scaler.transform(native_arr)[0]
    
    scaled_data = {}
    for c in required_cols:
        if c in cont_cols:
            scaled_data[c] = float(scaled_arr[cont_cols.index(c)])
        else:
            scaled_data[c] = float(native_data.get(c, 0.0))
            
    df = pd.DataFrame([scaled_data])
    opt = models['opt']
    
    ae_input = df[opt.ae_feature_cols].values[0].astype(np.float32)
    res = opt.optimize(ae_input)
        
    cf_df = pd.DataFrame([res['x_cf']], columns=opt.ae_feature_cols)
    cf_native_df = cf_df.copy()
    cf_native_df[cont_cols] = scaler.inverse_transform(cf_df[cont_cols].values)
    
    cf_native = cf_native_df.iloc[0].to_dict()
    
    cf_cox_input = cf_df[opt.cox_feature_cols]
    cph = models['cph']
    survival_func = cph.predict_survival_function(cf_cox_input)
    times = survival_func.index.tolist()
    probs = survival_func.iloc[:, 0].tolist()
    t_10 = [t for t in times if t <= 10.0]
    p_10 = [probs[times.index(t)] for t in t_10]
    
    # XAI for counterfactual
    cf_feature_contributions = (cph.params_ * cf_cox_input.iloc[0]).to_dict()
    cf_sorted_contributions = dict(sorted(cf_feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True))
    
    diffs = []
    for feat in models['ACTIONABLE']:
        if feat in native_data and feat in cf_native:
            o_val = float(native_data[feat])
            c_val = float(cf_native[feat])
            if abs(o_val - c_val) > 0.5:
                diffs.append({
                    "feature": feat,
                    "original": o_val,
                    "recommended": c_val,
                    "change": c_val - o_val
                })
                
    return {
        "success": True, # The API call itself succeeded
        "status": res['status'],
        "relative_risk_reduction": res['relative_risk_reduction'],
        "prescription": diffs,
        "survival_times": t_10,
        "survival_probs": p_10,
        "cf_native": cf_native,
        "cf_xai_contributions": cf_sorted_contributions,
        "trajectory": res.get("trajectory", [])
    }

@app.post("/api/patient")
async def add_patient(req: PredictionRequest):
    custom_file = 'data/custom_patients.csv'
    df = pd.DataFrame([req.patient_data])
    
    if not os.path.exists(custom_file):
        df.to_csv(custom_file, index=False)
    else:
        df.to_csv(custom_file, mode='a', header=False, index=False)
        
    return {"success": True, "message": "Patient added successfully"}
