from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load(filename):
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        return pickle.load(f)

health_model    = load("health_model.pkl")
health_features = load("health_features.pkl")
home_model      = load("home_model.pkl")
home_features   = load("home_features.pkl")
auto_model      = load("auto_model.pkl")
auto_features   = load("auto_features.pkl")
life_model      = load("life_model.pkl")
life_features   = load("life_features.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data    = request.json
    results = {}

    # ── Health ────────────────────────────────────────
    try:
        region = data.get("region", "northeast")
        health_input = {
            "age":              float(data["age"]),
            "sex":              1.0 if data["sex"] == "male" else 0.0,
            "bmi":              float(data["bmi"]),
            "children":         float(data["children"]),
            "smoker":           1.0 if data["smoker"] == "yes" else 0.0,
            "region_northwest": 1.0 if region == "northwest" else 0.0,
            "region_southeast": 1.0 if region == "southeast" else 0.0,
            "region_southwest": 1.0 if region == "southwest" else 0.0,
        }
        h_arr  = np.array([[health_input[f] for f in health_features]])
        h_pred = max(0, float(health_model.predict(h_arr)[0]))
        results["health"] = {
            "annual":  round(h_pred, 2),
            "monthly": round(h_pred / 12, 2),
            "factors": {k: round(v, 2) for k, v in health_input.items()}
        }
    except Exception as e:
        results["health"] = {"error": str(e)}

    # ── Home ──────────────────────────────────────────
    try:
        home_input = {
            "P1_EMP_STATUS":        float(data.get("emp_status", 1)),
            "BUS_USE":              float(data.get("bus_use", 0)),
            "AD_BUILDINGS":         float(data.get("ad_buildings", 1)),
            "RISK_RATED_AREA_B":    float(data.get("risk_area", 3)),
            "SUM_INSURED_BUILDINGS":float(data.get("sum_insured", 200000)),
            "NCD_GRANTED_YEARS_B":  float(data.get("ncd_years", 3)),
            "AD_CONTENTS":          float(data.get("ad_contents", 1)),
            "RISK_RATED_AREA_C":    float(data.get("risk_area", 3)),
            "SUM_INSURED_CONTENTS": float(data.get("sum_contents", 50000)),
            "NCD_GRANTED_YEARS_C":  float(data.get("ncd_years", 3)),
            "CONTENTS_COVER":       float(data.get("contents_cover", 1)),
            "BUILDINGS_COVER":      float(data.get("buildings_cover", 1)),
            "P1_MAR_STATUS":        float(data.get("mar_status", 1)),
            "P1_SEX":               1.0 if data["sex"] == "male" else 0.0,
            "APPR_ALARM":           float(data.get("alarm", 0)),
            "APPR_LOCKS":           float(data.get("locks", 1)),
            "BEDROOMS":             float(data.get("bedrooms", 3)),
            "ROOF_CONSTRUCTION":    float(data.get("roof", 1)),
            "WALL_CONSTRUCTION":    float(data.get("wall", 1)),
            "FLOODING":             float(data.get("flooding", 0)),
            "NEIGH_WATCH":          float(data.get("neigh_watch", 0)),
            "OCC_STATUS":           float(data.get("occ_status", 1)),
            "OWNERSHIP_TYPE":       float(data.get("ownership", 1)),
            "PROP_TYPE":            float(data.get("prop_type", 1)),
            "SAFE_INSTALLED":       float(data.get("safe", 0)),
            "SUBSIDENCE":           float(data.get("subsidence", 0)),
            "YEARBUILT":            float(data.get("year_built", 1990)),
            "CLAIM3YEARS":          float(data.get("claim3years", 0)),
        }
        ho_arr  = np.array([[home_input[f] for f in home_features]])
        ho_pred = max(0, float(home_model.predict(ho_arr)[0]))
        results["home"] = {
            "annual":  round(ho_pred, 2),
            "monthly": round(ho_pred / 12, 2)
        }
    except Exception as e:
        results["home"] = {"error": str(e)}

    # ── Auto ──────────────────────────────────────────
    try:
        auto_input = {
            "Driver Age":                float(data.get("driver_age", 30)),
            "Driver Experience":         float(data.get("driver_experience", 5)),
            "Previous Accidents":        float(data.get("prev_accidents", 0)),
            "Annual Mileage (x1000 km)": float(data.get("annual_mileage", 15)),
            "Car Manufacturing Year":    float(data.get("car_year", 2018)),
            "Car Age":                   float(data.get("car_age", 5)),
        }
        a_arr  = np.array([[auto_input[f] for f in auto_features]])
        a_pred = max(0, float(auto_model.predict(a_arr)[0]))
        results["auto"] = {
            "annual":  round(a_pred, 2),
            "monthly": round(a_pred / 12, 2)
        }
    except Exception as e:
        results["auto"] = {"error": str(e)}

    # ── Life ──────────────────────────────────────────
    try:
        life_input = {
            "ENTRY AGE":       float(data.get("age", 30)),
            "SEX":             1.0 if data.get("sex") == "male" else 0.0,
            "POLICY TYPE 1":   float(data.get("policy_type", 1)),
            "PAYMENT MODE":    float(data.get("payment_mode", 1)),
            "BENEFIT":         float(data.get("life_benefit", 100000)),
            "SUBSTANDARD RISK":float(data.get("substandard_risk", 0)),
            "Policy Year":     float(data.get("policy_year", 10)),
        }
        l_arr  = np.array([[life_input[f] for f in life_features]])
        l_pred = max(0, float(life_model.predict(l_arr)[0]))
        results["life"] = {
            "annual":  round(l_pred, 2),
            "monthly": round(l_pred / 12, 2)
        }
    except Exception as e:
        results["life"] = {"error": str(e)}

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)