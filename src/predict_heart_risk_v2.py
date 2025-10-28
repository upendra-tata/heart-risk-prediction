import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib, os

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "results", "rf_balanced.pkl")

# --- Load Model ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("⚠️ Model not found. Train using random_forest_balanced.py first.")
    model, scaler, features = joblib.load(MODEL_PATH)
    return model, scaler, features

# --- Generate explanation text ---
def generate_reason(patient):
    reasons = []

    if patient["cp"] == 0:
        reasons.append("Typical chest pain strongly indicates cardiac risk.")
    elif patient["cp"] == 3:
        reasons.append("Asymptomatic chest pain is less risky but may hide silent issues.")

    if patient["trestbps"] > 130:
        reasons.append("Elevated blood pressure puts extra strain on the heart.")
    elif patient["trestbps"] < 90:
        reasons.append("Unusually low blood pressure may reduce oxygen supply.")

    if patient["chol"] >= 240:
        reasons.append("High cholesterol can cause artery blockage.")
    elif patient["chol"] < 200:
        reasons.append("Healthy cholesterol level helps protect the heart.")

    if patient["thalach"] < 120:
        reasons.append("Low maximum heart rate suggests limited cardiovascular strength.")
    elif patient["thalach"] > 140:
        reasons.append("Good maximum heart rate capacity indicates healthy fitness.")

    if patient["exang"] == 1:
        reasons.append("Exercise-induced angina signals possible oxygen shortage to the heart.")
    else:
        reasons.append("No chest pain during exercise — good heart oxygen supply.")

    if patient["oldpeak"] >= 2.0:
        reasons.append("High ST-depression under stress may reflect poor heart recovery.")
    elif patient["oldpeak"] < 1:
        reasons.append("Normal ST-depression — healthy stress response.")

    if patient["ca"] >= 2:
        reasons.append("Multiple blocked vessels increase risk.")
    else:
        reasons.append("Few or no blocked vessels detected — good circulation.")

    if patient["thal"] == 3:
        reasons.append("Reversible thalassemia defect indicates abnormal blood flow.")
    elif patient["thal"] == 1:
        reasons.append("Normal thalassemia type — good oxygen transport.")

    return " ".join(reasons)

# --- Predict Heart Risk ---
def predict_heart_risk(patient):
    model, scaler, features = load_model()
    X = pd.DataFrame([patient], columns=features)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1] * 100

    clinically_low = (
        patient["age"] < 35 and patient["trestbps"] < 130 and patient["chol"] < 230
        and patient["thalach"] > 140 and patient["exang"] == 0 and patient["oldpeak"] < 1.5
        and patient["ca"] <= 1
    )

    if clinically_low and probability < 90:
        risk = "Low Risk ❤️"
    
        print("\n🩺 Clinical Sanity Check: Healthy profile detected.")
    else:
        risk = "High Risk of Heart Disease 💔" if prediction == 1 else "Low Risk ❤️"

    print(f"\n🔍 Prediction: {risk}")
    

    reason = generate_reason(patient)
    print(f"\n🧩 Reason: {reason}")

    if risk.startswith("High"):
        print("\n🩺 Recommendations:")
        print("- Maintain a balanced diet (low-fat, high-fiber).")
        print("- Exercise at least 30 mins/day (brisk walking, cycling).")
        print("- Avoid smoking and limit alcohol.")
        print("- Manage stress with meditation or yoga.")
        print("- Regularly monitor BP, cholesterol, and blood sugar.")
        print("- Consult a cardiologist for periodic check-ups.")
    else:
        print("\n✅ Great! Keep up your healthy habits to maintain your heart health.")

# --- Helper for Inputs ---
def ask(prompt, info):
    return float(input(f"{prompt} [{info}]: "))

if __name__ == "__main__":
    print("Enter patient details to assess heart disease risk:")

    sex_input = input("Sex (1=Male, 0=Female, 2=Other): ").strip().lower()
    sex = 0.5 if sex_input in ["2", "other"] else float(sex_input)

    p = {
        "age": ask("Age", "Normal adult: 20–45 | Risk rises >45"),
        "sex": sex,
        "cp": ask("Chest pain type (0–3)", "0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic"),
        "trestbps": ask("Resting blood pressure (mmHg)", "Normal: 90–120 | High: >130"),
        "chol": ask("Cholesterol level (mg/dL)", "Healthy: <200 | Borderline: 200–239 | High: ≥240"),
        "fbs": ask("Fasting blood sugar >120 mg/dL (1=True, 0=False)", "Normal: 0 | High: 1"),
        "restecg": ask("Resting ECG results (0–2)", "0=Normal | 1=ST-T abnormality | 2=LVH"),
        "thalach": ask("Max heart rate achieved", "Healthy: 140–180 | Low: <120"),
        "exang": ask("Exercise induced angina (1=Yes, 0=No)", "0=No | 1=Yes"),
        "oldpeak": ask("ST depression induced by exercise", "Normal: <1 | Risk: ≥2"),
        "slope": ask("Slope of peak exercise ST segment (0–2)", "0=Upsloping | 1=Flat | 2=Downsloping"),
        "ca": ask("Number of major vessels (0–4)", "Normal: 0–1 | Risk: ≥2"),
        "thal": ask("Thalassemia (1–3)", "1=Normal | 2=Fixed defect | 3=Reversible defect")
    }

    print("\n---------------------------------------------")
    predict_heart_risk(p)
    print("---------------------------------------------")
