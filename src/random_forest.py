import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "results", "rf_balanced.pkl")

def train_and_evaluate():
    print("🩺 Training Balanced Random Forest Model...\n")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Proper stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a balanced, regularized Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        class_weight="balanced",   # helps if data is slightly imbalanced
        random_state=42
    )

    # Train and evaluate
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"🎯 Training Accuracy: {train_acc * 100:.2f}%")
    print(f"🎯 Testing Accuracy : {test_acc * 100:.2f}%")

    # Overfitting detection
    if train_acc - test_acc > 0.05:
        print("⚠️ Warning: Possible overfitting detected (training much higher than testing).")
    elif test_acc < 0.80:
        print("⚠️ Model may be underfitting. Try tuning hyperparameters.")
    else:
        print("✅ Model performance looks balanced and reliable.")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    # Cross-validation for robustness
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf)
    print(f"📊 Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler, list(X.columns)), MODEL_PATH)
    print("\n💾 Balanced Random Forest model saved to results/rf_balanced.pkl")

if __name__ == "__main__":
    train_and_evaluate()
