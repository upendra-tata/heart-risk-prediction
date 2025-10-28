
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Universal dataset path ---
DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
DATA_PATH = os.path.abspath(DATA_PATH)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale data correctly (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ Tuned Random Forest to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=150,      # number of trees
        max_depth=6,           # limits tree growth
        min_samples_split=5,   # requires more samples to split
        min_samples_leaf=3,    # prevents very small leaf nodes
        bootstrap=True,        # random sampling of data
        random_state=42
    )

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print("\n🌲 Random Forest Classifier (Tuned) 🌲")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
