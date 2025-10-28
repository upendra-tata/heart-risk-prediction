import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# --- Universal dataset path ---
DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "heart.csv")
DATA_PATH = os.path.abspath(DATA_PATH)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Models to Compare ---
    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_split=4, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_split=5, min_samples_leaf=3, bootstrap=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.5, random_state=42)
    }

    results = []

    # --- Train and Evaluate Each Model ---
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": round(acc * 100, 2)})

    # --- Create Results DataFrame ---
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    print("\n📊 Model Accuracy Comparison 📊")
    print(results_df.to_string(index=False))

    # --- Save Results ---
    results_dir = os.path.join(os.path.dirname(__file__), os.pardir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "accuracy_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")

    # --- Plot Bar Chart ---
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"], color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plot_path = os.path.join(results_dir, "accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Bar chart saved to: {plot_path}")

if __name__ == "__main__":
    main()
