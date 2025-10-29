<<<<<<< HEAD
# heart-disease-prediction
project on heart disease prediction using machine learning models
=======
**How It Works**

**Data Collection**
The project uses the UCI Heart Disease Dataset, which includes clinical data such as age, cholesterol, blood pressure, ECG results, and exercise responses.

**Data Preprocessing**

Missing values are handled and categorical values are encoded.
Features are standardized using StandardScaler for better model performance.
Data is split into training (80%) and testing (20%) sets.

**Model Training**
Four machine learning models are trained and compared:

**Logistic Regression** – Linear classifier that predicts heart disease probability.
**Decision Tree** – Simple rule-based model for medical interpretation.
**Random Forest** – Ensemble of trees providing higher accuracy and stability.
**Naive Bayes** – Probabilistic model assuming independence among features.

**Prediction Logic**
After training, the model predicts whether a patient is at high or low risk based on:
Blood pressure, cholesterol, and age
ECG and exercise test results
Chest pain type and thalassemia score

**Result Interpretation**

The model outputs the risk prediction (High/Low) along with a confidence score.


**Short description:** This repository contains code to preprocess a heart disease dataset and train classification models (Decision Tree, Random Forest, Naive Bayes, Logistic Regression). The project is prepared for running locally in **VS Code**.

## Contents
- `src/` – Python scripts for preprocessing, models, and visualization
- `data/` – dataset download helper (empty by default; use the download script)
- `results/` – example outputs and saved model metrics
- `requirements.txt` – Python dependencies

## Dataset
The dataset used in the original project is hosted on Google Drive. To download it into `data/`, run:

```bash
pip install -r requirements.txt
python src/download_dataset.py
```

The script downloads the dataset to `data/heart.csv`. If you already have `heart.csv`, place it inside the `data/` folder.

> **Dataset URL used by the download script:**  
> `https://drive.google.com/uc?export=download&id=1U5Iwn7X_oJWmSiYBSF2AuQdiEvIaUMXv`

(If that Drive file is moved or removed, replace the ID in `src/download_dataset.py`.)

## Setup (VS Code)
1. Install Python 3.8+ and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # mac/linux
   venv\Scripts\activate    # windows (PowerShell/CMD)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the project folder in VS Code. Install the **Python** and **Jupyter** extensions if you want to run notebooks or interactive cells.

## How to run
1. Download the dataset:
   ```bash
   python src/download_dataset.py
   ```
2. Preprocess data (creates `data/processed_data.csv`):
   ```bash
   python src/preprocess.py
   ```
3. Train & evaluate models (each script will load `data/processed_data.csv`):
   ```bash
   python src/decision_tree.py
   python src/random_forest.py
   python src/naive_bayes.py
   python src/logistic_regression.py
   ```
4. Visualizations (produces PNGs in `results/`):
   ```bash
   python src/visualization.py
   ```

## Notes
- Scripts are written to be simple and easy to run in VS Code.
- If you prefer the dataset embedded in the repo, replace `src/download_dataset.py` with the CSV file placed at `data/heart.csv`.

## License & Acknowledgements


>>>>>>> 86267da (Initial commit - Heart Disease Prediction Project)
