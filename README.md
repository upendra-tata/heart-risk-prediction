<<<<<<< HEAD
# heart-disease-prediction
project on heart disease prediction using machine learning models
=======



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
- Use as you like for academic purposes. Replace with your preferred license if needed.
- Dataset source referenced in the original report (Google Drive link).
>>>>>>> 86267da (Initial commit - Heart Disease Prediction Project)
