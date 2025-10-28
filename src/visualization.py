
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed data not found. Run src/preprocess.py first.")
    df = pd.read_csv(DATA_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=25)
    plt.title('Age distribution by target')
    plt.savefig(os.path.join(OUT_DIR, 'age_distribution.png'))
    plt.close()

    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f')
    plt.title('Correlation heatmap')
    plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'))
    plt.close()
    print('Saved visualization images to results/')

if __name__ == '__main__':
    main()
