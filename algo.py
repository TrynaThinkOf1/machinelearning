import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('lung_cancer_data.csv', sep=',')

le = LabelEncoder()
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

X = df.iloc[:, :-1]
y = df['LUNG_CANCER'].values

regressor = LogisticRegression(max_iter=500)
regressor.fit(X, y)

joblib.dump(regressor, "regressor_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

def plot():
    x_values = X.values.flatten()
    y_values = np.repeat(y, X.shape[1])

    plt.scatter(x_values, y_values, alpha=0.6, c=y_values, cmap='coolwarm', edgecolor='k')

    plt.xlabel("Feature Values (Flattened)")
    plt.ylabel("Lung Cancer (0 = No, 1 = Yes)")
    plt.title("Feature Values vs Lung Cancer")
    plt.colorbar(label="Lung Cancer")
    plt.show()

if __name__ == "__main__":
    plot()
