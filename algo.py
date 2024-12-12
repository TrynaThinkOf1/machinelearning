import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

def process():
    df = pd.read_csv('lung_cancer_data.csv', sep=',')

    le = LabelEncoder()
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

    X = df.iloc[:, :-1]

    y = df['LUNG_CANCER'].values

    regressor = LogisticRegression()
    regressor.fit(X, y)

    joblib.dump(regressor, "lung_cancer_model.pkl")
    joblib.dump(X.columns, "model_columns.pkl")

if __name__ == "__main__":
    process()