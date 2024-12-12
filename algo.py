import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('lung_cancer_data.csv', sep=',')

le = LabelEncoder()
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

X = df.iloc[:, :-1]

y = df['LUNG_CANCER'].values

regressor = LogisticRegression()
regressor.fit(X, y)

def getPrediction(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    probability = regressor.predict_proba(input_df)[0][1] * 100
    return probability

input_data = {'AGE': 15, 'SMOKING': 0, 'YELLOW_FINGERS': 0, 'ANXIETY': 0, 'CHRONIC_DISEASE': 0,
              'FATIGUE': 0, 'ALLERGY': 0, 'ALCOHOL_CONSUMING': 0, 'COUGHING': 0,
              'SHORTNESS_OF_BREATH': 0, 'SWALLOWING_DIFFICULTY': 0, 'CHEST_PAIN': 0,
              'GENDER': 0}

probability = getPrediction(input_data)
print(f"The probability of lung cancer is: {probability}%")
