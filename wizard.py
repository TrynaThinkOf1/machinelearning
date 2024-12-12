"""DATA NEEDED: GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, CHRONIC_DISEASE, FATIGUE,
ALLERGY, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY,
CHEST_PAIN
"""
import pandas as pd
import joblib

regressor = joblib.load("lung_cancer_model.pkl")
model_columns = joblib.load("model_columns.pkl")

def getPrediction(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    probability = regressor.predict_proba(input_df)[0][1] * 100
    #print(f"Input Data: {input_df}")
    #print(f"Coefficients: {regressor.coef_}")
    #print(f"Prediction Probability: {probability}")
    return round(probability, 2)

def getData():
    gender = (1 if input("Gender: ") in ["male", "m", "1"] else 0)
    age = int(input("Age: "))
    smoking = (1 if input("Do you smoke or vape? ") in ["yes", "y", "1"] else 0)
    yellow_fingers = (1 if input("Do you have yellow fingers? ") in ["yes", "y", "1"] else 0)
    anxiety = (1 if input("Do you often feel anxious? ") in ["yes", "y", "1"] else 0)
    chronic_disease = (1 if input("Do you have a chronic disease? ") in ["yes", "y", "1"] else 0)
    fatigue = (1 if input("Do you often feel fatigued? ") in ["yes", "y", "1"] else 0)
    allergy = (1 if input("Do you have allergies? ") in ["yes", "y", "1"] else 0)
    alcohol_consuming = (1 if input("Do you consume alcohol? ") in ["yes", "y", "1"] else 0)
    coughing = (1 if input("Do you cough often? ") in ["yes", "y", "1"] else 0)
    shortness_of_breath = (1 if input("Do you have shortness of breath? ") in ["yes", "y", "1"] else 0)
    swallowing_difficulty = (1 if input("Do you have difficulty swallowing? ") in ["yes", "y", "1"] else 0)
    chest_pain = (1 if input("Do you have chest pain? ") in ["yes", "y", "1"] else 0)

    data = {
        "GENDER": gender,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "CHRONIC_DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "ALCOHOL_CONSUMING": alcohol_consuming,
        "COUGHING": coughing,
        "SHORTNESS_OF_BREATH": shortness_of_breath,
        "SWALLOWING_DIFFICULTY": swallowing_difficulty,
        "CHEST_PAIN": chest_pain,
    }
    return data

if __name__ == "__main__":
    data = getData()
    print(f"The chances of your having lung cancer are: {getPrediction(data)}%")