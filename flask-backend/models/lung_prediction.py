import pandas as pd
import pickle

# Load the model from the file
with open('flask-backend/models/lc_rf.pkl', 'rb') as f:
    model = pickle.load(f)

def keys_to_remove():
    data = {
        "Patient Id": "P1",
        "Age": 73,
        "Gender": 1,
        "Air Pollution": 0,
        "Alcohol use": 0,
        "Dust Allergy": 0,
        "Occupational Hazards": 6,
        "Genetic Risk": 0,
        "chronic Lung Disease": 0,
        "Balanced Diet": 0,
        "Obesity": 1,
        "Smoking": 1,
        "Passive Smoker": 1,
        "Chest Pain": 2,
        "Coughing of Blood": 4,
        "Fatigue": 3,
        "Weight Loss": 4,
        "Shortness of Breath": 2,
        "Wheezing": 2,
        "Swallowing Difficulty": 3,
        "Clubbing of Finger Nails": 1,
        "Frequent Cold": 2,
        "Dry Cough": 3,
        "Snoring": 4
    }

    return ["Age", "Patient Id", "Occupational Hazards", "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing", "Swallowing Difficulty", "Dry Cough", "Clubbing of Finger Nails", "Frequent Cold", "Snoring", "Smoking"]


def format_data(data):
    mapping = {'AirPollution': 'Air Pollution', 'AlcoholUse': 'Alcohol use', 'BalancedDiet': 'Balanced Diet',
                'ChronicLungDisease': 'chronic Lung Disease', 'DustAllergy': 'Dust Allergy', 'Gender': 'Gender',
                'Obesity': 'Obesity', 'PassiveSmoker': 'Passive Smoker', 'GeneticRisk': 'Genetic Risk'}

    return {mapping[key]: value for key, value in data.items()}



def predict_lc(data):
    [data.pop(key, None) for key in keys_to_remove()]

    print(data)

    prob = model.predict_proba(pd.DataFrame([data]))[0]

    print(model.predict(pd.DataFrame([data])))
    print(prob)
    print(model.classes_)

    print(prob[0] * 0.21 + prob[1] * 0.02 + prob[2] * 0.07)

    return prob[0] * 0.21 + prob[1] * 0.02 + prob[2] * 0.07

