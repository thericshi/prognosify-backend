import pandas as pd
import pickle


def predict_hd(data):
    # Load the model from the file
    with open('flask-backend/models/BRFSS2015lr76.pkl', 'rb') as f:
        model = pickle.load(f)

    coef = model.coef_
    coef[0][0] = 0.03
    coef[0][4] = -0.01
    coef[0][5] = -0.015
    model.coef_ = coef

    print(coef)

    data['BMI'] -= 20

    if data['BMI'] < 0:
        coef = model.coef_
        coef[0][0] = -0.04
        model.coef_ = coef

    df = pd.DataFrame([data])

    pd.options.display.max_columns = 20

    print(df)

    print(model.predict_proba(df))

    return model.predict_proba(df)[0][1]


