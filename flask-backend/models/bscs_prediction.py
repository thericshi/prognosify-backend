import pandas as pd
import random
import pickle

# Load the model from the file
with open('bscs_lr.pkl', 'rb') as f:
    model = pickle.load(f)

print(model.coef_)

data = {
    'age_group_5_years': 9,
    'race_eth': 1,
    'first_degree_hx': 1,
    'age_first_birth': 4,
    'current_hrt': 0,
    'menopaus': 2,
    'bmi_group': 3,
    'biophx': 0,
    'breast_cancer_history': 1,
}

encoded_data = {'age_group_5_years': 6, 'age_first_birth': 0, 'bmi_group': 1, 'biophx': 0, 'breast_cancer_history': 0, 'race_eth_1': 0, 'race_eth_2': 0, 'race_eth_3': 0, 'race_eth_4': 1, 'race_eth_5': 0, 'race_eth_6': 0, 'menopaus_1': 1, 'menopaus_2': 0, 'menopaus_3': 0}

df = pd.DataFrame([encoded_data])

pd.options.display.max_columns = 20

print(df)

df = df.drop(columns=["breast_cancer_history"])

# print(model.coef_)

print(model.predict_proba(df))
print(model.predict(df))


