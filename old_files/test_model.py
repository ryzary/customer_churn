import pickle
import pandas as pd


def predict(input_dict, encoder, model):
    X = encoder.transform(input_dict)
    y_pred = model.predict_proba(X)[:, 1]
    y_pred = y_pred[0].round(4)
    return {"Prediction": y_pred}


def load_model(model_path):
    with open(model_path, 'rb') as file:
        (encoder, model) = pickle.load(file)
    return encoder, model


# input
customer = [{'tenure': 50, 'MonthlyCharges': 10, 'TotalCharges': 2894.55, 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'Yes', 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
             'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Mailed check'}]

encoder, model = load_model(model_path='models/LR')
print(predict(customer, encoder, model))
