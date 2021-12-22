import pickle


def load_model(model_path):
    with open(model_path, 'rb') as file:
        (encoder, model) = pickle.load(file)
    return encoder, model


def predict_churn(customer, encoder, model):
    X = encoder.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    if y_pred >= 0.5:
        churn = True
    else:
        churn = False

    return y_pred.round(3), churn
