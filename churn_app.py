from pickle import load
from flask import Flask, request, jsonify
from utils import load_model, predict_churn

app = Flask('churn')
encoder, model = load_model('./models/LR')


# POST is used because we wanna send our input
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    y_pred, churn = predict_churn(customer, encoder, model)

    result = {
        'Churn Probability': y_pred,
        'churn': churn
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
