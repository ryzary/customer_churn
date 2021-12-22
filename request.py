import requests

customer_input = {'tenure': 50, 'MonthlyCharges': 10, 'TotalCharges': 2894.55, 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'Yes', 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
                  'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Mailed check'}

url = 'http://192.168.150.101:9696/predict'


def get_prediction(url, input):
    response = requests.post(url, json=input).json()
    return response


print(get_prediction(url, customer_input))
