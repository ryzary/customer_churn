from os import X_OK, write
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import pickle

numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod']

df_train_full = pd.read_csv('data/train.csv', index_col=0)


def train(X, y, C=1.0):
    dicts = X[numerical+categorical].to_dict(orient='records')

    encoder = DictVectorizer(sparse=False)

    X_encoded = encoder.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_encoded, y)

    return encoder, model


def predict(X, encoder, model):
    dicts = X.to_dict(orient='records')

    X = encoder.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)

scores = []

fold = 0
for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]

    y_train = df_train['Churn'].values
    y_val = df_val['Churn'].values

    encoder, model = train(df_train, y_train)
    y_pred = predict(df_val, encoder, model)

    auc = roc_auc_score(y_val, y_pred)
    print(f'AUC on fold {fold} is {auc}')
    scores.append(auc)
    fold += 1

print('Mean scores & std: ', round(np.mean(scores), 3), round(np.std(scores), 3))


# save model and encoder
final_encoder, final_model = train(
    df_train_full[numerical+categorical], y=df_train_full['Churn'])

# with open('./models/LR', 'wb') as file:
#     pickle.dump((final_encoder, final_model), file)
