import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import clean_text
import feature_engineering


def evaluate(model_name, y_test_cat, y_test_pry, y_pred_cat, y_pred_pry):
    print('-'*100)
    print(model_name)
    print('-'*100, '\n')
    print('Category Prediction')
    print(f'Accuracy Score: {accuracy_score(y_test_cat, y_pred_cat)}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test_cat, y_pred_cat))
    print('Classification Report:')
    print(classification_report(y_test_cat, y_pred_cat))
    print('Priority Prediction')
    print(f'Accuracy Score: {accuracy_score(y_test_pry, y_pred_pry)}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test_pry, y_pred_pry))
    print('Classification Report:')
    print(classification_report(y_test_pry, y_pred_pry))
    print()

#                   Load data
curr_dir = Path(__file__).parent
data_path = curr_dir.parent/'data/support_tickets_dataset_v2.csv'
df = pd.read_csv(data_path, usecols=['subject', 'ticket_text','category','priority'])


#                   Preprocessing
df['text'] = df['subject'] + ' ' + df['ticket_text']
data = df.drop(columns=['subject', 'ticket_text'])
data['text'] = data['text'].apply(func=lambda x:clean_text(x))


#                   Feature Engineering
# print(data['priority'].value_counts())
x = data['text']
y = data.drop(columns=['text'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_trf, train_cat_encoded, train_pry_encoded, vectorizer, cat_encoder, pry_encoder = \
    feature_engineering.fit_transform_features(
        x_train, y_train['category'], y_train['priority']
    )
x_test_trf, test_cat_encoded, test_pry_encoded = feature_engineering.transform_features(
        x_test, y_test['category'], y_test['priority'], vectorizer, cat_encoder, pry_encoder
    )


#                   Train Models

lg_model_cat = LogisticRegression(random_state=42)
lg_model_cat.fit(x_train_trf, train_cat_encoded)
pred_cat_lgmodel = lg_model_cat.predict(x_test_trf)
lg_model_pry = LogisticRegression(random_state=42, class_weight='balanced')
lg_model_pry.fit(x_train_trf, train_pry_encoded)
pred_pry_lgmodel = lg_model_pry.predict(x_test_trf)

nb_model_cat = ComplementNB()
nb_model_cat.fit(x_train_trf, train_cat_encoded)
pred_cat_nbmodel = nb_model_cat.predict(x_test_trf)
nb_model_pry = ComplementNB()
nb_model_pry.fit(x_train_trf, train_pry_encoded)
pred_pry_nbmodel = nb_model_pry.predict(x_test_trf)

rfc_model_cat = RandomForestClassifier(random_state=42)
rfc_model_cat.fit(x_train_trf, train_cat_encoded)
pred_cat_rfcmodel = rfc_model_cat.predict(x_test_trf)
rfc_model_pry = RandomForestClassifier(random_state=42)
rfc_model_pry.fit(x_train_trf, train_pry_encoded)
pred_pry_rfcmodel = rfc_model_pry.predict(x_test_trf)


if __name__=='__main__':
    #                       Evaluate Models
    evaluate('Logistic Regression Model', test_cat_encoded, test_pry_encoded, pred_cat_lgmodel, pred_pry_lgmodel)
    evaluate('Naive Bayes Model', test_cat_encoded, test_pry_encoded, pred_cat_nbmodel, pred_pry_nbmodel)
    evaluate('Random Forest Classifier', test_cat_encoded, test_pry_encoded, pred_cat_rfcmodel, pred_pry_rfcmodel)
    
    
    #                       Save Models
    for sub_dir in ['category_models', 'priority_models', 'best_models']:
        (curr_dir.parent/'models'/sub_dir).mkdir(parents=True, exist_ok=True)
    models_path = curr_dir.parent/'models'
    with open(models_path/'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, file=f)
    with open(models_path/'category_encoder.pkl', 'wb') as f:
        pickle.dump(cat_encoder, file=f)
    with open(models_path/'priority_encoder.pkl', 'wb') as f:
        pickle.dump(pry_encoder, file=f)
    with open(models_path/'category_models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lg_model_cat, file=f)
    with open(models_path/'category_models/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model_cat, file=f)
    with open(models_path/'category_models/random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(rfc_model_cat, file=f)
    with open(models_path/'priority_models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lg_model_pry, file=f)
    with open(models_path/'priority_models/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model_pry, file=f)
    with open(models_path/'priority_models/random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(rfc_model_pry, file=f)
    with open(models_path/'best_models/category_rfc_model.pkl', 'wb') as f:
        pickle.dump(rfc_model_cat, file=f)
    with open(models_path/'best_models/priority_lg_model.pkl', 'wb') as f:
        pickle.dump(lg_model_pry, file=f)

