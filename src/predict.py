import pickle
from pathlib import Path
from preprocessing import clean_text

models_path = Path(__file__).parent.parent/'models'
with open(models_path/'tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open(models_path/'category_encoder.pkl', 'rb') as file:
    category_encoder = pickle.load(file)
with open(models_path/'priority_encoder.pkl', 'rb') as file:
    priority_encoder = pickle.load(file)
with open(models_path/'best_models/category_rfc_model.pkl', 'rb') as file:
    cat_pred_model = pickle.load(file)
with open(models_path/'best_models/priority_lg_model.pkl', 'rb') as file:
    pry_pred_model = pickle.load(file)

def prediction(subject, ticket_text):
    one_ticket =  isinstance(subject, str)
    subjects = [subject] if one_ticket else list(subject)
    ticket_texts = [ticket_text] if one_ticket else list(ticket_text)
    if len(subjects)!=len(ticket_texts):
        raise ValueError('Ticket text and subject must be the same length')
    cleaned_text = [clean_text(s+' '+t) for s, t in zip(subjects, ticket_texts)]
    x_trf = tfidf_vectorizer.transform(cleaned_text)
    category = cat_pred_model.predict(x_trf)
    category = category_encoder.inverse_transform(category)
    priority = pry_pred_model.predict(x_trf)
    priority = priority_encoder.inverse_transform(priority.reshape(-1,1)).ravel()
    if one_ticket:
        return (category[0], priority[0])
    return (category, priority)

