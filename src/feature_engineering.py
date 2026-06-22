import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def get_tfidf_vectorizer(max_features=5000, ngram_range=(1,2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )

def get_label_encoder():
    return LabelEncoder()

priority_order = ['Low', 'Medium', 'High', 'Critical']
def get_ordinal_encoder(categories=None):
    categories = categories or priority_order
    return OrdinalEncoder(categories=[categories])

def fit_transform_features(text, category, priority, pry_categories=None):
    vectorizer = get_tfidf_vectorizer()
    cat_encoder = get_label_encoder()
    pry_encoder = get_ordinal_encoder(pry_categories)
    return (vectorizer.fit_transform(text), cat_encoder.fit_transform(category), 
        pry_encoder.fit_transform(priority.to_frame()).ravel(), 
        vectorizer, cat_encoder, pry_encoder)

def transform_features(text, category, priority, vectorizer, cat_encoder, pry_encoder):
    return (vectorizer.transform(text), cat_encoder.transform(category), 
        pry_encoder.transform(priority.to_frame()).ravel())

    