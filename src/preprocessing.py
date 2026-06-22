from contractions import fix
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import re

negations = {'no', 'not', 'nor'}
stop_words = set(stopwords.words('english'))-negations
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    tokenized_words = [(word, tag) for word,tag in tagged_words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tokenized_words]
    cleaned_text = ' '.join(lemmatized_words)
    return cleaned_text

# print(clean_text('Customs held my international order ORD-71905 and I haven\'t gotten any updates in over a week.'))


