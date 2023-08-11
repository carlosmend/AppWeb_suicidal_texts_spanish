import spacy
import re
import string
import spacy

nlp = spacy.load('es_core_news_lg')

def cleanup(text):
    text = text.lower()
    text=re.sub(r'http:?\S+','',text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub("([^\x00-\x7F\u00C0-\u017F])+", " ", text)    
    doc = nlp(text)
    lemas_fila = [token.lemma_ for token in doc]
    text=' '.join(lemas_fila)
    
    return text

