import sys
sys.path.append('C:/Users/Bea/fastText/fasttext-env/Lib/site-packages')

import os
import re
import numpy as np
import pandas as pd
import pickle
import fasttext
import fasttext.util
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

csv_path = 'C:/Users/Bea/Documents/TFG/csv_gl_comments_artistas.csv'
pkl_path = 'C:/Users/Bea/Documents/TFG/best_model_SVM.pkl'

def generate_sentence_embeddings(text, fasttext_model):

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = tokenizer.tokenize(text)
    embeddings = [fasttext_model.get_word_vector(word) for word in tokens]
    sentence_embedding = sum(embeddings) / len(embeddings)
    return ' '.join(str(val) for val in sentence_embedding)

df = pd.read_csv(csv_path)
df = df.dropna(subset=['text'])  # Remove rows with empty texts

# Download FastText model and load it
fasttext.util.download_model('gl', if_exists='ignore')  # Galician
fasttext_model = fasttext.load_model('cc.gl.300.bin')

with open(pkl_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Generate sentence embeddings
sentence_embeddings = df['text'].apply(lambda text: generate_sentence_embeddings(text, fasttext_model))
sentence_embeddings = np.array(sentence_embeddings.tolist())

predictions = model.predict(sentence_embeddings)

print("Predicciones: ", predictions)