import sys
import os
import re
import numpy as np
import pandas as pd
import pickle
import fasttext
import fasttext.util
import joblib
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

csv_path = 'C:/Users/Usuario/Desktop/COMMENTS_GAL/csv_gl_comments_artistas.csv'
pkl_path = 'C:/Users/Usuario/Documents/TFG/GalMisoCorpus2023/models/best_model_SVM.pkl'

def generate_sentence_embeddings(text, fasttext_model):
    try:
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tokenizer.tokenize(text)
        #print(tokens)
        embeddings = [fasttext_model.get_word_vector(word) for word in tokens]
        if len(embeddings) == 0:
            return np.zeros(fasttext_model.get_dimension())
        sentence_embedding = sum(embeddings) / len(embeddings)
        return sentence_embedding
    except Exception as e:
        print(f"Error generating embeddings for text '{text}': {e}")
        return np.zeros(fasttext_model.get_dimension())

# Read the CSV file and drop rows with empty texts
try:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text'])  # Remove rows with empty texts
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

# Download FastText model and load it
try:
    fasttext.util.download_model('gl', if_exists='ignore')  # Galician
    fasttext_model = fasttext.load_model('cc.gl.300.bin')
    print("FastText model loaded successfully.")
except Exception as e:
    print(f"Error loading FastText model: {e}")
    sys.exit(1)

# Load the trained model from the pickle file
try:
    with open(pkl_path, 'r+b') as model_file:
        model = joblib.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading pickle file: {e}")
    model = None

if model:
    try:
        # Generate sentence embeddings
        texto=df['text'].astype(str)
        #print(texto)
        sentence_embeddings = texto.apply(lambda text: generate_sentence_embeddings(text, fasttext_model))
        print(sentence_embeddings)
        sentence_embeddings = np.array(sentence_embeddings.tolist())
        print("Sentence embeddings generated successfully.")

        # Make predictions
        predictions = model.predict(sentence_embeddings)
        print("Predictions: ", predictions)
    except Exception as e:
        print(f"Error during prediction: {e}")
else:
    print("Model could not be loaded. Check the pickle file and version compatibility.")