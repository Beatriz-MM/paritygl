# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 
# Description: 
# Python version: 3.10.12

import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import fasttext
import fasttext.util
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.sparse import hstack


csv_path = '/home/beaunix/TFG/langdetect/PRUEBA/Prueba_embeddings/csv_gl_comments_prueba.csv'
output_path = '/home/beaunix/TFG/langdetect/PRUEBA/Prueba_embeddings/Ultimas_pruebas/ultimos_resultados/resultado_predictions_ultima_prueba.csv'


#

# Tweet preprocessing: clean text and remove noise
def preprocess_tweet(tweet):
    if not isinstance(tweet, str) or tweet is None:
        return None
    
    tweet = tweet.lower()
    tweet = re.sub(r'\n', '', tweet)
    tweet = re.sub(r'http://t.co/[a-zA-Z0-9]+', 'http://t.co', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'::', '', tweet)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1', tweet)
    if not tweet.strip():
        return None
    return tweet

# Generate FastText sentence embedding by averaging word vectors
def generate_sentence_embeddings(tweet, fasttext_model):
    try: 
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tokenizer.tokenize(tweet)

        if not tokens:
            raise ValueError(f"No tokens found for tweet '{tweet}'")

        embeddings = [fasttext_model.get_word_vector(word) for word in tokens]
        sentence_embedding = sum(embeddings) / len(embeddings)
        return ' '.join(str(val) for val in sentence_embedding)
        
    except Exception as e:
        print(f"Error generating embeddings for tweet '{tweet}': {e}")
        raise

#----------------------------------------------------------------------------------------
#CARGAMOS LOS DATASET DE LUCÍA Y LOS ETIQUETAMOS

def load_datasets():

    # Cargamos el dataset 0 (no misogino toots)
    df_toots = pd.read_csv('/home/beaunix/TFG/GalMisoCorpus2023/corpus/toots.csv')
    df_toots['content'] = df_toots['content'].apply(preprocess_tweet)
    df_toots = df_toots.dropna(subset=['content'])  # Eliminamos filas vacías
    X_0 = df_toots['content']
    y_0 = pd.Series([0] * len(X_0))

    # Cargamos el dataset 1 (misogino tweets)
    df_tweets = pd.read_csv('/home/beaunix/TFG/tweets.csv')
    df_tweets['content'] = df_tweets['content'].apply(preprocess_tweet)
    df_tweets = df_tweets.dropna(subset=['content'])  # Eliminamos filas vacías
    X_1 = df_tweets['content'] 
    y_1 = pd.Series([1] * len(X_1))

    # Combinamos los datasets
    X = pd.concat([X_0, X_1], ignore_index=True)
    y = pd.concat([y_0, y_1], ignore_index=True)

    return X, y

def prepare_embeddings(text_series, fasttext_model):
    sentence_embeddings = text_series.apply(lambda tweet: generate_sentence_embeddings(tweet, fasttext_model))
    sentence_embeddings = np.array(sentence_embeddings.tolist())
    return sentence_embeddings

fasttext.util.download_model('gl', if_exists='ignore')  # Galician
fasttext_model = fasttext.load_model('cc.gl.300.bin')

#Preparamos los datos para entrenar y generamos embeddings
X, y = load_datasets()
sentence_embeddings = prepare_embeddings(X, fasttext_model)

#---------------------------------------------------------------------------------

# Generación de BoW
vectorizer = CountVectorizer()
bow_features = vectorizer.fit_transform(X)

# Aplicación de TF-IDF
tfidf_transformer = TfidfTransformer()
tfidf_features = tfidf_transformer.fit_transform(bow_features)

# Selección de características con chi2
k_best_selector = SelectKBest(score_func=chi2) 

# Seleccionamos las k características más relevantes
selected_features = k_best_selector.fit_transform(tfidf_features, y)

# Concatenar embeddings y BoW con TF-IDF y chi2
sentence_embeddings = [np.fromstring(embedding, sep=' ') for embedding in sentence_embeddings]
sentence_embeddings = np.array(sentence_embeddings, dtype=np.float32)

print("FORMA setence_embeddings: ", sentence_embeddings.shape)

sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

#---------------------------------------------------------------------------------

print("Forma de combined_features:", combined_features.shape)
print("FORMA y:", y.shape)

# Dividimos el dataset en conjuntos de entrenamiento y testing, siendo el 70% entrenamiento y 30% pruebas
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.3)

svc = SVC()

print("Buscamos el mejor modelo con EL GRID")
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='f1', cv=10)
grid_search.fit(X_train, y_train)
modelo_entrenado = grid_search.best_estimator_


# Generamos la matriz de confusión
y_pred = modelo_entrenado.predict(X_test)
cmatrix = confusion_matrix(y_test, y_pred)
cmDisplay = ConfusionMatrixDisplay(cmatrix)

y_pred = modelo_entrenado.predict(X_test)
print("Evaluation on Test Set:")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#cmDisplay.plot()
#plt.show()

# Crear un heatmap con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)

plt.title("Matriz de Confusión")
plt.ylabel("Etiqueta Real")
plt.xlabel("Etiqueta Predicha")
plt.savefig("./ultimos_resultados/matriz_confusion_ultima_prueba.png")
plt.close()


# Read the CSV file and drop rows with empty texts
try:
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(preprocess_tweet)
    df = df.dropna(subset=['text'])  # Remove rows with empty texts
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

# Generamos embeddings para el nuevo corpus (mi CSV)
new_text_embeddings = prepare_embeddings(df['text'], fasttext_model)

new_text_embeddings = [np.fromstring(embedding, sep=' ') for embedding in new_text_embeddings]
new_text_embeddings = np.array(new_text_embeddings, dtype=np.float32)

# Generamos BoW y TF-IDF para el nuevo texto, luego seleccionamos las k características más relevantes
new_bow_features = vectorizer.transform(df['text'])
new_tfidf_features = tfidf_transformer.transform(new_bow_features)
new_selected_features = k_best_selector.transform(new_tfidf_features)

# Combinamos características
new_text_embeddings_sparse = sparse.csr_matrix(new_text_embeddings)
new_combined_features = hstack([new_text_embeddings_sparse, new_selected_features])


print("Forma de new_combined_features:", new_combined_features.shape)
print("Modelo espera:", modelo_entrenado.n_features_in_)

predictions = modelo_entrenado.predict(new_combined_features)

# Guardamos predicciones en el DataFrame
df['predictions'] = predictions
df[['text', 'predictions']].to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
