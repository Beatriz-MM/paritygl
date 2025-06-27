# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 23/06/2025
# Description: Trains an SVM model to classify Instagram comments as negative (label=1) or not (label=0).
# Merges labeled data with confirmed negatives, applies random undersampling, 
# and uses FastText embeddings (Galician) + TF-IDF features for classification.
# Python version: 3.10.12

import os
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import hstack


RANDOM_SEED = 42

# IMPORTANT: Insert correct paths
# Paths to input datasets
negative_dataset_path = "" # Example: "~/corpus/negative_dataset.csv"
csv_path = "" # Example: "~/csv_datasets/all_comments.csv"

# Paths to output files
report_path = "" # Example: "~/sampling_results/classification_comments_report_sampling.txt"
matrix_path = "" # Example: "~/sampling_results/confusion_matrix_sampling.png"
output_path = "" # Example: "~/sampling_results/result_predictions_sampling.csv"

# ---------- FUNCTIONS ----------

def load_dataset():
    """
    Load and merge the labeled dataset with additional negative samples.

    Reads a CSV with labeled data and another with confirmed negative examples,
    assigns label 1 to negatives, concatenates both, removes duplicates, 
    and shuffles the resulting dataset.

    Returns:
        pandas.DataFrame: Combined and shuffled dataset with 'text' and 'label' columns.
    """
    neg_df = pd.read_csv(negative_dataset_path, usecols=['text'])
    neg_df['label'] = 1  

    dataset_df = pd.read_csv(csv_path, usecols=['text', 'label'])

    # Merge and clean
    corpus_df = pd.concat([dataset_df, neg_df], ignore_index=True)
    corpus_df = corpus_df.drop_duplicates(subset='text')
    corpus_df = corpus_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Label distribution in the combined corpus:")
    print(corpus_df['label'].value_counts())

    return corpus_df


def get_fasttext_model():
    """
    Download and load the FastText model for Galician language (if not already present).

    Returns:
        fasttext.FastText._FastText: Loaded FastText model.
    """
    fasttext.util.download_model('gl', if_exists='ignore')
    return fasttext.load_model('cc.gl.300.bin')


def plot_conf_matrix(y_true, y_pred, title, save_path=None):
    """
    Plot and display a confusion matrix using seaborn heatmap.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Confusion Matrix saved at {save_path}")
    plt.show()
    plt.close()

def prepare_embeddings(text_series, fasttext_model, batch_size=1000):
    """
    Generate document embeddings by averaging FastText word vectors.

    Args:
        text_series (pandas.Series): Series of text documents.
        fasttext_model: Loaded FastText model.
        batch_size (int, optional): Number of texts to process per batch. Default is 1000.

    Returns:
        numpy.ndarray: Array of shape (n_samples, 300) with averaged embeddings.
    """
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    embeddings = []

    for start in range(0, len(text_series), batch_size):
        batch_texts = text_series[start:start + batch_size]
        batch_embeddings = []

        for text in batch_texts:
            tokens = tokenizer.tokenize(text)
            vectors = [fasttext_model.get_word_vector(token) for token in tokens]
            if not vectors:
                batch_embeddings.append(np.zeros(300))
            else:
                batch_embeddings.append(np.mean(vectors, axis=0))

        embeddings.extend(batch_embeddings)

    return np.array(embeddings, dtype=np.float32)


def train_model(X_train, y_train, combined_features, param_grid):
    """
    Train a classification model using grid search with cross-validation.

    Args:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Labels corresponding to training data.
        combined_features: Scikit-learn pipeline or model with transformers and estimator.
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.

    Returns:
        sklearn.base.BaseEstimator: Best trained model from grid search.
    """
    grid_search = GridSearchCV(estimator=combined_features, param_grid=param_grid, scoring='f1', cv=10)
    grid_search.fit(X_train, y_train)
    trained_model = grid_search.best_estimator_
    return trained_model


# ---------- MAIN ----------

df = load_dataset()

# Remove empty or NaN text entries (just in case, to avoid downstream errors)
df = df.dropna(subset=['text'])                      
df = df[df['text'].str.strip() != ""]          

X = df['text']
y = df['label']

# Download and load FastText Galician model
if not os.path.exists('cc.gl.300.bin'):
    fasttext.util.download_model('gl', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.gl.300.bin')

# Generate embeddings in batches
sentence_embeddings = prepare_embeddings(X, fasttext_model, batch_size=1000)
print("Shape of sentence_embeddings:", sentence_embeddings.shape)

# TF-IDF + chi2 feature selection
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
bow_features = vectorizer.fit_transform(X)
tfidf_features = tfidf_transformer.fit_transform(bow_features)

k_best_selector = SelectKBest(score_func=chi2) 
selected_features = k_best_selector.fit_transform(tfidf_features, y)

# Combine embeddings with TF-IDF features
sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

# Apply Random Undersampling to class 0
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_SEED)
combined_features_resampled, y_resampled = rus.fit_resample(combined_features, y)

param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

print("Shape of combined_features:", combined_features.shape)
print("Shape of y:", y.shape)

X_text_resampled, y_resampled = rus.fit_resample(pd.DataFrame({'text': X}), y)

## Split the dataset into training and test sets, with 70% training and 30% testing
X_train_vec, X_test_vec, y_train, y_test, X_train_texts, X_test_texts = train_test_split(
    combined_features_resampled,
    y_resampled,
    X_text_resampled['text'],
    test_size=0.3,
    random_state=RANDOM_SEED
)

svc = SVC(random_state=RANDOM_SEED)
trained_model = train_model(X_train_vec, y_train, svc, param_grid)

# Save trained model
joblib.dump(trained_model, 'insta_svm_fasttext_tfidf_sampling.pkl')


# ----------- EVALUATION -----------

y_pred = trained_model.predict(X_test_vec)

print("\n Evaluation on Test Set: \n")
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Save classification report to file
report = classification_report(y_test, y_pred)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n\n")
    f.write(report)

print(f" Classification report saved to {report_path}")

# Confusion Matrix
plot_conf_matrix(y_test, y_pred, "Confusion Matrix", matrix_path)

# Save classification report
X_test_raw = pd.DataFrame({
    'text': X_test_texts.values,
    'true_label': y_test.values,
    'predicted_label': y_pred
})

X_test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f" Predictions saved to {output_path}")