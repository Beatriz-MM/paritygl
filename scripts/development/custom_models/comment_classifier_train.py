# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 05/06/2025
# Description: This script trains a linear SVM model to classify Instagram comments as negative (label=1) or not (label=0).  
# It combines a set of confirmed negative examples with a dataset, applies undersampling, uses FastText embeddings (cc.gl.300.bin), 
# and evaluates the model performance.
# Python version: 3.10.12

import os
import glob
import pandas as pd
import numpy as np
import joblib
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


RANDOM_SEED = 42

# Paths to input datasets
negative_dataset_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/negative_dataset.csv" 
csv_folder_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/CSV_DATA/"
csv_pattern = "csv_gl_comments_*.csv"  # pattern to match all CSVs

# Paths to output files
report_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/classification_comments_report.txt"
matrix_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/confusion_matrix_training.png"
output_path = '/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/result_predictions_training.csv'

# ---------- FUNCTIONS ----------

def load_dataset():

    # Load confirmed negative comments and label them as 1
    neg_df = pd.read_csv(negative_dataset_path, usecols=['text'])
    neg_df['label'] = 1  

    # Load all datasets (assumed positive or neutral) and label as 0
    csv_files = glob.glob(os.path.join(csv_folder_path, csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder_path} with pattern {csv_pattern}")
    print(f"Found {len(csv_files)} files matching pattern.")

    list_df = []
    for file in csv_files:
        df = pd.read_csv(file, usecols=['text'])
        df['label'] = 0
        list_df.append(df)

    dataset_df = pd.concat(list_df, ignore_index=True)

    # Concatenate both to build the full corpus
    corpus_df = pd.concat([dataset_df, neg_df], ignore_index=True)

    # Drop duplicate texts (if any)
    corpus_df = corpus_df.drop_duplicates(subset='text')

    # Shuffle the full corpus
    corpus_df = corpus_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Label distribution in full corpus:")
    print(corpus_df['label'].value_counts())

    return corpus_df


def get_fasttext_model():
    fasttext.util.download_model('gl', if_exists='ignore')
    return fasttext.load_model('cc.gl.300.bin')


def compute_embeddings(texts, model):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    embeddings = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        vectors = [model.get_word_vector(token) for token in tokens]
        if not vectors:
            embeddings.append(np.zeros(300))
        else:
            embeddings.append(np.mean(vectors, axis=0))
    return np.array(embeddings)


def plot_conf_matrix(y_true, y_pred, title, save_path=None):
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

# ---------- MAIN ----------

# Load and label data
df = load_dataset()

X_raw = df['text']
y = df['label']

# Apply undersampling to balance the classes
rus = RandomUnderSampler(random_state=RANDOM_SEED)
X_bal, y_bal = rus.fit_resample(pd.DataFrame(X_raw), y)
X_bal = X_bal['text'] # Convert to Series

# Generate FastText embeddings
ft_model = get_fasttext_model()
X_embed = compute_embeddings(X_bal, ft_model)

# Store original comment texts to match predictions later
X_texts = X_bal.reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test, X_texts_train, X_texts_test = train_test_split(
    X_embed, y_bal, X_texts, test_size=0.3, random_state=RANDOM_SEED
)

# Train SVM
clf = SVC(kernel='linear', random_state=RANDOM_SEED)
clf.fit(X_train, y_train)

# Save trained model
joblib.dump(clf, 'model_SVM_instagram_fasttext.pkl')


# ----------- EVALUATION -----------

y_pred = clf.predict(X_test)

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
    'text': X_texts_test,
    'true_label': y_test.values,
    'predicted_label': y_pred
})

X_test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f" Predictions saved to {output_path}")