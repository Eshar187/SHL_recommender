import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy

# Load and clean data
df = pd.read_csv("shlpage_extraction.csv")
df["Assessment length (minutes)"] = df["Assessment length"].str.extract(r'=\\s*(\\d+)').astype(float)
df.dropna(axis=1, how='all', inplace=True)
df['Job type'].fillna('', inplace=True)
df['Description'].fillna('', inplace=True)
df['Job levels'].fillna('', inplace=True)
df['text_blob'] = df['Job type'] + ' ' + df['Description'] + ' ' + df['Job levels']

# TF-IDF training
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['text_blob'])

# Save vectorizer and matrix
# joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
# scipy.sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
# df.to_csv("shl_catalog_clean.csv", index=False)  # Save cleaned data

