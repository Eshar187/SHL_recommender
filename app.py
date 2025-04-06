import streamlit as st
import pandas as pd
import joblib
import scipy
from sklearn.metrics.pairwise import cosine_similarity


#extracted saved model
vectorizer = joblib.load("tfidf_vectorizer.joblib")
tfidf_matrix = scipy.sparse.load_npz("tfidf_matrix.npz")
df = pd.read_csv("shl_catalog_clean.csv")

def recommend(query, top_n=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Job type', 'Job levels', 'Languages']]

# Streamlit 
st.title("SHL Assessment Recommendation Engine")
user_input = st.text_input("Enter skills or job description:")
if user_input:
    results = recommend(user_input)
    st.write("Top Recommendations:")
    st.dataframe(results)
