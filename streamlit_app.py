
import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("job_cluster_model.pkl")
vectorizer = joblib.load("skill_vectorizer.pkl")

# Function to clean and process skills input
def clean_skills(skill_text):
    skill_text = skill_text.lower()
    skill_text = re.sub(r'[^a-zA-Z0-9, ]', '', skill_text)
    skill_text = re.sub(r'\s+', ' ', skill_text)
    return skill_text.strip()

# Streamlit App UI
st.title("Job Clustering Based on Required Skills")
st.write("This app classifies a job description or skill list into a pre-learned cluster.")

user_input = st.text_area("Enter job description or skills (comma-separated):", "")

if st.button("Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter a job description or list of skills.")
    else:
        cleaned_input = clean_skills(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Predicted Cluster: {prediction}")

        # Optionally: load clustered dataset and show similar jobs
        try:
            df = pd.read_csv("clustered_job_postings.csv")
            matched_jobs = df[df["Cluster"] == prediction][["Title", "Company", "Skills"]].head(5)
            st.subheader("Sample Jobs from This Cluster:")
            st.dataframe(matched_jobs)
        except:
            st.info("Clustered job CSV not found. Upload 'clustered_job_postings.csv' to show sample jobs.")
