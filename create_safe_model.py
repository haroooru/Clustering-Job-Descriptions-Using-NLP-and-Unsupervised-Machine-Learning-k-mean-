import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

# Sample job skills data
skills_data = [
    "Python, Machine Learning, Data Analysis",
    "Java, Spring Boot, Microservices",
    "SEO, Content Writing, Digital Marketing",
    "Excel, Data Entry, Typing",
    "TensorFlow, Deep Learning, Python",
    "Customer Support, Communication, CRM",
    "HTML, CSS, JavaScript, React",
    "AWS, Azure, Cloud Computing",
    "SQL, ETL, Data Warehousing",
    "Project Management, Agile, Scrum"
]

# Create DataFrame
df = pd.DataFrame({"Skills": skills_data})

# Clean the skills text
def clean_skills(skill_text):
    skill_text = skill_text.lower()
    skill_text = re.sub(r'[^a-zA-Z0-9, ]', '', skill_text)
    skill_text = re.sub(r'\s+', ' ', skill_text)
    return skill_text.strip()

df["Cleaned_Skills"] = df["Skills"].apply(clean_skills)

# Vectorization using default tokenizer (safe for joblib)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Cleaned_Skills"])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save both model and vectorizer (Streamlit-safe)
joblib.dump(vectorizer, "skill_vectorizer.pkl")
joblib.dump(kmeans, "job_cluster_model.pkl")

print("✅ Model and vectorizer saved successfully.")
