
# Job Clustering using Skill-Based Analysis (KMeans + TF-IDF)

This project uses web scraping and machine learning to cluster job listings based on required skills. It helps in identifying similar job roles (e.g., AI/ML, Data Science) from raw job descriptions.

## Features

- Web Scraping: Extracts job listings from online sources
- Preprocessing: Cleans and tokenizes job skills
- TF-IDF Vectorization: Converts text to feature vectors
- KMeans Clustering: Groups similar jobs into clusters
- Model Saving: Stores trained model and vectorizer using `joblib`
- Visualization: Elbow method to select optimal number of clusters
- Alerting System: Predicts the cluster for new job inputs

## Libraries Used

- requests, beautifulsoup4 - for scraping
- pandas, numpy - for data processing
- scikit-learn - for TF-IDF, clustering, and model handling
- matplotlib - for plotting elbow graph
- joblib - for saving model/vectorizer

## How to Run

1. Clone the repo

   ```bash
   git clone https://github.com/your-username/job-clustering.git
   cd job-clustering
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook

   Open the Jupyter Notebook (`job_filtering_predictive_analytics.ipynb`) and run all cells.

## Sample Output

- Clustered job listings based on skills
- Cluster prediction for new jobs
- Model saved: `job_cluster_model.pkl`
- Vectorizer saved: `skill_vectorizer.pkl`

## Example: Predict New Job Cluster

```python
new_job = ["Looking for an experienced backend developer with Python, Flask, and PostgreSQL"]
new_vec = vectorizer.transform(new_job)
predicted_cluster = kmeans.predict(new_vec)[0]
print(f"Predicted Cluster: {predicted_cluster}")
```

## Output Files

| File Name                | Description                            |
|--------------------------|----------------------------------------|
| clustered_job_postings.csv | Final job listings with cluster labels |
| job_cluster_model.pkl     | Trained KMeans model                |
| skill_vectorizer.pkl      | TF-IDF Vectorizer                   |

## Future Improvements

- Automate alerts via email or Slack
- Use DBSCAN or HDBSCAN for better cluster shapes
- Deploy as a web app (Streamlit/Flask)

## Author

- Your Name  
- GitHub: https://github.com/haroooru

## License

This project is for academic and educational use only.
