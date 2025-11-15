import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
n_samples = 500
income = np.random.randint(30000, 200000, n_samples)
savings = np.random.randint(5000, 100000, n_samples)
expenses = np.random.randint(10000, 150000, n_samples)
age = np.random.randint(20, 70, n_samples)

data = pd.DataFrame({
    'Income': income,
    'Savings': savings,
    'Expenses': expenses,
    'Age': age
})
## scaling the value 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(data_scaled)

data['RiskScore'] = kmeans.labels_ + 1  

joblib.dump(kmeans, "kmeans_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def predict_risk(user_input):
    """Predicts the risk score based on user financial data."""
    user_input_scaled = scaler.transform([user_input])
    predicted_cluster = kmeans.predict(user_input_scaled)[0] 
    return predicted_cluster