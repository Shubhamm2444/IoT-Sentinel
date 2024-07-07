Step 6: Real Data Collection
For demonstration, let's assume you have a CSV file containing real IoT sensor data. You'll need to load and preprocess this data.

import pandas as pd

# Load real IoT data from a CSV file
file_path = 'path_to_your_real_data.csv'
data = pd.read_csv(file_path)

# Assuming 'data' has columns representing sensor readings and 'labels' for anomalies
X = data.drop('labels', axis=1)  # Features
y = data['labels']  # Anomaly labels

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
