Step 5: Anomaly Detection and Evaluation
After training the autoencoder, anomalies can be detected by calculating the reconstruction error. An instance with a high reconstruction error relative to others is likely to be an anomaly.

# Predict on the test set
predictions = autoencoder.predict(X_test_scaled)

# Calculate reconstruction error
mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)

# Define a threshold for anomaly detection
threshold = np.mean(mse) + 3*np.std(mse)  # Example threshold

# Label anomalies based on the threshold
y_pred = np.where(mse > threshold, 1, 0)

# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
