Step 2: Dataset Simulation
Since creating a real IoT dataset can be complex, we'll simulate a simple dataset representing sensor readings from IoT devices. Let's generate synthetic data for demonstration purposes.

import numpy as np
import pandas as pd
# Simulate IoT data
np.random.seed(42)
n_samples = 1000
normal_data = pd.DataFrame({
    'sensor1': np.random.normal(0, 1, n_samples),
    'sensor2': np.random.normal(0, 1, n_samples)
})
# Introduce anomalies for demonstration
anomalies = pd.DataFrame({
    'sensor1': np.random.normal(5, 1, 20),
    'sensor2': np.random.normal(-5, 1, 20)
})
data = pd.concat([normal_data, anomalies]).reset_index(drop=True)
labels = np.array([0]*len(normal_data) + [1]*len(anomalies))

# Visualize the data (optional)
import matplotlib.pyplot as plt
plt.scatter(data['sensor1'], data['sensor2'], c=labels, cmap='coolwarm')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.title('Simulated IoT Data with Anomalies')
plt.show()
