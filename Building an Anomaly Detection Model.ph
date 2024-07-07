Step 4: Building an Anomaly Detection Model
For anomaly detection, we can use an Autoencoder, which is an unsupervised learning technique commonly used for learning efficient codings.

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define the Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 2  # Typically smaller than input dimension
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_scaled, X_test_scaled))
