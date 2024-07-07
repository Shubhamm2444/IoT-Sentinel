Step 8: Experimenting with Different Models and Hyperparameters
Now, let's explore using different anomaly detection models and fine-tuning their hyperparameters.

1: Example: Variational Autoencoder (VAE)

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Define the Variational Autoencoder (VAE) model
def create_vae(input_dim):
    original_dim = input_dim
    
    # Encoder
    inputs = Input(shape=(original_dim,))
    z_mean = Dense(8, activation='relu')(inputs)
    z_log_var = Dense(8, activation='relu')(inputs)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 8), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(8,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(8, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    vae = Model(inputs, x_decoded_mean)

    # Loss function: reconstruction loss + KL divergence regularization
    reconstruction_loss = K.sum(K.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae

# Assuming input_dim is the dimension of your data (number of features)
input_dim = X_train.shape[1]
vae = create_vae(input_dim)

# Train the VAE model
vae.fit(X_train_scaled, epochs=50, batch_size=32, validation_data=(X_test_scaled, None))

# Evaluate the VAE model
predictions = vae.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)
threshold = np.mean(mse) + 3*np.std(mse)
y_pred = np.where(mse > threshold, 1, 0)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix (VAE):")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))



2:Example: Isolation Forest

from sklearn.ensemble import IsolationForest

# Create an Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination as needed

# Fit the model
isolation_forest.fit(X_train_scaled)

# Predict anomalies
y_pred_if = isolation_forest.predict(X_test_scaled)
y_pred_if = np.where(y_pred_if == -1, 1, 0)  # Convert predictions to binary (anomaly: 1, normal: 0)

print("Confusion Matrix (Isolation Forest):")
print(confusion_matrix(y_test, y_pred_if))
print("Accuracy:", accuracy_score(y_test, y_pred_if))


