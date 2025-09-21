# ==============================================================================
# STEP 8 (STATE-OF-THE-ART ATTEMPT): NEURAL NETWORK FOR MAXIMUM ACCURACY
# ==============================================================================

# This script assumes the variables X_train, y_train, X_test, and y_test exist.

# --- Import necessary libraries for Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import time

print("--- Starting Neural Network Training for Maximum Accuracy ---")

# === Part 1: Define the Neural Network Architecture ===
# ------------------------------------------------------------------------------
# We will build a Sequential model, which is a simple stack of layers.

neural_network = Sequential([
    # Input Layer: Must match the number of features in our training data
    # 'relu' (Rectified Linear Unit) is a standard, effective activation function.
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),

    # Dropout Layer 1: A regularization technique to prevent overfitting.
    # It randomly sets 30% of the input units to 0 at each update during training.
    Dropout(0.3),

    # Hidden Layer 2: Another dense layer to learn more complex patterns.
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Hidden Layer 3: A final hidden layer.
    Dense(32, activation='relu'),
    Dropout(0.3),

    # Output Layer: A single neuron with a 'sigmoid' activation function.
    # Sigmoid is perfect for binary classification as it outputs a probability (0 to 1).
    Dense(1, activation='sigmoid')
])

# === Part 2: Compile the Model ===
# ------------------------------------------------------------------------------
# Here we configure the model for training.
neural_network.compile(
    optimizer='adam',                  # Adam is a robust and popular optimization algorithm.
    loss='binary_crossentropy',        # The standard loss function for binary classification.
    metrics=['accuracy']               # We want to monitor accuracy during training.
)

# Print a summary of the model's architecture
print("\n--- Model Architecture Summary ---")
neural_network.summary()

# === Part 3: Train the Model with Early Stopping ===
# ------------------------------------------------------------------------------
# We will use an 'EarlyStopping' callback to prevent overfitting. It will monitor
# the validation accuracy and stop training if it doesn't improve for 10 epochs.
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

print("\n--- Starting Model Training ---")
start_time = time.time()

history = neural_network.fit(
    X_train,
    y_train,
    epochs=100,                      # Train for up to 100 epochs (passes through the data)
    batch_size=512,                  # Process data in batches of 512 samples
    validation_split=0.2,            # Use 20% of the training data for validation
    callbacks=[early_stopping],      # Apply the early stopping rule
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time
print(f"\nNeural network training completed in {training_time / 60:.2f} minutes.")


# === Part 4: Evaluating the Final Neural Network Model ===
# ------------------------------------------------------------------------------
print("\n--- Evaluating the Final Neural Network ---")

# Make predictions on the test set. The output will be probabilities.
y_pred_proba_nn = neural_network.predict(X_test)
# Convert probabilities to binary class labels (0 or 1) using a 0.5 threshold.
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)

# Print the final accuracy score
final_accuracy = accuracy_score(y_test, y_pred_nn)
print(f"\nFinal Neural Network Accuracy Score: {final_accuracy:.4f}")

# Print the full classification report
print("\n--- Classification Report for Neural Network Model ---")
report_final = classification_report(y_test, y_pred_nn, target_names=['Background (0)', 'Signal (1)'])
print(report_final)

print("\n==============================================================================")
print("Final accuracy optimization with a Neural Network is complete.")
print(f"The highest accuracy achieved is {final_accuracy*100:.2f}%.")
print("==============================================================================")
