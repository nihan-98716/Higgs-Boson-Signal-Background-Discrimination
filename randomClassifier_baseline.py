# ==============================================================================
# STEP 3: CHOOSING AND TRAINING A BASELINE MODEL
# ==============================================================================

# This script assumes that the variables X_train and y_train have been created
# by running the 'step_2_split_data.py' script.

from sklearn.ensemble import RandomForestClassifier
import time

# 1. Initialize the Model
# We choose RandomForestClassifier as our baseline.
# n_estimators=100 means the model will build 100 "decision trees".
# random_state=42 ensures our results are reproducible.
# n_jobs=-1 tells the model to use all available CPU cores, which speeds up training.
baseline_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 2. Train the Model
# The .fit() method is where the model "learns" from the training data.
print("--- Starting Model Training ---")
print(f"Model: {type(baseline_model).__name__}")

start_time = time.time()
baseline_model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds.")

print("\n==============================================================================")
print("Baseline model has been trained and is ready for evaluation.")
print("Variable created: baseline_model")
print("==============================================================================")
