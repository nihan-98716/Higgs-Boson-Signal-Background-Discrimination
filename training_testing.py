# ==============================================================================
# STEP 2: SPLITTING THE DATA INTO TRAINING AND TESTING SETS
# ==============================================================================

# This script assumes that the variables 'X_scaled' and 'y' have been created
# by running the main 'higgs_boson_analysis.py' script.

from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets.
# stratify=y ensures that the proportion of signal/background events is the same
# in both the training and testing sets, which is crucial for accurate evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Print the shapes of the resulting datasets to confirm the split
print("\n--- Data Splitting Complete ---")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

print("\n==============================================================================")
print("Data is now split and ready for model training.")
print("Variables created: X_train, X_test, y_train, y_test")
print("==============================================================================")

