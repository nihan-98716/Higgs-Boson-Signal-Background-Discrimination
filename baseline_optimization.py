# @title
# ==============================================================================
# STEP 5: HYPERPARAMETER TUNING AND OPTIMIZATION
# ==============================================================================

# This script assumes the variables X_train, y_train, X_test, and y_test exist
# from running the previous steps.

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import time

print("--- Starting Hyperparameter Tuning with RandomizedSearchCV ---")

# 1. Define the Hyperparameter Grid
# These are the settings we want to test.
# We'll provide a range of values for the most important parameters.
param_dist = {
    'n_estimators': [100, 200, 300],            # Number of trees in the forest
    'max_features': ['sqrt', 'log2'],             # Number of features to consider at every split
    'max_depth': [10, 20, 30, None],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],               # Minimum number of samples required at a leaf node
    'bootstrap': [True]                         # Method of selecting samples for training each tree
}

# 2. Initialize the RandomizedSearchCV
# This will search for the best hyperparameters using cross-validation.
# We're creating a new classifier instance for the search.
rf = RandomForestClassifier(random_state=42)

# n_iter=20: It will try 20 different random combinations of parameters.
# cv=3: It will use 3-fold cross-validation.
# scoring='roc_auc': The goal is to find the combination that maximizes the AUC score.
# n_jobs=-1: Use all available CPU cores to speed up the search.
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)

# 3. Run the Search
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

search_time = end_time - start_time
print(f"\nRandomized search completed in {search_time / 60:.2f} minutes.")

# 4. Get the Best Model and Evaluate It
print("\n--- Evaluating the Optimized Model ---")

# Print the best parameters found
print("Best Hyperparameters found:")
print(random_search.best_params_)

# The best model is automatically refitted on the full training data
best_model = random_search.best_estimator_

# Make predictions with the best model
y_pred_tuned = best_model.predict(X_test)

# Print the new classification report
print("\n--- Classification Report for Tuned Model ---")
report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Background (0)', 'Signal (1)'])
print(report_tuned)

# Print the new AUC score
y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
print(f"Tuned Model Area Under Curve (AUC): {roc_auc_tuned:.4f}")

print("\n==============================================================================")
print("Hyperparameter tuning is complete.")
print("Compare the new Classification Report and AUC score with the baseline model.")
print("==============================================================================")
