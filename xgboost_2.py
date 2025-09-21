# @title
# ==============================================================================
# STEP 5: ADVANCED TUNING FOR MAXIMUM PRECISION WITH XGBOOST
# ==============================================================================

# This script assumes the variables X_train, y_train, X_test, and y_test exist
# from running the previous steps.

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import time

print("--- Starting Advanced Hyperparameter Tuning with XGBoost ---")

# 1. Define the Hyperparameter Grid for XGBoost
# We are using a more powerful model and a wider range of parameters to search.
# 'scale_pos_weight' is crucial for imbalanced datasets like this one.
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

# 2. Initialize the RandomizedSearchCV with XGBoost
# We are now using XGBClassifier, a state-of-the-art model.
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# The most important change: we set scoring='precision'.
# This tells the search to find the hyperparameter combination that
# results in the HIGHEST POSSIBLE PRECISION for the signal class.
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,  # We'll try more combinations for a more thorough search
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='precision' # Optimizing specifically for precision
)

# 3. Run the Search
print("Searching for the best hyperparameters to maximize precision...")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

search_time = end_time - start_time
print(f"\nRandomized search completed in {search_time / 60:.2f} minutes.")

# 4. Get the Best Model and Evaluate It
print("\n--- Evaluating the Optimized Model ---")

# Print the best parameters found
print("Best Hyperparameters for High Precision:")
print(random_search.best_params_)

# The best model is automatically refitted on the full training data
best_model = random_search.best_estimator_

# Make predictions with the best model
y_pred_tuned = best_model.predict(X_test)

# Print the new classification report
print("\n--- Classification Report for Precision-Tuned Model ---")
report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Background (0)', 'Signal (1)'])
print(report_tuned)

# Print the new AUC score for comparison
y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
print(f"Tuned Model Area Under Curve (AUC): {roc_auc_tuned:.4f}")

print("\n==============================================================================")
print("Hyperparameter tuning for maximum precision is complete.")
print("The model has been optimized to reduce false positives (incorrectly identified signals).")
print("==============================================================================")

