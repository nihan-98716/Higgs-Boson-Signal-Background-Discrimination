# ==============================================================================
# STEP 4: EVALUATING MODEL PERFORMANCE
# ==============================================================================

# This script assumes the variables baseline_model, X_test, and y_test exist
# from running the previous steps.

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import numpy as np

# 1. Make Predictions on the Test Data
print("--- Making predictions on the test set... ---")
y_pred = baseline_model.predict(X_test)
print("Predictions complete.")

# 2. Generate and Print the Classification Report
# This report gives us precision, recall, and f1-score.
print("\n--- Classification Report ---")
report = classification_report(y_test, y_pred, target_names=['Background (0)', 'Signal (1)'])
print(report)

# 3. Generate and Plot the Confusion Matrix
# This shows us exactly where the model made correct and incorrect predictions.
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Background (0)', 'Signal (1)']
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix for Baseline Model', fontsize=16)
plt.show()

# 4. Generate and Plot the ROC Curve and AUC Score
# This shows the model's ability to distinguish between the two classes.
print("\n--- Generating ROC Curve and AUC Score ---")
# Get prediction probabilities for the positive class (Signal)
y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)
print(f"Area Under Curve (AUC): {roc_auc:.4f}")

# Plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\n==============================================================================")
print("Model evaluation is complete.")
print("==============================================================================")
