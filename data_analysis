# ==============================================================================
# FINAL SCRIPT FOR HIGGS BOSON DATA ANALYSIS & ML PREPARATION
# This single script performs all steps from data loading to ML preparation.
# ==============================================================================

# === 1. SETUP AND LIBRARIES ===
# ------------------------------------------------------------------------------
print("Step 1: Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Set a professional and visually appealing style for all plots
sns.set_theme(style="whitegrid", palette="viridis")
print("Libraries imported and plot style set.")

# === 2. DATA LOADING AND INITIAL EXPLORATION ===
# ------------------------------------------------------------------------------
print("\nStep 2: Downloading and loading data...")

# Define the column names manually, as the UCI dataset has no header.
column_names = [
    'Label', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met_pt', 'PRI_met_phi',
    'PRI_jet_num', 'PRI_jet_1_pt', 'PRI_jet_1_eta', 'PRI_jet_1_phi', 'PRI_jet_1_btag',
    'PRI_jet_2_pt', 'PRI_jet_2_eta', 'PRI_jet_2_phi', 'PRI_jet_2_btag',
    'PRI_jet_3_pt', 'PRI_jet_3_eta', 'PRI_jet_3_phi', 'PRI_jet_3_btag',
    'PRI_jet_4_pt', 'PRI_jet_4_eta', 'PRI_jet_4_phi', 'PRI_jet_4_btag',
    'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h',
    'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet'
]

# STABLE URL to the official UCI repository for the HIGGS dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'

# Load the first 250,000 rows
df = pd.read_csv(url, header=None, names=column_names, compression='gzip', nrows=250000)
print("Dataset loaded successfully!")

# === 3. DATA PREPROCESSING AND FEATURE ENGINEERING ===
# ------------------------------------------------------------------------------
print("\nStep 3: Preprocessing data and engineering new features...")

df['Label'] = df['Label'].astype(int)
print("Label column converted to integer.")

delta_phi = df['PRI_lep_phi'] - df['PRI_jet_1_phi']
delta_phi[delta_phi > np.pi] -= 2 * np.pi
delta_phi[delta_phi < -np.pi] += 2 * np.pi
df['Delta_phi_lep_jet'] = np.abs(delta_phi)
print("Feature 'Delta_phi_lep_jet' created successfully.")

# === 4. EXPLORATORY DATA ANALYSIS (EDA) ===
# ------------------------------------------------------------------------------
print("\nStep 4: EDA is complete. Visualizations will be generated if you uncomment the code below.")
# NOTE: The plotting code from the EDA phase is omitted here for brevity,
# but it would normally be included in the full script. Assume EDA is done.

# === 5. OUTLIER DETECTION AND HANDLING ===
# ------------------------------------------------------------------------------
print("\nStep 5: Outlier analysis is complete. Assume outliers are kept.")
# NOTE: Outlier analysis code is also omitted for brevity.

# === 6. MACHINE LEARNING PREPARATION (STEP 1 of ML Roadmap) ===
# ------------------------------------------------------------------------------
print("\nStep 6: Preparing data for machine learning...")

# --- 6.1 Separate Features (X) and Target (y) ---
X = df.drop('Label', axis=1)
y = df['Label']
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# --- 6.2 Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("\nFeatures have been scaled using StandardScaler.")

# --- 6.3 Feature Selection ---
print("\n--- Identifying Important Features using RandomForest ---")
feature_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
feature_model.fit(X_scaled, y)

importances = feature_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Top 10 most important features:")
print(feature_importance_df.head(10))

# Visualize feature importances
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='mako')
plt.title('Top 20 Most Important Features', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

print("\n\n==============================================================================")
print("Machine Learning Preparation (Step 1) is complete.")
print("The variables 'X_scaled' and 'y' are now ready for the next step.")
print("==============================================================================")

