import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# ------Data Loading & Cleaning------
df = pd.read_csv(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\forest_cover_prediction\train.csv")
print("Dataset Loaded")

# Drop ID column (not useful for predictions)
df.drop(columns=["Id"], inplace=True)
print("ID Column Dropped")

# Check for missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()
print("Data Cleaning Done")

# ------Feature Engineering------
print("\nFeature Engineering...")

# Define numerical and categorical features
num_features = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", 
                "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", 
                "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

wilderness_cols = [col for col in df.columns if "Wilderness_Area" in col]
soil_cols = [col for col in df.columns if "Soil_Type" in col]

# No need for encoding as these are already binary features

print("Feature Engineering Done")

# ------Splitting the data into train and test sets------
# Define features and target
X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data Split: Train Size = {X_train.shape[0]}, Test Size = {X_test.shape[0]}")

# ------Feature Scaling------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------Model Training------
print("\nTraining Random Forest Model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluation
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------Hyperparameter Tuning------
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\nBest Hyperparameters:", grid_search.best_params_)
print(f"Best Accuracy (Cross-validation): {grid_search.best_score_:.4f}")

# Train Final Model
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train_scaled, y_train)

# Evaluate Optimized Model
y_pred_best = best_rf.predict(X_test_scaled)
print("\nOptimized Model Performance:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Save Optimized Model & Scaler
joblib.dump(best_rf, "best_rf_forest_cover_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Optimized Model Saved!")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Optimized Model
model = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\forest_cover.pro\best_rf_forest_cover_model.joblib")

# Get Feature Importances
feature_importances = model.feature_importances_
feature_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"
] + [f"Soil_Type{i}" for i in range(1, 41)]

# Sort feature importances
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Plot Feature Importances
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importances[:15], y=sorted_features[:15], palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances")
plt.show()

# ------Test Predictions------
import random

# Select 5 random test samples
random_indices = random.sample(range(len(X_test_scaled)), 5)
X_sample = X_test_scaled[random_indices]
y_actual = y_test.iloc[random_indices]

# Predict using the optimized model
y_pred_sample = best_rf.predict(X_sample)

# Display results
print("\nSample Predictions:")
for i in range(len(y_actual)):
    print(f"Actual Cover Type: {y_actual.iloc[i]}, Predicted: {y_pred_sample[i]}")

import joblib
feature_names = [...]  # List of feature names used in training
joblib.dump(feature_names, "feature_names.joblib")
print("Feature names saved!")