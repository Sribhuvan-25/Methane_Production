import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ----------------------------
# LOAD DATA
# ----------------------------

# Load feature table
features = pd.read_csv('feature-table.tsv', sep='\t', skiprows=1, index_col=0).T

# Load metadata
metadata = pd.read_csv('sample-metadata.tsv', sep='\t', index_col='sample-id')

# Ensure sample IDs match
features.index = features.index.astype(str)
metadata.index = metadata.index.astype(str)

# Merge features and metadata
data = features.join(metadata, how='inner')

# ----------------------------
# PREPARE DATA
# ----------------------------

# Define target and groups
target_column = 'ACE-km'  # Change to 'H2-km' if needed
group_column = 'Group'

X = data.drop(columns=[target_column, group_column])
y = data[target_column]
groups = data[group_column]

# Convert categorical features if any (should already be one-hot encoded)
# X = pd.get_dummies(X)

# ----------------------------
# PERFORM LOGO-CV WITH RFE
# ----------------------------

logo = LeaveOneGroupOut()
rf = RandomForestRegressor(random_state=42)

# Recursive Feature Elimination with Cross-Validation
rfe = RFECV(
    estimator=rf,
    step=0.05,  # Adjust step size if needed
    cv=logo.split(X, y, groups),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)

rfe.fit(X, y)

# ----------------------------
# EVALUATE MODEL
# ----------------------------

# Predictions
y_pred = rfe.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Important features
important_features = X.columns[rfe.support_]
print("Selected Features:")
print(important_features)

# ----------------------------
# SAVE RESULTS
# ----------------------------

# Save the model
joblib.dump(rfe, 'trained_regressor.joblib')

# Save feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rfe.estimator_.feature_importances_
})
feature_importance.to_csv('feature_importance.csv', index=False)

# Save predictions
predictions = pd.DataFrame({
    'SampleID': X.index,
    'Observed': y,
    'Predicted': y_pred
})
predictions.to_csv('predictions.csv', index=False)

# Save selected features
selected_features = pd.DataFrame({
    'SelectedFeatures': important_features
})
selected_features.to_csv('selected_features.csv', index=False)
