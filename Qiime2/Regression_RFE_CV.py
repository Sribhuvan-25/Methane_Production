# import pandas as pd
# import numpy as np
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFECV
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib

# # ----------------------------
# # LOAD DATA
# # ----------------------------

# # Load feature table, transpose so samples are rows and features are columns
# features = pd.read_csv('feature-table.tsv', sep='\t', skiprows=1, index_col=0).T

# # Reset index to create a 'sample-id' column with sequential values
# features = features.reset_index(drop=True)
# features['sample-id'] = range(1, len(features) + 1)

# # Ensure 'sample-id' is string and strip whitespace in features
# features['sample-id'] = features['sample-id'].astype(str).str.strip()

# # Load metadata
# metadata = pd.read_csv('sample-metadata.tsv', sep='\t')

# # Ensure 'sample-id' is string and strip whitespace in metadata
# metadata['sample-id'] = metadata['sample-id'].astype(str).str.strip()

# print("Features sample IDs:", features['sample-id'].tolist())
# print("Metadata sample IDs:", metadata['sample-id'].tolist())

# # Merge features and metadata on 'sample-id'
# data = pd.merge(features, metadata, on='sample-id', how='inner')

# print(f"Features shape (before merge): {features.shape}")
# print(f"Metadata shape: {metadata.shape}")
# print(f"Merged data shape: {data.shape}")

# if data.empty:
#     raise ValueError("Merged data has no samples. Check 'sample-id' alignment between features and metadata.")

# # Set 'sample-id' as index after merging
# data.set_index('sample-id', inplace=True)

# # ----------------------------
# # PREPARE DATA
# # ----------------------------

# # Define target and groups
# target_column = 'ACE-km'  # Change to 'H2-km' if needed
# group_column = 'Group'

# # Verify target and group columns exist
# if target_column not in data.columns or group_column not in data.columns:
#     raise KeyError(f"Missing required columns: {target_column} or {group_column}")

# X = data.drop(columns=[target_column, group_column])
# y = data[target_column]
# groups = data[group_column]

# # Debug X and y
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")
# if X.shape[0] == 0 or y.shape[0] == 0:
#     raise ValueError("X or y has no samples. Verify the input data.")

# # ----------------------------
# # PERFORM LOGO-CV WITH RFE
# # ----------------------------

# logo = LeaveOneGroupOut()
# rf = RandomForestRegressor(random_state=42)

# # Initialize arrays to store results
# all_y_true = []
# all_y_pred = []

# # Initialize a DataFrame to store importances for all features across folds
# all_feature_importances = pd.DataFrame(0, index=X.columns, columns=["ImportanceSum"])

# # Perform LOGO-CV
# for train_idx, test_idx in logo.split(X, y, groups):
#     # Split the data into training and testing
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
#     # Apply Recursive Feature Elimination on training data only
#     rfe = RFECV(
#         estimator=rf,
#         step=0.05,  # Adjust step size if needed
#         cv=5,  # Inner CV for RFE
#         scoring='neg_mean_squared_error',
#         n_jobs=-1,
#     )
#     rfe.fit(X_train, y_train)
    
#     # Predict on the test set
#     y_pred = rfe.predict(X_test)
    
#     # Append predictions and true values for later evaluation
#     all_y_true.extend(y_test)
#     all_y_pred.extend(y_pred)
    
#     # Get feature importances for selected features
#     fold_importances = pd.Series(
#         rfe.estimator_.feature_importances_,
#         index=X_train.columns[rfe.support_]
#     )
    
#     # Align feature importances with all features and update the total
#     fold_importances = fold_importances.reindex(all_feature_importances.index, fill_value=0)
#     all_feature_importances["ImportanceSum"] += fold_importances

# # Compute mean feature importances across folds
# all_feature_importances["ImportanceMean"] = (
#     all_feature_importances["ImportanceSum"] / len(groups.unique())
# )

# # Sort feature importances by the mean importance
# all_feature_importances = all_feature_importances.sort_values(by="ImportanceMean", ascending=False)

# # ----------------------------
# # EVALUATE MODEL
# # ----------------------------

# # Convert predictions to numpy arrays
# all_y_true = np.array(all_y_true)
# all_y_pred = np.array(all_y_pred)

# # Metrics
# mse = mean_squared_error(all_y_true, all_y_pred)
# r2 = r2_score(all_y_true, all_y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")

# # Save predictions
# predictions = pd.DataFrame({
#     'Observed': all_y_true,
#     'Predicted': all_y_pred
# })
# predictions.to_csv('predictions.csv', index=False)

# # ----------------------------
# # SAVE FEATURE IMPORTANCE
# # ----------------------------

# # Save feature importance using only selected features
# feature_importance = all_feature_importances.reset_index().rename(
#     columns={"index": "Feature"}
# )
# feature_importance.to_csv('feature_importance.csv', index=False)

# # Save selected features
# selected_features_df = feature_importance[["Feature", "ImportanceMean"]]
# selected_features_df.to_csv('selected_features.csv', index=False)

# # ----------------------------
# # SAVE THE MODEL
# # ----------------------------

# # Save the last trained RFE model for reproducibility
# rfe.cv = None  # Remove non-pickleable attribute
# joblib.dump(rfe, 'trained_regressor.joblib')


import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# LOAD DATA
# ----------------------------

# Load feature table, transpose so samples are rows and features are columns
features = pd.read_csv('feature-table.tsv', sep='\t', skiprows=1, index_col=0).T

# Reset index to create a 'sample-id' column with sequential values
features = features.reset_index(drop=True)
features['sample-id'] = range(1, len(features) + 1)

# Ensure 'sample-id' is string and strip whitespace in features
features['sample-id'] = features['sample-id'].astype(str).str.strip()

# Load metadata
metadata = pd.read_csv('sample-metadata.tsv', sep='\t')

# Ensure 'sample-id' is string and strip whitespace in metadata
metadata['sample-id'] = metadata['sample-id'].astype(str).str.strip()

print("Features sample IDs:", features['sample-id'].tolist())
print("Metadata sample IDs:", metadata['sample-id'].tolist())

# Merge features and metadata on 'sample-id'
data = pd.merge(features, metadata, on='sample-id', how='inner')

print(f"Features shape (before merge): {features.shape}")
print(f"Metadata shape: {metadata.shape}")
print(f"Merged data shape: {data.shape}")

if data.empty:
    raise ValueError("Merged data has no samples. Check 'sample-id' alignment between features and metadata.")

# Set 'sample-id' as index after merging
data.set_index('sample-id', inplace=True)

# ----------------------------
# PREPARE DATA
# ----------------------------

# Define target and groups
target_column = 'ACE-km'  # Change to 'H2-km' if needed
group_column = 'Group'

# Verify target and group columns exist
if target_column not in data.columns or group_column not in data.columns:
    raise KeyError(f"Missing required columns: {target_column} or {group_column}")

X = data.drop(columns=[target_column, group_column])
y = data[target_column]
groups = data[group_column]

# Debug X and y
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("X or y has no samples. Verify the input data.")

# ----------------------------
# PERFORM LOGO-CV WITH RFE
# ----------------------------

logo = LeaveOneGroupOut()
rf = RandomForestRegressor(random_state=42)

# Initialize arrays to store results
all_y_true = []
all_y_pred = []

# Initialize a DataFrame to store importances for all features across folds
all_feature_importances = pd.DataFrame(0, index=X.columns, columns=["ImportanceSum"])

# Perform LOGO-CV
for train_idx, test_idx in logo.split(X, y, groups):
    # Split the data into training and testing
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Apply Recursive Feature Elimination on training data only
    rfe = RFECV(
        estimator=rf,
        step=0.05,  # Adjust step size if needed
        cv=5,  # Inner CV for RFE
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    rfe.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rfe.predict(X_test)
    
    # Append predictions and true values for later evaluation
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Get feature importances for selected features
    fold_importances = pd.Series(
        rfe.estimator_.feature_importances_,
        index=X_train.columns[rfe.support_]
    )
    
    # Align feature importances with all features and update the total
    fold_importances = fold_importances.reindex(all_feature_importances.index, fill_value=0)
    all_feature_importances["ImportanceSum"] += fold_importances

# Compute mean feature importances across folds
all_feature_importances["ImportanceMean"] = (
    all_feature_importances["ImportanceSum"] / len(groups.unique())
)

# Sort feature importances by the mean importance
all_feature_importances = all_feature_importances.sort_values(by="ImportanceMean", ascending=False)

# ----------------------------
# EVALUATE MODEL
# ----------------------------

# Convert predictions to numpy arrays
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Metrics
mse = mean_squared_error(all_y_true, all_y_pred)
r2 = r2_score(all_y_true, all_y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save predictions
predictions = pd.DataFrame({
    'Observed': all_y_true,
    'Predicted': all_y_pred
})
predictions.to_csv('predictions.csv', index=False)

# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(all_y_true, all_y_pred, alpha=0.7)
plt.plot([all_y_true.min(), all_y_true.max()], [all_y_true.min(), all_y_true.max()], 'r--', lw=2)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.savefig('actual_vs_predicted_plot.png')
plt.close()

# ----------------------------
# SAVE FEATURE IMPORTANCE
# ----------------------------

# Save feature importance using only selected features
feature_importance = all_feature_importances.reset_index().rename(
    columns={"index": "Feature"}
)
feature_importance.to_csv('feature_importance.csv', index=False)

# Save selected features
selected_features_df = feature_importance[["Feature", "ImportanceMean"]]
selected_features_df.to_csv('selected_features.csv', index=False)

# ----------------------------
# SAVE THE MODEL
# ----------------------------

# Save the last trained RFE model for reproducibility
rfe.cv = None  # Remove non-pickleable attribute
joblib.dump(rfe, 'trained_regressor.joblib')
