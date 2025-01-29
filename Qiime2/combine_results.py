import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the feature table and metadata
feature_table = pd.read_csv('feature-table.tsv', index_col=0)
metadata = pd.read_csv('sample-metadata.tsv', sep='\t')

# Extract features and target variable
X = feature_table.values
y = metadata['ACE-km'].values

# Initialize model and cross-validator
model = RandomForestRegressor()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Collect predictions
actual = []
predicted = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test fold
    y_pred = model.predict(X_test)

    # Store results
    actual.extend(y_test)
    predicted.extend(y_pred)

# Save results to a CSV
results = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
results.to_csv('cross_validation_predictions.csv', index=False)

# Create the plot
plt.scatter(results['Actual'], results['Predicted'], alpha=0.7)
plt.xlabel('Actual ACE-km')
plt.ylabel('Predicted ACE-km')
plt.title('Predicted vs Actual ACE-km (All Folds)')
plt.grid(True)
plt.show()
