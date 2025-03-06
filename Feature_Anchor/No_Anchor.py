import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_directories():
    """Create directories for organizing outputs"""
    directories = [
        'results_no_anchor/plots',
        'results_no_anchor/metrics'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_performance_plot(all_actual, all_predictions, target, final_r2, final_mse):
    """Create and save performance plot with detailed information"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(all_actual), min(all_predictions))
    max_val = max(max(all_actual), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add metrics to title
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{target} Prediction\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'results_no_anchor/plots/results_{target}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_linearsvr_cv(data_path, target="ACE-km"):
    """
    Run LinearSVR using all features
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("H2-km" or "ACE-km")
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Define target columns to exclude from features
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    
    # Get features (all columns except target columns)
    features = [col for col in df.columns if col not in target_columns]
    X = df[features]
    y = df[target]
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of features used: {len(features)}")

    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actual = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = LinearSVR(
            random_state=42,
            max_iter=100000,
            tol=1e-4,
            dual=True
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        
        all_predictions.extend(y_pred)
        all_actual.extend(y_val)
        
        print(f"Fold {fold+1} completed")

    # Calculate metrics
    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)

    print(f"\nResults for {target}:")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")

    # Create and save plot
    create_performance_plot(all_actual, all_predictions, target, final_r2, final_mse)

    # Save numerical results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_no_anchor/metrics/results_{target}.csv', index=False)

    return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    # Create directories first
    create_directories()
    
    data_path = "../Data/New_data.csv"
    
    # Dictionary to store results
    all_results = {}
    
    # Run predictions for both targets
    for target in ["H2-km", "ACE-km"]:
        print(f"\nRunning {target} predictions...")
        r2, mse, rmse, results = run_linearsvr_cv(data_path, target=target)
        
        all_results[target] = {
            'R2': r2,
            'MSE': mse,
            'RMSE': rmse
        }
    
    # Save overall results
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = 'Target'
    results_df.to_csv('results_no_anchor/metrics/overall_results.csv')
    
    # Print final results
    print("\nFinal Results:")
    print(results_df) 