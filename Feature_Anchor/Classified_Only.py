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
        'results_classified/plots',
        'results_classified/metrics'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the different feature lists from saved files"""
    acetoclastic = pd.read_csv('features/acetoclastic_features.csv')['feature'].tolist()
    hydrogenotrophic = pd.read_csv('features/hydrogenotrophic_features.csv')['feature'].tolist()
    syntrophic = pd.read_csv('features/syntrophic_features.csv')['feature'].tolist()
    return acetoclastic, hydrogenotrophic, syntrophic

def create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse):
    """Create and save performance plot"""
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    min_val = min(min(all_actual), min(all_predictions))
    max_val = max(max(all_actual), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{target} Prediction ({mode})\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'results_classified/plots/results_{target}_{mode}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_linearsvr_cv(data_path, target="ACE-km", mode="ace_only"):
    """
    Run LinearSVR using only classified features based on mode
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("H2-km" or "ACE-km")
    mode : str
        One of ["h2", "ace_only", "ace_all", "ace_conditional"]
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Load feature lists
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    # Select features based on mode
    if mode == "h2":
        selected_features = hydrogenotrophic
    elif mode == "ace_only":
        selected_features = acetoclastic
    elif mode == "ace_all":
        selected_features = acetoclastic + hydrogenotrophic + syntrophic
    elif mode == "ace_conditional":
        selected_features = acetoclastic  # Will be handled differently for each condition
    else:
        raise ValueError("Invalid mode specified")
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    df = df.apply(pd.to_numeric)
    
    X = df[selected_features]
    y = df[target]
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of features used: {len(selected_features)}")

    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if mode == "ace_conditional":
        all_predictions = []
        all_actual = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle low ACE-km cases
            mask_train_low = y_train < 10
            mask_val_low = y_val < 10
            
            if sum(mask_train_low) > 0:
                X_train_low = X_train[mask_train_low]
                scaler_low = StandardScaler()
                X_train_low_scaled = scaler_low.fit_transform(X_train_low)
                
                model_low = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
                model_low.fit(X_train_low_scaled, y_train[mask_train_low])
                
                if sum(mask_val_low) > 0:
                    X_val_low = X_val[mask_val_low]
                    X_val_low_scaled = scaler_low.transform(X_val_low)
                    y_pred_low = model_low.predict(X_val_low_scaled)
                    all_predictions.extend(y_pred_low)
                    all_actual.extend(y_val[mask_val_low])
            
            # Handle high ACE-km cases
            mask_train_high = y_train >= 10
            mask_val_high = y_val >= 10
            
            if sum(mask_train_high) > 0:
                X_train_high = X_train[mask_train_high]
                scaler_high = StandardScaler()
                X_train_high_scaled = scaler_high.fit_transform(X_train_high)
                
                model_high = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
                model_high.fit(X_train_high_scaled, y_train[mask_train_high])
                
                if sum(mask_val_high) > 0:
                    X_val_high = X_val[mask_val_high]
                    X_val_high_scaled = scaler_high.transform(X_val_high)
                    y_pred_high = model_high.predict(X_val_high_scaled)
                    all_predictions.extend(y_pred_high)
                    all_actual.extend(y_val[mask_val_high])
            
            print(f"Fold {fold+1} completed")

        # Calculate metrics
        final_r2 = r2_score(all_actual, all_predictions)
        final_mse = mean_squared_error(all_actual, all_predictions)
        final_rmse = np.sqrt(final_mse)

        # Create plot
        create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse)

        # Save results
        results_df = pd.DataFrame({
            'Actual': all_actual,
            'Predicted': all_predictions
        })
        results_df.to_csv(f'results_classified/metrics/results_{target}_{mode}.csv', index=False)

        return final_r2, final_mse, final_rmse, results_df

    else:
        # Standard scaling and prediction for non-conditional cases
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_val_scaled = scaler.transform(X)
        
        model = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
        model.fit(X_train_scaled, y)
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        final_r2 = r2_score(y, y_pred)
        final_mse = mean_squared_error(y, y_pred)
        final_rmse = np.sqrt(final_mse)

        print(f"\nResults for {target} ({mode} mode):")
        print(f"R² Score: {final_r2:.4f}")
        print(f"MSE: {final_mse:.4f}")
        print(f"RMSE: {final_rmse:.4f}")

        # Create and save plot
        create_performance_plot(y, y_pred, target, mode, final_r2, final_mse)

        # Save results
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        results_df.to_csv(f'results_classified/metrics/results_{target}_{mode}.csv', index=False)

        return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    # Create directories first
    create_directories()
    
    data_path = "../Data/classified_data.csv"
    
    # Dictionary to store results
    all_results = {}
    
    # Run all configurations
    configurations = [
        ("H2-km", "h2"),
        ("ACE-km", "ace_only"),
        ("ACE-km", "ace_all"),
        ("ACE-km", "ace_conditional")
    ]
    
    for target, mode in configurations:
        print(f"\nRunning {target} predictions with {mode} mode...")
        r2, mse, rmse, results = run_linearsvr_cv(data_path, target=target, mode=mode)
        
        all_results[f"{target}_{mode}"] = {
            'Target': target,
            'Mode': mode,
            'R2': r2,
            'MSE': mse,
            'RMSE': rmse
        }
    
    # Save overall results
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = 'Configuration'
    results_df.to_csv('results_classified/metrics/overall_results.csv')
    
    # Print final results
    print("\nFinal Results:")
    print(results_df) 