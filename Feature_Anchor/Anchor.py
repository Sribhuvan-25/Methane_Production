import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create necessary directories
def create_directories():
    """Create directories for organizing outputs"""
    directories = [
        'data',
        'features',
        'results/plots',
        'results/metrics'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the different feature lists from saved files"""
    acetoclastic = pd.read_csv('features/acetoclastic_features.csv')['feature'].tolist()
    hydrogenotrophic = pd.read_csv('features/hydrogenotrophic_features.csv')['feature'].tolist()
    syntrophic = pd.read_csv('features/syntrophic_features.csv')['feature'].tolist()
    return acetoclastic, hydrogenotrophic, syntrophic

def select_features_with_rfe(X, y, anchored_features, n_features_to_select=10):
    """
    Perform RFE while keeping anchored features
    """
    # Ensure X and y have the same number of samples
    if len(X) != len(y):
        raise ValueError(f"X and y must have same number of samples. Got X: {len(X)}, y: {len(y)}")
    
    # Separate anchored and non-anchored features
    non_anchored_features = [col for col in X.columns if col not in anchored_features]
    X_non_anchored = X[non_anchored_features]
    
    # Calculate how many additional features to select
    n_additional = max(0, n_features_to_select - len(anchored_features))
    
    if n_additional > 0 and len(non_anchored_features) > 0:
        # Perform RFE on non-anchored features
        estimator = LinearSVR(random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=n_additional)
        rfe.fit(X_non_anchored, y)
        
        # Get selected feature names
        selected_additional = [f for f, selected in zip(non_anchored_features, rfe.support_) if selected]
    else:
        selected_additional = []
    
    # Combine anchored and selected features
    final_features = anchored_features + selected_additional
    
    print(f"\nFeature Selection Results:")
    print(f"Anchored features: {len(anchored_features)}")
    print(f"Additional features selected: {len(selected_additional)}")
    print(f"Total features: {len(final_features)}")
    
    return final_features

def run_linearsvr_cv(data_path, target="ACE-km", mode="ace_only", n_features=10):
    """
    Run LinearSVR with different feature anchoring strategies
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    
    # Load feature lists
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    # Get all available features (excluding target columns)
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    
    """
        # Case 1: H2-km prediction
        mode="h2"  # Uses only hydrogenotrophic features (3 features)

        # Case 2: ACE-km with acetoclastic only
        mode="ace_only"  # Uses only acetoclastic feature (1 feature)

        # Case 3: ACE-km with all features
        mode="ace_all"  # Uses all 17 classified features

        # Case 4: ACE-km conditional
        mode="ace_conditional"
        # When ACE-km < 10: Uses acetoclastic feature (1 feature)
        # When ACE-km ≥ 10: Uses all 17 classified features
    """
    
    # Select anchored features based on mode
    if mode == "h2":
        anchored_features = hydrogenotrophic
    elif mode == "ace_only":
        anchored_features = acetoclastic
    elif mode == "ace_all":
        anchored_features = acetoclastic + hydrogenotrophic + syntrophic
    elif mode == "ace_conditional":
        # Will handle this separately
        anchored_features = acetoclastic
    else:
        raise ValueError("Invalid mode specified")

    if mode != "ace_conditional":
        # Perform feature selection
        selected_features = select_features_with_rfe(
            df[all_features], 
            df[target], 
            anchored_features, 
            n_features
        )
        X = df[selected_features]
    else:
        X = df[all_features]  # We'll select features separately for each condition
    
    y = df[target]
    
    print(f"\nDataset shape after cleaning: {df.shape}")
    print(f"Number of features used: {len(X.columns)}")

    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actual = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if mode == "ace_conditional":
            mask_low = df[target] < 10
            print(f"\nSamples with ACE-km < 10: {sum(mask_low)}")
            print(f"Samples with ACE-km ≥ 10: {sum(~mask_low)}")
            
            # For low ACE-km: use only acetoclastic features + additional
            features_low = select_features_with_rfe(
                df[all_features][mask_low],
                df[target][mask_low],
                acetoclastic,
                n_features
            )
            
            # For high ACE-km: use all classified features + additional
            features_high = select_features_with_rfe(
                df[all_features][~mask_low],
                df[target][~mask_low],
                acetoclastic + hydrogenotrophic + syntrophic,
                n_features
            )
            
            # Train model for low ACE-km
            X_train_low = df[all_features][mask_low][features_low]
            scaler_low = StandardScaler()
            X_train_low_scaled = scaler_low.fit_transform(X_train_low)
            
            model_low = LinearSVR(random_state=42)
            model_low.fit(X_train_low_scaled, df[target][mask_low])
            
            # Train model for high ACE-km
            X_train_high = df[all_features][~mask_low][features_high]
            scaler_high = StandardScaler()
            X_train_high_scaled = scaler_high.fit_transform(X_train_high)
            
            model_high = LinearSVR(random_state=42)
            model_high.fit(X_train_high_scaled, df[target][~mask_low])
            
            # Predict for low ACE-km validation samples
            if sum(mask_val_low) > 0:
                X_val_low = df[all_features][mask_low][features_low]
                X_val_low_scaled = scaler_low.transform(X_val_low)
                y_pred_low = model_low.predict(X_val_low_scaled)
                all_predictions.extend(y_pred_low)
                all_actual.extend(y_val[mask_val_low])
            
            # Predict for high ACE-km validation samples
            if sum(mask_val_high) > 0:
                X_val_high = df[all_features][~mask_low][features_high]
                X_val_high_scaled = scaler_high.transform(X_val_high)
                y_pred_high = model_high.predict(X_val_high_scaled)
                all_predictions.extend(y_pred_high)
                all_actual.extend(y_val[mask_val_high])
        
        else:
            # Standard scaling and prediction for non-conditional cases
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = LinearSVR(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            all_predictions.extend(y_pred)
            all_actual.extend(y_val)
        
        print(f"Fold {fold+1} completed")

    # Calculate and display metrics
    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)

    print(f"\nResults for {target} ({mode} mode):")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")

    # Track selected features
    if mode == "ace_conditional":
        # For conditional mode, save both low and high ACE-km features
        feature_importance_low = pd.DataFrame({
            'Feature': features_low,
            'Is_Anchored': [f in acetoclastic for f in features_low],
            'ACE_km_Range': ['<10'] * len(features_low)
        })
        
        feature_importance_high = pd.DataFrame({
            'Feature': features_high,
            'Is_Anchored': [f in (acetoclastic + hydrogenotrophic + syntrophic) for f in features_high],
            'ACE_km_Range': ['>=10'] * len(features_high)
        })
        
        # Combine both feature sets
        feature_importance = pd.concat([feature_importance_low, feature_importance_high], 
                                     ignore_index=True)
    else:
        # For non-conditional modes
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Is_Anchored': [f in anchored_features for f in selected_features]
        })
    
    # Save feature importance information
    feature_importance.to_csv(f'results/metrics/features_{target}_{mode}.csv', index=False)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.5)
    plt.plot([min(all_actual), max(all_actual)], [min(all_actual), max(all_actual)], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values for {target}\n{mode} mode\nR² = {final_r2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/results_{target}_{mode}.png')
    plt.close()

    # Save results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results/metrics/results_{target}_{mode}.csv', index=False)

    return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    # Create directory structure
    create_directories()
    
    # Set paths
    data_path = "data/classified_data.csv"
    n_features = 20  # Total features to select (including anchored features)
    
    # Save summary of all results
    summary_results = []
    
    # For H2-km prediction
    print("\nRunning H2-km prediction...")
    r2_h2, mse_h2, rmse_h2, results_h2 = run_linearsvr_cv(
        data_path, 
        target="H2-km",
        mode="h2",
        n_features=n_features
    )
    summary_results.append({
        'Target': 'H2-km',
        'Mode': 'h2',
        'R2': r2_h2,
        'MSE': mse_h2,
        'RMSE': rmse_h2
    })
    
    # For ACE-km predictions (all three cases)
    print("\nRunning ACE-km prediction (acetoclastic only)...")
    r2_ace1, mse_ace1, rmse_ace1, results_ace1 = run_linearsvr_cv(
        data_path, 
        target="ACE-km",
        mode="ace_only",
        n_features=n_features
    )
    summary_results.append({
        'Target': 'ACE-km',
        'Mode': 'ace_only',
        'R2': r2_ace1,
        'MSE': mse_ace1,
        'RMSE': rmse_ace1
    })
    
    print("\nRunning ACE-km prediction (all features)...")
    r2_ace2, mse_ace2, rmse_ace2, results_ace2 = run_linearsvr_cv(
        data_path, 
        target="ACE-km",
        mode="ace_all",
        n_features=n_features
    )
    summary_results.append({
        'Target': 'ACE-km',
        'Mode': 'ace_all',
        'R2': r2_ace2,
        'MSE': mse_ace2,
        'RMSE': rmse_ace2
    })
    
    print("\nRunning ACE-km prediction (conditional)...")
    r2_ace3, mse_ace3, rmse_ace3, results_ace3 = run_linearsvr_cv(
        data_path, 
        target="ACE-km",
        mode="ace_conditional",
        n_features=n_features
    )
    summary_results.append({
        'Target': 'ACE-km',
        'Mode': 'ace_conditional',
        'R2': r2_ace3,
        'MSE': mse_ace3,
        'RMSE': rmse_ace3
    })
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/metrics/summary_results.csv', index=False)
    
    print("\nAll results have been saved to the 'results' directory.") 