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
    """Create necessary directories for outputs"""
    directories = [
        'results_anchor',
        'results_anchor/metrics',
        'results_anchor/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the different feature lists from saved files"""
    acetoclastic = pd.read_csv('features/acetoclastic_features.csv')['feature'].tolist()
    hydrogenotrophic = pd.read_csv('features/hydrogenotrophic_features.csv')['feature'].tolist()
    syntrophic = pd.read_csv('features/syntrophic_features.csv')['feature'].tolist()
    return acetoclastic, hydrogenotrophic, syntrophic

def select_features_with_rfe(X, y, anchored_features, n_features=5):
    """
    Perform RFE while keeping anchored features and return feature importance
    """
    # Separate anchored and non-anchored features
    non_anchored_features = [col for col in X.columns if col not in anchored_features]
    X_non_anchored = X[non_anchored_features]
    
    if n_features > 0 and len(non_anchored_features) > 0:
        # Perform RFE and get feature importance
        estimator = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_non_anchored, y)
        
        # Get feature importance for selected non-anchored features
        selected_mask = rfe.support_
        selected_features = X_non_anchored.columns[selected_mask]
        final_model = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
        final_model.fit(X_non_anchored[selected_features], y)
        
        # Calculate importance scores and sort features
        importance_scores = np.abs(final_model.coef_)
        feature_importance_pairs = list(zip(selected_features, importance_scores))
        sorted_features = [f for f, _ in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        sorted_importance = [i for _, i in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        
        # Combine anchored and sorted non-anchored features
        final_features = anchored_features + sorted_features
        final_importance = [0] * len(anchored_features) + sorted_importance
    else:
        final_features = anchored_features
        final_importance = [0] * len(anchored_features)
    
    return final_features, final_importance

def create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features):
    """Create and save performance plot with detailed information"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    # Add perfect prediction line
    min_val = 0
    max_val = max(max(all_actual), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Update title based on mode
    if mode == "h2":
        title = f"H2-km Prediction\nUsing {3} Hydrogenotrophic + {n_features} Additional Features"
    elif mode == "ace_only":
        title = f"ACE-km Prediction\nUsing {1} Acetoclastic + {n_features} Additional Features"
    elif mode == "ace_all":
        title = f"ACE-km Prediction\nUsing All {17} Classified + {n_features} Additional Features"
    elif mode == "ace_conditional_low":
        title = f"ACE-km Prediction (< 10)\nUsing {1} Acetoclastic + {n_features} Additional Features"
    elif mode == "ace_conditional_high":
        title = f"ACE-km Prediction (≥ 10)\nUsing {17} Classified + {n_features} Additional Features"
    elif mode == "ace_conditional":
        title = f"ACE-km Prediction (Combined)\nUsing Conditional Features + {n_features} Additional Features"
    else:
        title = f"Prediction Results with {n_features} Additional Features"
    
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with specific filename
    filename = f'results_anchor/plots/results_{target}_{mode}_{n_features}features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_linearsvr_cv(data_path, target="ACE-km", mode="ace_only", n_features=800):
    """
    Run LinearSVR with different feature anchoring strategies
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("H2-km" or "ACE-km")
    mode : str
        Feature selection mode ("h2", "ace_only", "ace_all", "ace_conditional")
    n_features : int
        Number of additional features to select beyond anchored features
    """
    # Add directory creation at the start
    create_directories()
    
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
        selected_features, importance = select_features_with_rfe(
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
    
    # Modify LinearSVR parameters for better convergence
    model = LinearSVR(
        random_state=42,
        max_iter=2000,  # Increase max iterations
        tol=1e-4       # Adjust tolerance
    )
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if mode == "ace_conditional":
            # Define masks for both training and validation sets
            mask_train_low = y_train < 10
            mask_train_high = y_train >= 10
            mask_val_low = y_val < 10
            mask_val_high = y_val >= 10
            
            print(f"\nFold {fold+1}:")
            print(f"Training samples with ACE-km < 10: {sum(mask_train_low)}")
            print(f"Training samples with ACE-km ≥ 10: {sum(mask_train_high)}")
            print(f"Validation samples with ACE-km < 10: {sum(mask_val_low)}")
            print(f"Validation samples with ACE-km ≥ 10: {sum(mask_val_high)}")
            
            # For low ACE-km
            if sum(mask_train_low) > 0:
                features_low, importance_low = select_features_with_rfe(
                    X_train[mask_train_low],
                    y_train[mask_train_low],
                    acetoclastic,
                    n_features
                )
                
                # Train model for low ACE-km
                X_train_low = X_train[mask_train_low][features_low]
                scaler_low = StandardScaler()
                X_train_low_scaled = scaler_low.fit_transform(X_train_low)
                
                model_low = LinearSVR(
                    random_state=42,
                    max_iter=100000,
                    tol=1e-4,
                    dual=True  # Changed to True
                )
                model_low.fit(X_train_low_scaled, y_train[mask_train_low])
                
                # Predict for low ACE-km validation samples
                if sum(mask_val_low) > 0:
                    X_val_low = X_val[mask_val_low][features_low]
                    X_val_low_scaled = scaler_low.transform(X_val_low)
                    y_pred_low = model_low.predict(X_val_low_scaled)
                    all_predictions.extend(y_pred_low)
                    all_actual.extend(y_val[mask_val_low])
            
            # For high ACE-km
            if sum(mask_train_high) > 0:
                features_high, importance_high = select_features_with_rfe(
                    X_train[mask_train_high],
                    y_train[mask_train_high],
                    acetoclastic + hydrogenotrophic + syntrophic,
                    n_features
                )
                
                # Train model for high ACE-km
                X_train_high = X_train[mask_train_high][features_high]
                scaler_high = StandardScaler()
                X_train_high_scaled = scaler_high.fit_transform(X_train_high)
                
                model_high = LinearSVR(
                    random_state=42,
                    max_iter=100,
                    tol=1e-4,
                    dual=True  # Changed to True
                )
                model_high.fit(X_train_high_scaled, y_train[mask_train_high])
                
                # Predict for high ACE-km validation samples
                if sum(mask_val_high) > 0:
                    X_val_high = X_val[mask_val_high][features_high]
                    X_val_high_scaled = scaler_high.transform(X_val_high)
                    y_pred_high = model_high.predict(X_val_high_scaled)
                    all_predictions.extend(y_pred_high)
                    all_actual.extend(y_val[mask_val_high])
            
            print(f"Fold {fold+1} completed")
        
        else:
            # Standard scaling and prediction for non-conditional cases
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
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
            'Feature_Importance': importance_low
        })
        
        feature_importance_high = pd.DataFrame({
            'Feature': features_high,
            'Is_Anchored': [f in (acetoclastic + hydrogenotrophic + syntrophic) for f in features_high],
            'Feature_Importance': importance_high
        })
        
        # Combine both feature sets
        feature_importance = pd.concat([feature_importance_low, feature_importance_high], 
                                     ignore_index=True)
    else:
        # For non-conditional modes
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Is_Anchored': [f in anchored_features for f in selected_features],
            'Feature_Importance': importance
        })
    
    # Save feature importance information
    feature_importance.to_csv(f'results_anchor/metrics/features_{target}_{mode}_{n_features}features.csv', index=False)

    # Create and save plot
    create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features)

    # Save numerical results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_anchor/metrics/results_{target}_{mode}_{n_features}features.csv', index=False)

    if mode == "ace_conditional":
        # Initialize separate results for low and high ACE-km
        all_predictions_combined = []
        all_actual_combined = []
        all_predictions_low = []
        all_actual_low = []
        all_predictions_high = []
        all_actual_high = []
        
        # Process low ACE-km samples
        mask_low = df[target] < 10
        if sum(mask_low) > 0:
            features_low, importance_low = select_features_with_rfe(
                df[mask_low][all_features],
                df[mask_low][target],
                acetoclastic,
                n_features
            )
            
            # Save feature importance for low range
            feature_importance_low = pd.DataFrame({
                'Feature': features_low,
                'Is_Anchored': [f in acetoclastic for f in features_low],
                'Feature_Importance': importance_low,
                'ACE_km_Range': ['<10'] * len(features_low)
            })
            feature_importance_low.to_csv(
                f'results_anchor/metrics/features_{target}_conditional_low_{n_features}features.csv',
                index=False
            )
            
            # Process low range predictions
            X_low = df[mask_low][features_low]
            y_low = df[mask_low][target]
            r2_low, mse_low, rmse_low, results_df_low = process_subset(
                X_low, y_low, 
                target_suffix=f"{target}_conditional_low",
                mode="ace_conditional_low",
                n_features=n_features
            )
            
            # Add to combined results
            all_predictions_combined.extend(results_df_low['Predicted'])
            all_actual_combined.extend(results_df_low['Actual'])
        
        # Process high ACE-km samples
        mask_high = df[target] >= 10
        if sum(mask_high) > 0:
            features_high, importance_high = select_features_with_rfe(
                df[mask_high][all_features],
                df[mask_high][target],
                acetoclastic + hydrogenotrophic + syntrophic,
                n_features
            )
            
            # Save feature importance for high range
            feature_importance_high = pd.DataFrame({
                'Feature': features_high,
                'Is_Anchored': [f in (acetoclastic + hydrogenotrophic + syntrophic) for f in features_high],
                'Feature_Importance': importance_high,
                'ACE_km_Range': ['≥10'] * len(features_high)
            })
            feature_importance_high.to_csv(
                f'results_anchor/metrics/features_{target}_conditional_high_{n_features}features.csv',
                index=False
            )
            
            # Process high range predictions
            X_high = df[mask_high][features_high]
            y_high = df[mask_high][target]
            r2_high, mse_high, rmse_high, results_df_high = process_subset(
                X_high, y_high, 
                target_suffix=f"{target}_conditional_high",
                mode="ace_conditional_high",
                n_features=n_features
            )
            
            # Add to combined results
            all_predictions_combined.extend(results_df_high['Predicted'])
            all_actual_combined.extend(results_df_high['Actual'])
        
        # Calculate combined metrics
        final_r2_combined = r2_score(all_actual_combined, all_predictions_combined)
        final_mse_combined = mean_squared_error(all_actual_combined, all_predictions_combined)
        final_rmse_combined = np.sqrt(final_mse_combined)
        
        # Create combined plot
        create_performance_plot(
            all_actual_combined, 
            all_predictions_combined, 
            target, 
            "ace_conditional", 
            final_r2_combined, 
            final_mse_combined, 
            n_features
        )
        
        # Save combined results
        results_df_combined = pd.DataFrame({
            'Actual': all_actual_combined,
            'Predicted': all_predictions_combined
        })
        results_df_combined.to_csv(
            f'results_anchor/metrics/results_{target}_conditional_{n_features}features.csv',
            index=False
        )
        
        # Save combined feature importance
        feature_importance_combined = pd.concat([feature_importance_low, feature_importance_high])
        feature_importance_combined.to_csv(
            f'results_anchor/metrics/features_{target}_conditional_{n_features}features.csv',
            index=False
        )
        
        return {
            'low': (r2_low, mse_low, rmse_low, results_df_low) if sum(mask_low) > 0 else None,
            'high': (r2_high, mse_high, rmse_high, results_df_high) if sum(mask_high) > 0 else None,
            'combined': (final_r2_combined, final_mse_combined, final_rmse_combined, results_df_combined)
        }
    else:
        return final_r2, final_mse, final_rmse, results_df

def process_subset(X, y, target_suffix, mode, n_features):
    """Helper function to process a subset of data"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = []
    all_actual = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = LinearSVR(random_state=42, max_iter=2000, tol=1e-4, dual=True)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        all_predictions.extend(y_pred)
        all_actual.extend(y_val)
    
    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)
    
    # Create plot
    create_performance_plot(all_actual, all_predictions, target_suffix, mode, final_r2, final_mse, n_features)
    
    # Save results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_anchor/metrics/results_{target_suffix}_{mode}_{n_features}features.csv', index=False)
    
    return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    # Create directories before running
    create_directories()
    
    data_path = "../Data/New_data.csv"
    additional_features = [50, 100, 200, 400, 800]
    all_results = {}
    
    # For H2-km prediction
    print("\nRunning H2-km predictions...")
    for n_features in additional_features:
        print(f"\nTesting with {n_features} additional features")
        r2_h2, mse_h2, rmse_h2, results_h2 = run_linearsvr_cv(
            data_path, 
            target="H2-km",
            mode="h2",
            n_features=n_features
        )
        all_results[f'H2_km_{n_features}'] = {
            'R2': r2_h2,
            'MSE': mse_h2,
            'RMSE': rmse_h2,
            'Additional_Features': n_features
        }
    
    # For ACE-km Case 1 (acetoclastic only)
    print("\nRunning ACE-km acetoclastic only predictions...")
    for n_features in additional_features:
        r2_ace1, mse_ace1, rmse_ace1, results_ace1 = run_linearsvr_cv(
            data_path, 
            target="ACE-km",
            mode="ace_only",
            n_features=n_features
        )
        all_results[f'ACE_km_aceonly_{n_features}'] = {
            'R2': r2_ace1,
            'MSE': mse_ace1,
            'RMSE': rmse_ace1,
            'Additional_Features': n_features
        }
    
    # For ACE-km Case 2 (all features)
    print("\nRunning ACE-km all features predictions...")
    for n_features in additional_features:
        r2_ace2, mse_ace2, rmse_ace2, results_ace2 = run_linearsvr_cv(
            data_path, 
            target="ACE-km",
            mode="ace_all",
            n_features=n_features
        )
        all_results[f'ACE_km_all_{n_features}'] = {
            'R2': r2_ace2,
            'MSE': mse_ace2,
            'RMSE': rmse_ace2,
            'Additional_Features': n_features
        }
    
    # For ACE-km Case 3 (conditional)
    print("\nRunning ACE-km conditional predictions...")
    for n_features in additional_features:
        results = run_linearsvr_cv(
            data_path, 
            target="ACE-km",
            mode="ace_conditional",
            n_features=n_features
        )
        
        # Handle low ACE-km results
        if results['low'] is not None:
            r2_low, mse_low, rmse_low, results_low = results['low']
            all_results[f'ACE_km_conditional_low_{n_features}'] = {
                'R2': r2_low,
                'MSE': mse_low,
                'RMSE': rmse_low,
                'Additional_Features': n_features
            }
        
        # Handle high ACE-km results
        if results['high'] is not None:
            r2_high, mse_high, rmse_high, results_high = results['high']
            all_results[f'ACE_km_conditional_high_{n_features}'] = {
                'R2': r2_high,
                'MSE': mse_high,
                'RMSE': rmse_high,
                'Additional_Features': n_features
            }
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results).T
    results_df.index = results_df.index.astype(str)  # Convert index to string type
    results_df.index.name = 'Model_Configuration'
    results_df.to_csv('results_anchor/metrics/feature_comparison_results.csv')
    
    # Print best configurations
    print("\nBest configurations based on R2 score:")
    for target in ['H2_km', 'ACE_km_aceonly', 'ACE_km_all', 'ACE_km_conditional']:
        target_results = results_df[results_df.index.str.startswith(target)]
        if not target_results.empty:
            best_config = target_results.loc[target_results['R2'].idxmax()]
            print(f"\n{target}:")
            print(f"Best R2: {best_config['R2']:.4f}")
            print(f"MSE: {best_config['MSE']:.4f}")
            print(f"Additional features: {best_config['Additional_Features']}") 