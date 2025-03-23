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
        'results_extension',
        'results_extension/metrics',
        'results_extension/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the specific feature lists based on the image provided"""
    # These are the specific OTUs from the image with their classifications
    acetoclastic = [
        "d__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta"
    ]
    
    hydrogenotrophic = [
        "d__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium",
        # "d__Archaea;p__Halobacterota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium"  # This is an alternate format that might be in the data
    ]
    
    syntrophic = [
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Smithellaceae;g__Smithella",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophobacteria;o__Syntrophobacterales;f__Syntrophobacteraceae;g__Syntrophobacter",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-01",
        "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured",
        "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus",
        "d__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;g__uncultured"
    ]
    
    print(f"Using hardcoded taxonomic features from the image:")
    print(f"- Acetoclastic: {len(acetoclastic)} feature(s)")
    print(f"- Hydrogenotrophic: {len(hydrogenotrophic)} feature(s)")
    print(f"- Syntrophic: {len(syntrophic)} feature(s)")
    
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
    if mode == "case4":
        title = f"Case 4: ACE-km Prediction\nUsing All 8 Classified Features + {n_features} Additional Features"
    elif mode == "case5":
        title = f"Case 5: ACE-km Prediction (≤ 10)\nUsing 1 Acetoclastic Feature + {n_features} Additional Features"
    elif mode == "case6":
        title = f"Case 6: ACE-km Prediction (Biomass F)\nUsing 1 Acetoclastic Feature + {n_features} Additional Features"
    elif mode == "case7":
        title = f"Case 7: ACE-km Prediction (> 10)\nUsing All 8 Classified Features + {n_features} Additional Features"
    elif mode == "case8":
        title = f"Case 8: ACE-km Prediction (Biomass G)\nUsing All 8 Classified Features + {n_features} Additional Features"
    else:
        title = f"Prediction Results with {n_features} Additional Features"
    
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with specific filename
    filename = f'results_extension/plots/results_{target}_{mode}_{n_features}features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def filter_available_features(df, feature_list):
    """Check which features from the provided list are available in the dataset"""
    available_features = [feature for feature in feature_list if feature in df.columns]
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    
    return available_features, missing_features

def run_linearsvr_cv(data_path, target="ACE-km", mode="case4", n_features=50):
    """
    Run LinearSVR with different feature anchoring strategies for extension cases
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("ACE-km")
    mode : str
        Feature selection mode ("case4", "case5", "case6", "case7", "case8")
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
    
    # Check which features are available in the dataset
    acetoclastic_available, acetoclastic_missing = filter_available_features(df, acetoclastic)
    hydrogenotrophic_available, hydrogenotrophic_missing = filter_available_features(df, hydrogenotrophic)
    syntrophic_available, syntrophic_missing = filter_available_features(df, syntrophic)
    
    # Print availability information
    print("\nFeature availability:")
    print(f"- Acetoclastic: {len(acetoclastic_available)}/{len(acetoclastic)} available")
    if acetoclastic_missing:
        print(f"  Missing: {acetoclastic_missing}")
    
    print(f"- Hydrogenotrophic: {len(hydrogenotrophic_available)}/{len(hydrogenotrophic)} available")
    if hydrogenotrophic_missing:
        print(f"  Missing: {hydrogenotrophic_missing}")
    
    print(f"- Syntrophic: {len(syntrophic_available)}/{len(syntrophic)} available")
    if syntrophic_missing:
        print(f"  Missing: {syntrophic_missing}")
    
    # Get all available features (excluding target columns)
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    
    # Remove outlier at ACE-km ~ 33 (actual ~ 50)
    # Using a threshold of 30 for ACE-km to identify the outlier
    outlier_mask = df[target] > 30
    if sum(outlier_mask) > 0:
        print(f"Removing {sum(outlier_mask)} outlier with ACE-km > 30")
        df = df[~outlier_mask]
    
    # Apply additional filtering based on the case
    if mode == "case4":
        # Case 4: Use all 8 features plus additional features and all ACE-km values
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        # No additional filtering
    elif mode == "case5":
        # Case 5: Use acetoclastic feature plus additional features and ACE-km values <= 10
        anchored_features = acetoclastic_available
        df = df[df[target] <= 10]
        print(f"Filtered to {len(df)} samples with ACE-km <= 10")
    elif mode == "case6":
        # Case 6: Use acetoclastic feature plus additional features for Biomass "F"
        anchored_features = acetoclastic_available
        if "Biomass_F" in df.columns:
            df = df[df["Biomass_F"] == 1]
            print(f"Filtered to {len(df)} samples with Biomass F")
        else:
            print("Warning: Biomass_F column not found")
    elif mode == "case7":
        # Case 7: Use all 8 features plus additional features and ACE-km values > 10
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        df = df[df[target] > 10]
        print(f"Filtered to {len(df)} samples with ACE-km > 10")
    elif mode == "case8":
        # Case 8: Use all 8 features plus additional features for Biomass "G"
        anchored_features = acetoclastic_available + hydrogenotrophic_available + syntrophic_available
        if "Biomass_G" in df.columns:
            df = df[df["Biomass_G"] == 1]
            print(f"Filtered to {len(df)} samples with Biomass G")
        else:
            print("Warning: Biomass_G column not found")
    else:
        raise ValueError("Invalid mode specified")
    
    # Check if we have enough samples to proceed
    if len(df) < 5:
        print(f"Warning: Only {len(df)} samples left after filtering. Cannot proceed with cross-validation.")
        return None, None, None, None
    
    # Check if any anchored features are available
    if not anchored_features:
        print("Warning: No anchored features available in the dataset. Cannot proceed.")
        return None, None, None, None
    
    # Perform feature selection
    selected_features, importance = select_features_with_rfe(
        df[all_features], 
        df[target], 
        anchored_features, 
        n_features
    )
    X = df[selected_features]
    y = df[target]
    
    print(f"\nDataset shape after cleaning and filtering: {df.shape}")
    print(f"Number of features used: {len(X.columns)}")
    print(f"Number of anchored features used: {len(anchored_features)}")

    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actual = []
    
    # Modify LinearSVR parameters for better convergence
    model = LinearSVR(
        random_state=42,
        max_iter=2000,  # Increase max iterations
        tol=1e-4,       # Adjust tolerance
        dual=True       # Use dual formulation
    )
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Standard scaling before model fitting
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
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Is_Anchored': [f in anchored_features for f in selected_features],
        'Feature_Importance': importance
    })
    
    # Save feature importance information
    feature_importance.to_csv(f'results_extension/metrics/features_{target}_{mode}_{n_features}features.csv', index=False)

    # Create and save plot
    create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features)

    # Save numerical results
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_extension/metrics/results_{target}_{mode}_{n_features}features.csv', index=False)

    return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    # Create directories before running
    create_directories()
    
    data_path = "../Data/New_data.csv"
    additional_features = [50]  # Can modify to try different numbers of features
    all_results = {}
    
    # Define all cases to run
    cases = [
        ("case4", "Using all 8 features plus ≤50 additional features and all ACE-km values"),
        ("case5", "Using acetoclastic feature plus ≤50 additional features and ACE-km values ≤10"),
        ("case6", "Using acetoclastic feature plus ≤50 additional features for Biomass F"),
        ("case7", "Using all 8 features plus ≤50 additional features and ACE-km values >10"),
        ("case8", "Using all 8 features plus ≤50 additional features for Biomass G")
    ]
    
    # Run all cases with specified number of features
    for mode, description in cases:
        print(f"\n{'=' * 80}")
        print(f"Running {mode}: {description}")
        print(f"{'=' * 80}")
        
        for n_features in additional_features:
            print(f"\nUsing {n_features} additional features")
            
            # Run the model
            r2, mse, rmse, results = run_linearsvr_cv(
                data_path=data_path,
                target="ACE-km",
                mode=mode,
                n_features=n_features
            )
            
            # Store results if model ran successfully
            if r2 is not None:
                all_results[f'ACE_km_{mode}_{n_features}'] = {
                    'R2': r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'Additional_Features': n_features
                }
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results).T
        results_df.index = results_df.index.astype(str)  # Convert index to string type
        results_df.index.name = 'Model_Configuration'
        results_df.to_csv('results_extension/metrics/extension_comparison_results.csv')
        
        # Print summary of results
        print("\nSummary of results:")
        print(results_df.to_string())
    else:
        print("\nNo results to summarize. Check for errors in the cases.") 