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
        # Perform RFE and get feature importance with increased max_iter to avoid convergence warnings
        estimator = LinearSVR(random_state=42, max_iter=1000, tol=1e-4, dual=True)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_non_anchored, y)
        
        # Get feature importance for selected non-anchored features
        selected_mask = rfe.support_
        selected_features = X_non_anchored.columns[selected_mask]
        final_model = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
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
    # Create a simple, clean figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with consistent styling for all points (smaller size)
    plt.scatter(all_actual, all_predictions, alpha=0.7, color='#1f77b4', s=40)
    
    # Calculate min and max values 
    min_val_x = min(all_actual)
    max_val_x = max(all_actual)
    min_val_y = min(all_predictions)
    max_val_y = max(all_predictions)
    
    # Add 10% padding to ensure all points are clearly visible
    x_padding = (max_val_x - min_val_x) * 0.1
    y_padding = (max_val_y - min_val_y) * 0.1
    
    # If padding is too small (in case of very similar values), use a minimum padding
    min_padding = max(max_val_x, max_val_y) * 0.05
    x_padding = max(x_padding, min_padding)
    y_padding = max(y_padding, min_padding)
    
    # Set axis limits with padding, ensuring we don't go below 0
    x_min = max(0, min_val_x - x_padding)
    x_max = max_val_x + x_padding
    y_min = max(0, min_val_y - y_padding)
    y_max = max_val_y + y_padding
    
    # Use the maximum range for both axes to maintain square aspect ratio
    plot_min = min(x_min, y_min)
    plot_max = max(x_max, y_max)
    
    # Add perfect prediction line
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Perfect Prediction (y=x)')
    
    # Calculate and add the regression line for this case
    coeffs = np.polyfit(all_actual, all_predictions, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    x_reg = np.array([plot_min, plot_max])
    y_reg = slope * x_reg + intercept
    plt.plot(x_reg, y_reg, 'g-', label=f'Regression Line (y={slope:.2f}x+{intercept:.2f})')
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    
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
    plt.title(f'{title}\n{metrics}', fontsize=14)
    
    plt.legend(fontsize=10)
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
        max_iter=10000,  # Increased from 2000 to 10000 to avoid convergence warnings
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

def create_combined_plot(case_results, case_names):
    """
    Create a combined plot showing multiple cases with separate regression lines
    
    Parameters:
    ----------
    case_results : dict
        Dictionary with case name as key and DataFrame with 'Actual' and 'Predicted' columns as value
    case_names : list
        List of case names to include in the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Define colors for different cases
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Initialize for axis limits
    all_actual = []
    all_predicted = []
    all_r2 = {}
    all_mse = {}
    
    # Map case names to more readable descriptions
    case_descriptions = {
        'case4': 'All Features',
        'case5': 'ACE-km ≤ 10',
        'case6': 'Biomass F',
        'case7': 'ACE-km > 10',
        'case8': 'Biomass G'
    }
    
    # Plot each case with its own color and regression line
    for i, case in enumerate(case_names):
        if case not in case_results:
            print(f"Warning: {case} not found in results")
            continue
            
        results = case_results[case]
        actual = results['Actual']
        predicted = results['Predicted']
        
        # Get readable case name
        case_desc = case_descriptions.get(case, case)
        
        # Add to aggregate data for axis limits
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        
        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        all_r2[case] = r2
        all_mse[case] = mse
        
        # Plot scatter for this case
        color = colors[i % len(colors)]
        plt.scatter(actual, predicted, alpha=0.7, s=40, 
                   color=color, label=f'{case_desc} (R²={r2:.4f})')
        
        # Calculate and plot regression line for this case
        coeffs = np.polyfit(actual, predicted, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Create label for regression line
        reg_label = f'{case_desc} Regression (y={slope:.2f}x+{intercept:.2f})'
        
        # Add regression line with label
        x_min, x_max = min(actual), max(actual)
        x_reg = np.array([x_min, x_max])
        y_reg = slope * x_reg + intercept
        plt.plot(x_reg, y_reg, color=color, linestyle='-', alpha=0.8, label=reg_label)
    
    # Calculate padding for axis limits
    min_val_x = min(all_actual)
    max_val_x = max(all_actual)
    min_val_y = min(all_predicted)
    max_val_y = max(all_predicted)
    
    x_padding = (max_val_x - min_val_x) * 0.1
    y_padding = (max_val_y - min_val_y) * 0.1
    
    # If padding is too small, use a minimum padding
    min_padding = max(max_val_x, max_val_y) * 0.05
    x_padding = max(x_padding, min_padding)
    y_padding = max(y_padding, min_padding)
    
    # Set axis limits with padding
    x_min = max(0, min_val_x - x_padding)
    x_max = max_val_x + x_padding
    y_min = max(0, min_val_y - y_padding)
    y_max = max_val_y + y_padding
    
    # Use the maximum range for both axes
    plot_min = min(x_min, y_min)
    plot_max = max(x_max, y_max)
    
    # Add perfect prediction line
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='Perfect Prediction (y=x)')
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    
    # Create title based on the cases being combined
    case_desc_list = [case_descriptions.get(case, case) for case in case_names if case in case_results]
    title = f"Combined Plot: {' vs '.join(case_desc_list)}"
    
    # Calculate combined R² and MSE if needed
    if len(case_names) > 1:
        all_actual_combined = []
        all_predicted_combined = []
        for case in case_names:
            if case in case_results:
                all_actual_combined.extend(case_results[case]['Actual'])
                all_predicted_combined.extend(case_results[case]['Predicted'])
        
        if all_actual_combined:
            combined_r2 = r2_score(all_actual_combined, all_predicted_combined)
            combined_mse = mean_squared_error(all_actual_combined, all_predicted_combined)
            combined_metrics = f'Combined: R² = {combined_r2:.4f}, MSE = {combined_mse:.4f}'
            plt.title(f'{title}\n{combined_metrics}', fontsize=14)
        else:
            plt.title(title, fontsize=14)
    else:
        plt.title(title, fontsize=14)
    
    # Improve the legend by adjusting size and position
    plt.legend(fontsize=9, loc='best', framealpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with specific filename
    case_str = '_and_'.join(case_names)
    filename = f'results_extension/plots/combined_results_{case_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved as {filename}")
    
    return all_r2, all_mse

if __name__ == "__main__":
    # Create directories before running
    create_directories()
    
    data_path = "../Data/New_data.csv"
    additional_features = [50, 100, 200, 400, 800]  # Can modify to try different numbers of features
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
        
        # After running all cases, create combined plots
        # For example, combine case5 (ACE-km ≤ 10) and case7 (ACE-km > 10)
        if all_results:
            # Organize results by case for combined plotting
            case_results = {}
            for mode, description in cases:
                n_features = 50  # Use the results with 50 additional features
                result_key = f'ACE_km_{mode}_{n_features}'
                
                if result_key in all_results:
                    # Get the saved result files
                    results_file = f'results_extension/metrics/results_ACE-km_{mode}_{n_features}features.csv'
                    if os.path.exists(results_file):
                        results_df = pd.read_csv(results_file)
                        # Rename columns to standard names if needed
                        if 'Actual' not in results_df.columns and 'actual' in results_df.columns:
                            results_df = results_df.rename(columns={'actual': 'Actual'})
                        if 'Predicted' not in results_df.columns and 'predicted' in results_df.columns:
                            results_df = results_df.rename(columns={'predicted': 'Predicted'})
                        
                        case_results[mode] = results_df
            
            # Create combined plot for case5 and case7 (low and high ACE-km values)
            if 'case5' in case_results and 'case7' in case_results:
                print("\nCreating combined plot for case5 (ACE-km ≤ 10) and case7 (ACE-km > 10)")
                combined_r2, combined_mse = create_combined_plot(
                    case_results, 
                    ['case5', 'case7']
                )
                
                print("\nCombined plot metrics:")
                for case, r2 in combined_r2.items():
                    print(f"{case}: R² = {r2:.4f}, MSE = {combined_mse[case]:.4f}")
            
            # Create combined plot for case6 and case8 (Biomass F and G)
            if 'case6' in case_results and 'case8' in case_results:
                print("\nCreating combined plot for case6 (Biomass F) and case8 (Biomass G)")
                combined_r2, combined_mse = create_combined_plot(
                    case_results, 
                    ['case6', 'case8']
                )
                
                print("\nCombined plot metrics:")
                for case, r2 in combined_r2.items():
                    print(f"{case}: R² = {r2:.4f}, MSE = {combined_mse[case]:.4f}")
    else:
        print("\nNo results to summarize. Check for errors in the cases.") 