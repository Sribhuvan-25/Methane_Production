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
import scipy.stats as stats
import datetime

# Create necessary directories
def create_directories():
    """Create necessary directories for outputs"""
    directories = [
        'results_biomass',
        'results_biomass/metrics',
        'results_biomass/plots'
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
    
    print(f"Using hardcoded taxonomic features:")
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

def filter_available_features(df, feature_list):
    """Check which features from the provided list are available in the dataset"""
    available_features = [feature for feature in feature_list if feature in df.columns]
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    
    return available_features, missing_features

def create_performance_plot_by_biomass(all_actual, all_predictions, target, mode, r2, mse, n_features, biomass_results=None, biomass_cols=None):
    """
    Create a scatter plot with points colored by biomass type.
    
    Parameters:
    ----------
    all_actual : list
        Actual target values
    all_predictions : list
        Predicted target values
    target : str
        Name of the target variable
    mode : str
        Mode of analysis (case6, case8, case9)
    r2 : float
        R² score
    mse : float
        Mean squared error
    n_features : int
        Number of features used
    biomass_results : pandas.DataFrame, optional
        DataFrame with predictions and biomass type information
    biomass_cols : list, optional
        List of biomass column names
    """
    plt.figure(figsize=(12, 10))
    
    # Find min and max values for setting axis limits
    min_val = min(min(all_actual), min(all_predictions))
    max_val = max(max(all_actual), max(all_predictions))
    
    # Add some padding to the limits
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    # Plot the perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label='Perfect Prediction')
    
    # Get case title
    if mode == "case6":
        title = f"Case 6: Biomass F - {target} Prediction\nUsing 1 Acetoclastic Feature + {n_features} Additional Features"
    elif mode == "case8":
        title = f"Case 8: Biomass G - {target} Prediction\nUsing All 8 Classified Features + {n_features} Additional Features"
    elif mode == "case9":
        title = f"Case 9: All Samples - {target} Prediction\nUsing 1 Acetoclastic Feature + {n_features} Additional Features"
    else:
        title = f"{mode} - {target} Prediction"
    
    # Determine point colors based on biomass type (for case9)
    if mode == "case9" and biomass_results is not None and biomass_cols is not None:
        print(f"Creating biomass-colored plot for {mode}")
        
        # Debug: Print biomass column names and sample counts to identify any issues
        print("Biomass columns:", biomass_cols)
        for col in biomass_cols:
            biomass_type = col.split('_')[1]
            count = sum(biomass_results[col] == 1)
            print(f"  - {col}: {count} samples")
        
        # Define colors for different biomass types
        colors = plt.cm.tab10(np.linspace(0, 1, len(biomass_cols) + 1))
        biomass_colors = {col: colors[i] for i, col in enumerate(biomass_cols)}
        biomass_colors['Unknown'] = colors[-1]
        
        # Add scatter points for each biomass type
        for i, col in enumerate(biomass_cols):
            biomass_type = col.split('_')[1]
            mask = biomass_results[col] == 1
            samples_count = sum(mask)
            
            if samples_count > 0:
                # Make sure the label clearly identifies which biomass type this is
                plt.scatter(
                    biomass_results.loc[mask, 'Actual'],
                    biomass_results.loc[mask, 'Predicted'],
                    alpha=0.8,
                    color=biomass_colors[col],
                    label=f'Biomass {biomass_type} (n={samples_count})',
                    s=80,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Calculate and add regression line for this biomass type if enough samples
                if samples_count >= 3:
                    x_vals = biomass_results.loc[mask, 'Actual']
                    y_vals = biomass_results.loc[mask, 'Predicted']
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    slope, intercept = coeffs
                    
                    # Plot regression line with biomass type clearly identified
                    x_range = np.array([min(x_vals), max(x_vals)])
                    y_range = slope * x_range + intercept
                    plt.plot(x_range, y_range, '-', color=biomass_colors[col], 
                            alpha=0.7, linewidth=2,
                            label=f'Biomass {biomass_type} fit (y={slope:.2f}x+{intercept:.2f})')
        
        # Handle unknown biomass samples
        unknown_mask = ~biomass_results[biomass_cols].any(axis=1)
        unknown_count = sum(unknown_mask)
        
        if unknown_count > 0:
            plt.scatter(
                biomass_results.loc[unknown_mask, 'Actual'],
                biomass_results.loc[unknown_mask, 'Predicted'],
                alpha=0.8,
                color=biomass_colors['Unknown'],
                label=f'Unknown Biomass (n={unknown_count})',
                s=80,
                edgecolors='black',
                linewidths=0.5,
                marker='s'  # square marker for unknown
            )
    else:
        # For case6 and case8, use a single color for all points
        color = '#1f77b4' if mode == "case6" else '#ff7f0e'  # blue for case6, orange for case8
        plt.scatter(
            all_actual, 
            all_predictions, 
            alpha=0.8, 
            color=color, 
            s=80,
            edgecolors='black',
            linewidths=0.5,
            label=f'Samples (n={len(all_actual)})'
        )
        
        # Calculate and add regression line
        coeffs = np.polyfit(all_actual, all_predictions, 1)
        slope, intercept = coeffs
        x_range = np.array([min(all_actual), max(all_actual)])
        y_range = slope * x_range + intercept
        plt.plot(x_range, y_range, '-', color='green', 
                alpha=0.8, linewidth=2,
                label=f'Regression Line (y={slope:.2f}x+{intercept:.2f})')
    
    # Configure plot aesthetics
    plt.xlabel(f'Actual {target}', fontsize=14)
    plt.ylabel(f'Predicted {target}', fontsize=14)
    plt.title(f'{title}\nR² = {r2:.4f}, MSE = {mse:.4f}', fontsize=16)
    
    # Set equal aspect and limits
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add a grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    # Add text with performance metrics
    plt.annotate(f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {np.sqrt(mse):.4f}',
                xy=(0.02, 0.96), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=12)
    
    # Save the plot
    plot_filename = f"results_biomass/plots/{mode}_{target}_n{n_features}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_filename}")
    
    # Also save a version with the date to keep track of different runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results_biomass/plots/{mode}_{target}_n{n_features}_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"Saved timestamped plot for reference")
    
    plt.close()
    
    # For case9, create additional distribution plot to analyze biomass sample distribution
    if mode == "case9" and biomass_results is not None and biomass_cols is not None:
        create_biomass_distribution_plot(biomass_results, biomass_cols, target)

def create_biomass_distribution_plot(biomass_results, biomass_cols, target):
    """Create a distribution plot showing the range of target values for each biomass type"""
    plt.figure(figsize=(12, 8))
    
    # Create df with biomass type as a categorical column
    df_plot = pd.DataFrame()
    df_plot[target] = biomass_results['Actual']
    df_plot['Biomass_Type'] = 'Unknown'
    
    # Assign biomass type to each sample
    for col in biomass_cols:
        biomass_type = col.split('_')[1]
        mask = biomass_results[col] == 1
        df_plot.loc[mask, 'Biomass_Type'] = biomass_type
    
    # Create violin plot to show distribution by biomass type
    ax = sns.violinplot(x='Biomass_Type', y=target, data=df_plot, palette='Set2', inner='box')
    
    # Add individual points as swarm plot
    sns.swarmplot(x='Biomass_Type', y=target, data=df_plot, color='black', alpha=0.7, size=5)
    
    # Add count labels
    for i, biomass_type in enumerate(['F', 'G', 'Unknown']):
        count = sum(df_plot['Biomass_Type'] == biomass_type)
        plt.text(i, df_plot[target].min(), f'n={count}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line at ACE-km = 10 (the cutoff used in case5)
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
    
    # Calculate statistics by biomass type
    stats_table = df_plot.groupby('Biomass_Type')[target].agg(['mean', 'median', 'min', 'max', 'count'])
    stats_table = stats_table.round(2)
    
    # Add text box with statistics
    plt.annotate(f"Biomass Type Statistics:\n{stats_table.to_string()}",
                xy=(0.02, 0.02), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='bottom', fontsize=10)
    
    # Configure plot
    plt.title(f'Distribution of {target} Values by Biomass Type', fontsize=16)
    plt.xlabel('Biomass Type', fontsize=14)
    plt.ylabel(target, fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plot_filename = f"results_biomass/plots/biomass_distribution_{target}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to: {plot_filename}")
    plt.close()

def analyze_biomass_metrics(biomass_metrics, all_metrics, mode, n_features):
    """
    Analyze and create visualizations for biomass-specific metrics.
    
    Parameters:
    ----------
    biomass_metrics : dict
        Dictionary with metrics by biomass type
    all_metrics : dict
        Dictionary with overall metrics
    mode : str
        Mode of analysis
    n_features : int
        Number of features used
    """
    if not biomass_metrics:
        return
    
    # Create bar plot comparing R² across biomass types
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    biomass_types = list(biomass_metrics.keys()) + ['Combined (All)']
    r2_values = [metrics['R2'] for metrics in biomass_metrics.values()] + [all_metrics['R2']]
    sample_counts = [metrics['Sample_Count'] for metrics in biomass_metrics.values()] + [all_metrics['Sample_Count']]
    
    # Bar colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(biomass_types)))
    
    # Create bars
    bars = plt.bar(biomass_types, r2_values, alpha=0.7, color=colors)
    
    # Add sample count labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={sample_counts[i]}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # Configure plot
    plt.title(f'R² Comparison by Biomass Type ({mode}, {n_features} additional features)', fontsize=14)
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(0, max(r2_values) * 1.2)  # Add space for labels
    plt.grid(axis='y', alpha=0.3)
    
    # Add reference line at R²=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='R²=0 (Baseline)')
    
    # Save plot
    plt.tight_layout()
    plot_filename = f"results_biomass/plots/{mode}_biomass_r2_comparison_n{n_features}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved R² comparison plot to: {plot_filename}")
    plt.close()
    
    # Create comparison table for metrics
    comparison_table = []
    for biomass_type, metrics in biomass_metrics.items():
        comparison_table.append({
            'Biomass_Type': biomass_type,
            'Sample_Count': metrics['Sample_Count'],
            'R²': f"{metrics['R2']:.4f}",
            'MSE': f"{metrics['MSE']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}"
        })
    
    # Add overall metrics for comparison
    comparison_table.append({
        'Biomass_Type': 'Combined (All)',
        'Sample_Count': all_metrics['Sample_Count'],
        'R²': f"{all_metrics['R2']:.4f}",
        'MSE': f"{all_metrics['MSE']:.4f}",
        'RMSE': f"{all_metrics['RMSE']:.4f}"
    })
    
    # Save comparison table as CSV
    metrics_df = pd.DataFrame(comparison_table)
    metrics_filename = f"results_biomass/metrics/{mode}_biomass_metrics_n{n_features}.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Saved biomass metrics to: {metrics_filename}")
    
    return metrics_df 

def run_linearsvr_cv(data_path, target="ACE-km", mode="case9", n_features=50):
    """
    Run LinearSVR with different feature anchoring strategies for cases 6, 8, and 9
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file
    target : str
        Target variable to predict ("ACE-km")
    mode : str
        Feature selection mode ("case6", "case8", "case9")
    n_features : int
        Number of additional features to select beyond anchored features
        
    Returns:
    -------
    final_r2 : float
        R² score on the test set
    final_mse : float
        Mean Squared Error on the test set
    final_rmse : float
        Root Mean Squared Error on the test set
    results_df : DataFrame
        DataFrame with actual and predicted values
    sample_info : dict
        Dictionary with sample count information
    biomass_metrics : dict, optional
        Dictionary with metrics by biomass type (only for case9)
    """
    # Create directories at the start
    create_directories()
    
    # Track sample counts
    sample_info = {}
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n{'*'*50}")
    print(f"Data loaded from: {data_path}")
    print(f"Original data shape: {df.shape}")
    
    # Check for biomass columns in original data
    original_biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
    print(f"Biomass columns in data: {len(original_biomass_cols)}")
    if original_biomass_cols:
        print(f"Biomass columns: {original_biomass_cols}")
        
        # Print distribution of biomass types in original data
        print("\nBiomass distribution in original data:")
        for col in original_biomass_cols:
            biomass_type = col.split('_')[1]
            count = df[col].sum()
            percentage = count / len(df) * 100
            print(f"  - Biomass {biomass_type}: {count} samples ({percentage:.1f}%)")
        
        # Count samples with no biomass type assigned
        no_biomass = (~df[original_biomass_cols].any(axis=1)).sum()
        if no_biomass > 0:
            percentage = no_biomass / len(df) * 100
            print(f"  - Unknown biomass: {no_biomass} samples ({percentage:.1f}%)")
    else:
        print("No biomass columns found in the data!")
    print(f"{'*'*50}\n")
    
    sample_info['original'] = len(df)
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    sample_info['after_removing_x'] = len(df)
    
    print(f"\n{'*' * 20} SAMPLE COUNT INFORMATION {'*' * 20}")
    print(f"Original dataset: {sample_info['original']} samples")
    print(f"After removing rows with 'x': {sample_info['after_removing_x']} samples")
    
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
    outliers_removed = sum(outlier_mask)
    sample_info['outliers_removed'] = outliers_removed
    
    if outliers_removed > 0:
        print(f"\nRemoving {outliers_removed} outlier(s) with {target} > 30")
        df = df[~outlier_mask]
    
    sample_info['after_outlier_removal'] = len(df)
    print(f"Samples after outlier removal: {sample_info['after_outlier_removal']}")
    
    # Store original DataFrame for later analysis
    original_df = df.copy()
    
    # Check for biomass type columns
    biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
    if biomass_cols:
        print(f"\nFound {len(biomass_cols)} biomass type columns: {', '.join(biomass_cols)}")
        for col in biomass_cols:
            biomass_type = col.split('_')[1]
            count = df[col].sum()
            percentage = count / len(df) * 100
            print(f"  - Biomass {biomass_type}: {count} samples ({percentage:.1f}%)")
        
        # Count samples with no biomass type assigned
        no_biomass = (~df[biomass_cols].any(axis=1)).sum()
        if no_biomass > 0:
            percentage = no_biomass / len(df) * 100
            print(f"  - Unknown biomass: {no_biomass} samples ({percentage:.1f}%)")
    
    # Apply case-specific filtering
    if mode == "case6":
        # Case 6: Use acetoclastic feature plus additional features for Biomass "F"
        anchored_features = acetoclastic_available
        if "Biomass_F" in df.columns:
            samples_before = len(df)
            
            # Filter to only keep samples with Biomass F
            df = df[df["Biomass_F"] == 1]
            samples_after = len(df)
            sample_info['before_filtering'] = samples_before
            sample_info['final'] = samples_after
            
            print(f"\n*** CASE 6: Filtered to {samples_after}/{samples_before} samples with Biomass F ***")
            print(f"Percentage of total: {samples_after/samples_before:.1%}")
            
            # Print distribution of target values in this subset
            print(f"\nDistribution of {target} values in Biomass F subset ({samples_after} samples):")
            print(f"  Mean: {df[target].mean():.2f}")
            print(f"  Median: {df[target].median():.2f}")
            print(f"  Min: {df[target].min():.2f}")
            print(f"  Max: {df[target].max():.2f}")
            print(f"  Standard deviation: {df[target].std():.2f}")
        else:
            print("Warning: Biomass_F column not found")
            sample_info['final'] = len(df)
    elif mode == "case8":
        # Case 8: Use all 8 features plus additional features for Biomass "G"
        anchored_features = acetoclastic_available # + hydrogenotrophic_available + syntrophic_available
        if "Biomass_G" in df.columns:
            samples_before = len(df)
            
            # Filter to only keep samples with Biomass G
            df = df[df["Biomass_G"] == 1]
            samples_after = len(df)
            sample_info['before_filtering'] = samples_before
            sample_info['final'] = samples_after
            
            print(f"\n*** CASE 8: Filtered to {samples_after}/{samples_before} samples with Biomass G ***")
            print(f"Percentage of total: {samples_after/samples_before:.1%}")
            
            # Print distribution of target values in this subset
            print(f"\nDistribution of {target} values in Biomass G subset ({samples_after} samples):")
            print(f"  Mean: {df[target].mean():.2f}")
            print(f"  Median: {df[target].median():.2f}")
            print(f"  Min: {df[target].min():.2f}")
            print(f"  Max: {df[target].max():.2f}")
            print(f"  Standard deviation: {df[target].std():.2f}")
        else:
            print("Warning: Biomass_G column not found")
            sample_info['final'] = len(df)
    elif mode == "case9":
        # Case 9: Use acetoclastic feature plus additional features and all ACE-km values (no filtering)
        anchored_features = acetoclastic_available #+ hydrogenotrophic_available + syntrophic_available
        print(f"\n*** CASE 9: Using all {len(df)} samples with 1 acetoclastic feature (no filtering) ***")
        sample_info['final'] = len(df)
        
        # Print distribution of target values in the full dataset
        print(f"\nDistribution of {target} values in full dataset ({len(df)} samples):")
        print(f"  Mean: {df[target].mean():.2f}")
        print(f"  Median: {df[target].median():.2f}")
        print(f"  Min: {df[target].min():.2f}")
        print(f"  Max: {df[target].max():.2f}")
        print(f"  Standard deviation: {df[target].std():.2f}")
        
        # For Case 9, print additional statistics by biomass type
        if biomass_cols:
            print("\nAnalyzing target values by biomass type:")
            for col in biomass_cols:
                biomass_type = col.split('_')[1]
                biomass_samples = df[df[col] == 1]
                count = len(biomass_samples)
                if count > 0:
                    print(f"\nStatistics for Biomass {biomass_type} ({count} samples):")
                    print(f"  Mean: {biomass_samples[target].mean():.2f}")
                    print(f"  Median: {biomass_samples[target].median():.2f}")
                    print(f"  Min: {biomass_samples[target].min():.2f}")
                    print(f"  Max: {biomass_samples[target].max():.2f}")
                    print(f"  Standard deviation: {biomass_samples[target].std():.2f}")
                    
                    # Count samples with target <= 10 (cutoff used in case5)
                    low_count = sum(biomass_samples[target] <= 10)
                    high_count = sum(biomass_samples[target] > 10)
                    print(f"  Samples with {target} ≤ 10: {low_count} ({low_count/count:.1%})")
                    print(f"  Samples with {target} > 10: {high_count} ({high_count/count:.1%})")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: case6, case8, case9")
    
    print(f"\n{'*' * 20} FINAL SAMPLE COUNT: {sample_info['final']} {'*' * 20}")
    
    # Check if we have enough samples to proceed
    if len(df) < 5:
        print(f"Warning: Only {len(df)} samples left after filtering. Cannot proceed with cross-validation.")
        return None, None, None, None, sample_info, None if mode == "case9" else None
    
    # Check if any anchored features are available
    if not anchored_features:
        print("Warning: No anchored features available in the dataset. Cannot proceed.")
        return None, None, None, None, sample_info, None if mode == "case9" else None
    
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
    print(f"Number of anchored features used: {len(anchored_features)}/{len(selected_features)} ({len(anchored_features)/len(selected_features):.1%})")

    # Add sample to feature ratio information
    sample_to_feature_ratio = len(df) / len(selected_features)
    sample_info['sample_to_feature_ratio'] = sample_to_feature_ratio
    print(f"Sample to feature ratio: {sample_to_feature_ratio:.2f}")
    
    # Check if the number of samples is low relative to the number of features
    if sample_to_feature_ratio < 5:
        print(f"\nWarning: Sample size ({len(df)}) is less than 5 times the number of features ({len(selected_features)}).")
        print("This may lead to overfitting and unreliable model performance metrics.")
    
    # Initialize K-Fold
    n_splits = min(5, len(df))  # Adjust number of folds if sample size is very small
    sample_info['n_folds'] = n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actual = []
    fold_r2_scores = []
    
    # Configure LinearSVR model
    model = LinearSVR(
        random_state=42,
        max_iter=10000,
        tol=1e-4,
        dual=True
    )
    
    # Perform cross-validation
    print(f"\nRunning {n_splits}-fold cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"  Fold {fold+1}: Training on {len(X_train)} samples, Validating on {len(X_val)} samples")
        
        # Standard scaling before model fitting
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        # Calculate fold-specific R²
        fold_r2 = r2_score(y_val, y_pred)
        fold_r2_scores.append(fold_r2)
        
        all_predictions.extend(y_pred)
        all_actual.extend(y_val)
        
        print(f"  Fold {fold+1} R²: {fold_r2:.4f}")

    # Calculate and display metrics
    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)
    
    # Store metrics in sample_info
    sample_info['R2'] = final_r2
    sample_info['MSE'] = final_mse
    sample_info['RMSE'] = final_rmse

    print(f"\nResults for {target} ({mode} mode with {sample_info['final']} samples):")
    print(f"R² Score: {final_r2:.4f}")
    print(f"Mean Fold R²: {np.mean(fold_r2_scores):.4f}")
    print(f"Fold R² Std Dev: {np.std(fold_r2_scores):.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    
    # If R² is negative, add an explanation
    if final_r2 < 0:
        print("\nNote: Negative R² value indicates the model performs worse than predicting the mean value.")
        print("This may be due to:")
        print(f"  - Small sample size (n={len(df)})")
        print(f"  - High variance in the data (target std dev: {df[target].std():.2f})")
        print("  - Non-linear relationships that LinearSVR can't capture")
        print("  - Selected features may not be predictive for this subset")
        
        # Simple mean predictor baseline
        y_mean = np.mean(all_actual)
        mean_mse = mean_squared_error(all_actual, [y_mean] * len(all_actual))
        print(f"\nBaseline (mean predictor) MSE: {mean_mse:.4f}")
        print(f"Model MSE: {final_mse:.4f}")
        if final_mse > mean_mse:
            print(f"Model performs worse than baseline by {final_mse/mean_mse - 1:.1%}")

    # Track selected features
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Is_Anchored': [f in anchored_features for f in selected_features],
        'Feature_Importance': importance
    })
    
    # Save feature importance information
    feature_importance.to_csv(f'results_biomass/metrics/features_{target}_{mode}_{n_features}features.csv', index=False)

    # Create DataFrame with predictions and actual values
    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    
    # Save numerical results
    results_df.to_csv(f'results_biomass/metrics/results_{target}_{mode}_{n_features}features.csv', index=False)
    
    # For Case 9, calculate metrics by biomass type
    biomass_metrics = {}
    if mode == "case9" and biomass_cols:
        print("\n== Calculating Metrics by Biomass Type ==")
        
        # Create DataFrame with predictions and biomass information
        biomass_results = pd.DataFrame()
        biomass_results['Actual'] = all_actual
        biomass_results['Predicted'] = all_predictions
        
        # Get validation indices to map back to original dataframe
        val_indices = []
        for _, (_, val_idx) in enumerate(kf.split(X)):
            val_indices.extend(val_idx)
        
        # Add biomass type information
        for i, idx in enumerate(val_indices):
            for col in biomass_cols:
                if col not in biomass_results.columns:
                    biomass_results[col] = 0
                biomass_results.loc[i, col] = df.iloc[idx][col]
        
        # Calculate metrics for each biomass type
        for col in biomass_cols:
            biomass_type = col.split('_')[1]
            mask = biomass_results[col] == 1
            biomass_count = sum(mask)
            
            if biomass_count >= 3:  # Need at least 3 samples for meaningful metrics
                biomass_actual = biomass_results.loc[mask, 'Actual']
                biomass_predicted = biomass_results.loc[mask, 'Predicted']
                
                # Calculate metrics
                biomass_r2 = r2_score(biomass_actual, biomass_predicted)
                biomass_mse = mean_squared_error(biomass_actual, biomass_predicted)
                biomass_rmse = np.sqrt(biomass_mse)
                
                # Store metrics
                biomass_metrics[biomass_type] = {
                    'R2': biomass_r2,
                    'MSE': biomass_mse,
                    'RMSE': biomass_rmse,
                    'Sample_Count': biomass_count
                }
                
                print(f"\nMetrics for Biomass {biomass_type} (n={biomass_count}):")
                print(f"  R² Score: {biomass_r2:.4f}")
                print(f"  MSE: {biomass_mse:.4f}")
                print(f"  RMSE: {biomass_rmse:.4f}")
                
                # Distribution of target values for this biomass type
                print(f"  {target} value range: {min(biomass_actual):.2f} - {max(biomass_actual):.2f}")
                print(f"  Samples with {target} ≤ 10: {sum(biomass_actual <= 10)} ({sum(biomass_actual <= 10)/biomass_count:.1%})")
                print(f"  Samples with {target} > 10: {sum(biomass_actual > 10)} ({sum(biomass_actual > 10)/biomass_count:.1%})")
            else:
                print(f"\nNot enough samples for Biomass {biomass_type} (only {biomass_count} samples)")
        
        # Calculate metrics for samples with unknown biomass type
        unknown_mask = ~biomass_results[biomass_cols].any(axis=1)
        unknown_count = sum(unknown_mask)
        
        if unknown_count >= 3:  # Need at least 3 samples
            unknown_actual = biomass_results.loc[unknown_mask, 'Actual']
            unknown_predicted = biomass_results.loc[unknown_mask, 'Predicted']
            
            # Calculate metrics
            unknown_r2 = r2_score(unknown_actual, unknown_predicted)
            unknown_mse = mean_squared_error(unknown_actual, unknown_predicted)
            unknown_rmse = np.sqrt(unknown_mse)
            
            # Store metrics
            biomass_metrics['Unknown'] = {
                'R2': unknown_r2,
                'MSE': unknown_mse,
                'RMSE': unknown_rmse,
                'Sample_Count': unknown_count
            }
            
            print(f"\nMetrics for Unknown Biomass (n={unknown_count}):")
            print(f"  R² Score: {unknown_r2:.4f}")
            print(f"  MSE: {unknown_mse:.4f}")
            print(f"  RMSE: {unknown_rmse:.4f}")
            
            # Distribution of target values for unknown biomass samples
            print(f"  {target} value range: {min(unknown_actual):.2f} - {max(unknown_actual):.2f}")
            print(f"  Samples with {target} ≤ 10: {sum(unknown_actual <= 10)} ({sum(unknown_actual <= 10)/unknown_count:.1%})")
            print(f"  Samples with {target} > 10: {sum(unknown_actual > 10)} ({sum(unknown_actual > 10)/unknown_count:.1%})")
        elif unknown_count > 0:
            print(f"\nNot enough samples with Unknown Biomass (only {unknown_count} samples)")
        
        # Analyze biomass-specific metrics
        all_metrics = {
            'R2': final_r2,
            'MSE': final_mse,
            'RMSE': final_rmse,
            'Sample_Count': len(all_actual)
        }
        analyze_biomass_metrics(biomass_metrics, all_metrics, mode, n_features)
        
        # Create performance plot with biomass coloring
        create_performance_plot_by_biomass(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features, 
                                          biomass_results=biomass_results, biomass_cols=biomass_cols)
    else:
        # For case6 and case8, create standard performance plot
        create_performance_plot_by_biomass(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features)
    
    # Return appropriate values based on mode
    if mode == "case9":
        return final_r2, final_mse, final_rmse, results_df, sample_info, biomass_metrics
    else:
        return final_r2, final_mse, final_rmse, results_df, sample_info 

def create_combined_biomass_plot(case_results, target="ACE-km", n_features=50):
    """
    Create a combined plot comparing all three cases (6, 8, and 9)
    
    Parameters:
    ----------
    case_results : dict
        Dictionary with results for each case
    target : str
        Target variable
    n_features : int
        Number of additional features used
    """
    plt.figure(figsize=(14, 10))
    
    # Define colors for different cases
    colors = {
        'case6': '#1f77b4',  # blue for Biomass F
        'case8': '#ff7f0e',  # orange for Biomass G
        'case9': '#2ca02c'   # green for all samples
    }
    
    # Define plot markers
    markers = {
        'case6': 'o',  # circle
        'case8': 's',  # square
        'case9': '^'   # triangle
    }
    
    # Map case names to more readable descriptions
    case_descriptions = {
        'case6': 'Biomass F (1 Feature)',
        'case8': 'Biomass G (8 Features)',
        'case9': 'All Samples (1 Feature)'
    }
    
    # Find global min and max for consistent axes
    all_actual = []
    all_predicted = []
    
    # Plot each case with its own color and regression line
    for case in ['case6', 'case8', 'case9']:
        if case not in case_results:
            print(f"Warning: Results for {case} not found")
            continue
            
        results = case_results[case]
        actual = results['Actual']
        predicted = results['Predicted']
        
        # Add to aggregate data for axis limits
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        
        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        sample_count = len(actual)
        
        # Plot scatter for this case
        plt.scatter(
            actual, predicted, 
            alpha=0.8, 
            s=70, 
            color=colors[case],
            marker=markers[case],
            edgecolors='black',
            linewidths=0.5,
            label=f'{case_descriptions[case]} (n={sample_count}, R²={r2:.4f})'
        )
        
        # Calculate and plot regression line for this case
        coeffs = np.polyfit(actual, predicted, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Create label for regression line
        reg_label = f'{case_descriptions[case]} Line (y={slope:.2f}x+{intercept:.2f})'
        
        # Add regression line with label
        x_min, x_max = min(actual), max(actual)
        x_reg = np.array([x_min, x_max])
        y_reg = slope * x_reg + intercept
        plt.plot(x_reg, y_reg, '-', color=colors[case], linewidth=2, alpha=0.7, label=reg_label)
    
    # Calculate padding for axis limits
    min_val = min(min(all_actual), min(all_predicted))
    max_val = max(max(all_actual), max(all_predicted))
    
    # Add some padding to the limits
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    # Add perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label='Perfect Prediction')
    
    # Configure plot aesthetics
    plt.xlabel(f'Actual {target}', fontsize=14)
    plt.ylabel(f'Predicted {target}', fontsize=14)
    plt.title(f'Comparison of Case 6, 8, and 9 for {target} Prediction\nWith {n_features} Additional Features', fontsize=16)
    
    # Set equal aspect and limits
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add a grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    # Add horizontal and vertical lines at ACE-km = 10 (cutoff used in case5)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='ACE-km = 10 cutoff')
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    
    # Save the plot
    plot_filename = f"results_biomass/plots/combined_comparison_{target}_n{n_features}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {plot_filename}")
    plt.close()

def create_biomass_histograms(df, target_column, biomass_cols):
    """
    Create histograms showing the distribution of target values across different biomass types
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    target_column : str
        Column name for the target variable (e.g., 'ACE-km')
    biomass_cols : list
        List of biomass column names
    """
    if not biomass_cols:
        print("No biomass columns found for histogram creation")
        return
    
    # Setup the plot
    plt.figure(figsize=(15, 10))
    
    # Define color palette for different biomass types and unknown
    colors = ['#1f77b4', '#ff7f0e', '#808080']  # blue for F, orange for G, gray for unknown
    
    # Get the number of bins based on the data range and sample size
    data_range = df[target_column].max() - df[target_column].min()
    n_bins = max(5, min(20, int(np.sqrt(len(df)))))  # Between 5 and 20 bins, based on sample size
    
    # Create bins with edges that align across all histograms
    bin_edges = np.linspace(df[target_column].min(), df[target_column].max(), n_bins + 1)
    
    # Determine which biomass types have samples
    valid_biomass_cols = []
    for col in biomass_cols:
        biomass_data = df[df[col] == 1][target_column]
        if len(biomass_data) > 0:
            valid_biomass_cols.append(col)
    
    # Check if there are unknown biomass samples
    unknown_mask = ~df[biomass_cols].any(axis=1)
    unknown_data = df.loc[unknown_mask, target_column]
    has_unknown = len(unknown_data) > 0
    
    # Calculate number of plots needed (all samples + valid biomass types + unknown if it exists)
    n_plots = 1 + len(valid_biomass_cols) + (1 if has_unknown else 0)
    
    # Create subplots only for data that exists
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    
    # If there's only one plot, axes won't be an array, so we need to handle it
    if n_plots == 1:
        axes = [axes]
    
    # Plot index counter
    plot_idx = 0
    
    # Plot the overall distribution first
    ax = axes[plot_idx]
    plot_idx += 1
    ax.hist(df[target_column], bins=bin_edges, alpha=0.7, color='green', edgecolor='black')
    ax.set_title(f'All Samples Distribution (n={len(df)})', fontsize=14)
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('Count', fontsize=12)
    
    # Add stats for the overall distribution
    stats_text = f"Mean: {df[target_column].mean():.2f}\n"
    stats_text += f"Median: {df[target_column].median():.2f}\n"
    stats_text += f"Min: {df[target_column].min():.2f}\n"
    stats_text += f"Max: {df[target_column].max():.2f}\n"
    stats_text += f"Std Dev: {df[target_column].std():.2f}"
    ax.annotate(stats_text, xy=(0.02, 0.70), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=10)
    
    # Calculate kernel density estimate for all data
    kde_x = np.linspace(df[target_column].min(), df[target_column].max(), 100)
    kde = stats.gaussian_kde(df[target_column])
    kde_y = kde(kde_x)
    
    # Create a twin axis for the KDE plot
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde_y, 'k-', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.grid(False)
    
    # Plot each valid biomass type
    for i, col in enumerate(valid_biomass_cols):
        ax = axes[plot_idx]
        plot_idx += 1
        biomass_type = col.split('_')[1]
        biomass_data = df[df[col] == 1][target_column]
        
        # Create histogram for this biomass type
        ax.hist(biomass_data, bins=bin_edges, alpha=0.7, color=colors[i % len(colors)], edgecolor='black')
        ax.set_title(f'Biomass {biomass_type} Distribution (n={len(biomass_data)})', fontsize=14)
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Count', fontsize=12)
        
        # Add stats for this distribution
        if len(biomass_data) >= 3:  # Only show stats if we have enough samples
            stats_text = f"Mean: {biomass_data.mean():.2f}\n"
            stats_text += f"Median: {biomass_data.median():.2f}\n"
            stats_text += f"Min: {biomass_data.min():.2f}\n"
            stats_text += f"Max: {biomass_data.max():.2f}\n"
            stats_text += f"Std Dev: {biomass_data.std():.2f}"
            ax.annotate(stats_text, xy=(0.02, 0.70), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=10)
            
            # Add KDE if we have enough samples
            if len(biomass_data) >= 5:
                kde_x = np.linspace(biomass_data.min(), biomass_data.max(), 100)
                kde = stats.gaussian_kde(biomass_data)
                kde_y = kde(kde_x)
                
                # Create a twin axis for the KDE plot
                ax2 = ax.twinx()
                ax2.plot(kde_x, kde_y, 'k-', linewidth=2, alpha=0.7)
                ax2.set_ylabel('Density', fontsize=12)
                ax2.grid(False)
        
        # Add percentage of samples in different ranges
        total = len(biomass_data)
        below_10 = len(biomass_data[biomass_data <= 10])
        above_10 = len(biomass_data[biomass_data > 10])
        
        range_text = f"≤ 10: {below_10} samples ({below_10/total:.1%})\n"
        range_text += f"> 10: {above_10} samples ({above_10/total:.1%})"
        ax.annotate(range_text, xy=(0.75, 0.70), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=10)
    
    # Plot unknown biomass samples if any exist
    if has_unknown:
        ax = axes[plot_idx]
        
        # Create histogram for unknown biomass
        ax.hist(unknown_data, bins=bin_edges, alpha=0.7, color=colors[-1], edgecolor='black')
        ax.set_title(f'Unknown Biomass Distribution (n={len(unknown_data)})', fontsize=14)
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Count', fontsize=12)
        
        # Add stats for unknown biomass distribution
        if len(unknown_data) >= 3:
            stats_text = f"Mean: {unknown_data.mean():.2f}\n"
            stats_text += f"Median: {unknown_data.median():.2f}\n"
            stats_text += f"Min: {unknown_data.min():.2f}\n"
            stats_text += f"Max: {unknown_data.max():.2f}\n"
            stats_text += f"Std Dev: {unknown_data.std():.2f}"
            ax.annotate(stats_text, xy=(0.02, 0.70), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=10)
            
            # Add KDE if we have enough samples
            if len(unknown_data) >= 5:
                kde_x = np.linspace(unknown_data.min(), unknown_data.max(), 100)
                kde = stats.gaussian_kde(unknown_data)
                kde_y = kde(kde_x)
                
                # Create a twin axis for the KDE plot
                ax2 = ax.twinx()
                ax2.plot(kde_x, kde_y, 'k-', linewidth=2, alpha=0.7)
                ax2.set_ylabel('Density', fontsize=12)
                ax2.grid(False)
        
        # Add percentage of samples in different ranges
        total = len(unknown_data)
        below_10 = len(unknown_data[unknown_data <= 10])
        above_10 = len(unknown_data[unknown_data > 10])
        
        range_text = f"≤ 10: {below_10} samples ({below_10/total:.1%})\n"
        range_text += f"> 10: {above_10} samples ({above_10/total:.1%})"
        ax.annotate(range_text, xy=(0.75, 0.70), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=10)
    
    # Add x-label to the bottom plot
    axes[-1].set_xlabel(target_column, fontsize=14)
    
    # Add overall title and adjust layout
    plt.suptitle(f'Distribution of {target_column} Values by Biomass Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save the figure
    plot_filename = f"results_biomass/plots/biomass_histograms_{target_column}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved biomass histograms to: {plot_filename}")
    plt.close()
    
    # Create an additional stacked histogram to directly compare distributions
    plt.figure(figsize=(12, 8))
    
    # Prepare data for each category
    biomass_data = {}
    for col in biomass_cols:
        biomass_type = col.split('_')[1]
        data = df[df[col] == 1][target_column].tolist()
        if data:  # Only add if there's data
            biomass_data[biomass_type] = data
    
    # Add unknown biomass only if it exists
    if has_unknown:
        biomass_data['Unknown'] = unknown_data.tolist()
    
    # Only create plot if we have data
    if biomass_data:
        # Plot stacked histogram
        plt.hist(
            [biomass_data[k] for k in biomass_data.keys()],
            bins=bin_edges,
            stacked=True,
            alpha=0.7,
            label=list(biomass_data.keys()),
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add vertical line at ACE-km = 10
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
        
        # Configure plot
        plt.title(f'Stacked Histogram of {target_column} Values by Biomass Type', fontsize=16)
        plt.xlabel(target_column, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Biomass Type')
        
        # Save stacked histogram
        stacked_filename = f"results_biomass/plots/biomass_stacked_histogram_{target_column}.png"
        plt.savefig(stacked_filename, dpi=300, bbox_inches='tight')
        print(f"Saved stacked histogram to: {stacked_filename}")
    else:
        print("No data available for stacked histogram")
    
    plt.close()

def create_biomass_type_distribution_summary(data_path):
    """
    Create a summary of biomass type distribution in the dataset
    
    Parameters:
    ----------
    data_path : str
        Path to the data file
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nAnalyzing biomass distribution in: {data_path}")
    
    # Check for biomass columns
    biomass_cols = [col for col in df.columns if col.startswith('Biomass_')]
    if not biomass_cols:
        print("No biomass columns found in the dataset")
        return
    
    # Remove rows containing 'x'
    df_clean = df[~df.isin(['x']).any(axis=1)]
    print(f"Original samples: {len(df)}, Clean samples: {len(df_clean)}")
    
    # Convert all columns to numeric
    df_clean = df_clean.apply(pd.to_numeric)
    
    # Create plot to show biomass distribution
    plt.figure(figsize=(10, 6))
    
    # Count samples by biomass type
    biomass_counts = {}
    for col in biomass_cols:
        biomass_type = col.split('_')[1]
        count = df_clean[col].sum()
        biomass_counts[biomass_type] = count
    
    # Count samples with no biomass type
    no_biomass = (~df_clean[biomass_cols].any(axis=1)).sum()
    biomass_counts['Unknown'] = no_biomass
    
    # Create bar chart
    plt.bar(biomass_counts.keys(), biomass_counts.values(), color=['blue', 'orange', 'gray'])
    
    # Add count labels
    for i, (biomass_type, count) in enumerate(biomass_counts.items()):
        plt.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    # Configure plot
    plt.title('Biomass Type Distribution in Dataset', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    plot_filename = "results_biomass/plots/biomass_distribution_summary.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved biomass distribution summary to: {plot_filename}")
    plt.close()
    
    # Additional analysis of ACE-km values by biomass type
    plt.figure(figsize=(12, 6))
    
    # Create a boxplot of ACE-km values by biomass type
    data_to_plot = []
    labels = []
    
    for col in biomass_cols:
        biomass_type = col.split('_')[1]
        values = df_clean[df_clean[col] == 1]['ACE-km'].values
        if len(values) > 0:
            data_to_plot.append(values)
            labels.append(f'Biomass {biomass_type}')
    
    # Add unknown biomass samples
    unknown_mask = ~df_clean[biomass_cols].any(axis=1)
    unknown_values = df_clean.loc[unknown_mask, 'ACE-km'].values
    if len(unknown_values) > 0:
        data_to_plot.append(unknown_values)
        labels.append('Unknown')
    
    # Create boxplot
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Add horizontal line at ACE-km = 10
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='ACE-km = 10 cutoff')
    
    # Configure plot
    plt.title('ACE-km Distribution by Biomass Type', fontsize=14)
    plt.ylabel('ACE-km Value', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_filename = "results_biomass/plots/ace_km_distribution_by_biomass.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved ACE-km distribution by biomass to: {plot_filename}")
    plt.close()
    
    # Create histograms to show detailed distribution
    create_biomass_histograms(df_clean, 'ACE-km', biomass_cols)

if __name__ == "__main__":
    # Create directories first
    create_directories()
    
    # Define data path
    data_path = "../Data/New_data.csv"
    
    # Create summary of biomass distribution
    create_biomass_type_distribution_summary(data_path)
    
    # Define additional feature counts to test
    feature_counts = [50, 100, 200]
    
    # Store results for all cases
    all_results = {}
    biomass_metrics_by_case = {}
    
    # Run all three cases with each feature count
    for n_features in feature_counts:
        print(f"\n{'=' * 80}")
        print(f"RUNNING ANALYSIS WITH {n_features} ADDITIONAL FEATURES")
        print(f"{'=' * 80}")
        
        case_results = {}
        
        # Run Case 6: Biomass F with 1 acetoclastic feature
        print(f"\n{'#' * 60}")
        print(f"CASE 6: BIOMASS F WITH 1 ACETOCLASTIC FEATURE")
        print(f"{'#' * 60}")
        
        r2_6, mse_6, rmse_6, results_6, sample_info_6 = run_linearsvr_cv(
            data_path=data_path,
            target="ACE-km",
            mode="case6",
            n_features=n_features
        )
        
        if r2_6 is not None:
            case_results['case6'] = results_6
            all_results[f'case6_{n_features}'] = {
                'R2': r2_6,
                'MSE': mse_6,
                'RMSE': rmse_6,
                'Sample_Count': sample_info_6.get('final', 0),
                'Sample_to_Feature_Ratio': sample_info_6.get('sample_to_feature_ratio', 0)
            }
        
        # Run Case 8: Biomass G with all 8 features
        print(f"\n{'#' * 60}")
        print(f"CASE 8: BIOMASS G WITH ALL 8 CLASSIFIED FEATURES")
        print(f"{'#' * 60}")
        
        r2_8, mse_8, rmse_8, results_8, sample_info_8 = run_linearsvr_cv(
            data_path=data_path,
            target="ACE-km",
            mode="case8",
            n_features=n_features
        )
        
        if r2_8 is not None:
            case_results['case8'] = results_8
            all_results[f'case8_{n_features}'] = {
                'R2': r2_8,
                'MSE': mse_8,
                'RMSE': rmse_8,
                'Sample_Count': sample_info_8.get('final', 0),
                'Sample_to_Feature_Ratio': sample_info_8.get('sample_to_feature_ratio', 0)
            }
        
        # Run Case 9: All samples with 1 acetoclastic feature
        print(f"\n{'#' * 60}")
        print(f"CASE 9: ALL SAMPLES WITH 1 ACETOCLASTIC FEATURE")
        print(f"{'#' * 60}")
        
        r2_9, mse_9, rmse_9, results_9, sample_info_9, biomass_metrics = run_linearsvr_cv(
            data_path=data_path,
            target="ACE-km",
            mode="case9",
            n_features=n_features
        )
        
        if r2_9 is not None:
            case_results['case9'] = results_9
            all_results[f'case9_{n_features}'] = {
                'R2': r2_9,
                'MSE': mse_9,
                'RMSE': rmse_9,
                'Sample_Count': sample_info_9.get('final', 0),
                'Sample_to_Feature_Ratio': sample_info_9.get('sample_to_feature_ratio', 0)
            }
            
            # Store biomass metrics for Case 9
            biomass_metrics_by_case[f'case9_{n_features}'] = biomass_metrics
        
        # Create combined plot comparing all three cases
        if len(case_results) > 1:
            create_combined_biomass_plot(case_results, "ACE-km", n_features)
    
    # Print summary of all results
    print(f"\n{'=' * 80}")
    print("SUMMARY OF RESULTS")
    print(f"{'=' * 80}")
    
    # Create summary table
    summary_table = []
    for case_key, metrics in all_results.items():
        # Parse case and feature count from key
        case, n_features = case_key.split('_')
        
        summary_table.append({
            'Case': case,
            'Features': n_features,
            'Samples': metrics['Sample_Count'],
            'Sample/Feature': f"{metrics['Sample_to_Feature_Ratio']:.2f}",
            'R²': f"{metrics['R2']:.4f}",
            'MSE': f"{metrics['MSE']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}"
        })
    
    # Sort summary table by case and feature count
    summary_table = sorted(summary_table, key=lambda x: (x['Case'], int(x['Features'])))
    
    # Display summary table
    if summary_table:
        # Compute column widths
        col_widths = {col: max(len(str(row[col])) for row in summary_table) for col in summary_table[0].keys()}
        col_widths = {col: max(col_widths[col], len(col)) for col in col_widths}
        
        # Print header
        header = ' | '.join(f"{col:{col_widths[col]}}" for col in summary_table[0].keys())
        print('\n' + header)
        print('-' * len(header))
        
        # Print rows
        for row in summary_table:
            print(' | '.join(f"{str(row[col]):{col_widths[col]}}" for col in row.keys()))
    
    # Save summary table to CSV
    pd.DataFrame(summary_table).to_csv('results_biomass/metrics/summary_results.csv', index=False)
    print("\nSummary table saved to: results_biomass/metrics/summary_results.csv")
    
    print("\nAnalysis complete!")
    print(f"Results are saved in the 'results_biomass' directory.")
    print("Key findings:")
    print("1. Check if biomass types have distinct ACE-km distributions")
    print("2. Compare performance across biomass-specific models vs. combined model")
    print("3. Examine the performance plots to understand prediction patterns")
    print("4. Review biomass-specific metrics in case9 to see if certain biomass types are predicted better") 