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

def create_directories():
    """Create necessary directories for outputs"""
    directories = [
        'results_dynamic',
        'results_dynamic/metrics',
        'results_dynamic/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    """Load the different feature lists from saved files"""
    acetoclastic = pd.read_csv('features/acetoclastic_features.csv')['feature'].tolist()
    hydrogenotrophic = pd.read_csv('features/hydrogenotrophic_features.csv')['feature'].tolist()
    syntrophic = pd.read_csv('features/syntrophic_features.csv')['feature'].tolist()
    return acetoclastic, hydrogenotrophic, syntrophic

def find_minimum_features(X, y, anchored_features, buffer_ratio=0.3):
    """
    Find minimum number of features needed to include all anchored features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    anchored_features : list
        List of features that must be included
    buffer_ratio : float
        Ratio of additional features to include beyond anchored features
    
    Returns:
    --------
    selected_features : list
        Final list of selected features
    importance : list
        Importance scores for selected features
    n_features : int
        Total number of features selected
    """
    # Initialize RFE with max features
    estimator = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
    n_features_total = X.shape[1]
    
    # First, run RFE with all features to get ranking
    rfe = RFE(estimator=estimator, n_features_to_select=1)  # Start with minimum features
    rfe.fit(X, y)
    
    # Get feature ranking
    feature_ranks = pd.DataFrame({
        'feature': X.columns,
        'rank': rfe.ranking_
    }).sort_values('rank')
    
    # Find the worst rank among anchor features
    max_anchor_rank = feature_ranks[feature_ranks['feature'].isin(anchored_features)]['rank'].max()
    
    # Calculate total features needed (including buffer)
    min_features_needed = max_anchor_rank
    buffer_features = int(min_features_needed * buffer_ratio)
    total_features = min_features_needed + buffer_features
    
    # Run final RFE with calculated number of features
    final_rfe = RFE(estimator=estimator, n_features_to_select=total_features)
    final_rfe.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[final_rfe.support_].tolist()
    
    # Fit model on selected features to get importance scores
    final_model = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
    final_model.fit(X[selected_features], y)
    importance = np.abs(final_model.coef_)
    
    return selected_features, importance, total_features

def create_performance_plot(all_actual, all_predictions, mode, n_features, final_r2, final_mse):
    """Create and save performance plot"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    # Set non-negative range for axes
    min_val = 0
    max_val = max(max(all_actual), max(all_predictions)) * 1.1
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    if mode == "h2":
        title = f"H2-km Prediction\nUsing {n_features} Features (with Hydrogenotrophic Anchoring)"
    elif mode == "ace_only":
        title = f"ACE-km Prediction\nUsing {n_features} Features (with Acetoclastic Anchoring)"
    else:  # ace_all
        title = f"ACE-km Prediction\nUsing {n_features} Features (with All Classified Anchoring)"
    
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'results_dynamic/plots/results_{mode}_{n_features}features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def process_prediction(X, y, anchored_features, mode):
    """Process predictions for a given mode"""
    # Find minimum features needed
    selected_features, importance, n_features = find_minimum_features(X, y, anchored_features)
    X = X[selected_features]
    
    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = []
    all_actual = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = LinearSVR(random_state=42, max_iter=10000, tol=1e-4, dual=True)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        all_predictions.extend(y_pred)
        all_actual.extend(y_val)
    
    # Calculate metrics
    r2 = r2_score(all_actual, all_predictions)
    mse = mean_squared_error(all_actual, all_predictions)
    rmse = np.sqrt(mse)
    
    # Create performance plot
    create_performance_plot(all_actual, all_predictions, mode, n_features, r2, mse)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Is_Anchored': [f in anchored_features for f in selected_features],
        'Feature_Importance': importance
    })
    feature_importance.to_csv(
        f'results_dynamic/metrics/features_{mode}.csv',
        index=False
    )
    
    return {
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Features': selected_features,
        'Importance': importance,
        'N_Features': n_features,
        'Predictions': all_predictions,
        'Actual': all_actual
    }

def run_dynamic_analysis(data_path):
    """Run analysis for all non-conditional cases with dynamic feature selection"""
    create_directories()
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df = df[~df.isin(['x']).any(axis=1)]
    df = df.apply(pd.to_numeric)
    
    # Load feature lists
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    # Get features
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    X = df[all_features]
    
    results = {}
    
    # H2-km prediction
    print("\nProcessing H2-km prediction...")
    results['h2'] = process_prediction(
        X,
        df['H2-km'],
        hydrogenotrophic,
        'h2'
    )
    
    # ACE-km with acetoclastic only
    print("\nProcessing ACE-km with acetoclastic features...")
    results['ace_only'] = process_prediction(
        X,
        df['ACE-km'],
        acetoclastic,
        'ace_only'
    )
    
    # ACE-km with all classified features
    print("\nProcessing ACE-km with all classified features...")
    results['ace_all'] = process_prediction(
        X,
        df['ACE-km'],
        acetoclastic + hydrogenotrophic + syntrophic,
        'ace_all'
    )
    
    # Create summary table
    summary = pd.DataFrame({
        'Mode': ['H2-km', 'ACE-km (Acetoclastic)', 'ACE-km (All Classified)'],
        'N_Features': [
            results['h2']['N_Features'],
            results['ace_only']['N_Features'],
            results['ace_all']['N_Features']
        ],
        'R2': [
            results['h2']['R2'],
            results['ace_only']['R2'],
            results['ace_all']['R2']
        ],
        'MSE': [
            results['h2']['MSE'],
            results['ace_only']['MSE'],
            results['ace_all']['MSE']
        ],
        'RMSE': [
            results['h2']['RMSE'],
            results['ace_only']['RMSE'],
            results['ace_all']['RMSE']
        ]
    })
    
    summary.to_csv('results_dynamic/metrics/summary.csv', index=False)
    
    return results, summary

if __name__ == "__main__":
    data_path = "../Data/New_data.csv"
    results, summary = run_dynamic_analysis(data_path)
    
    print("\nResults Summary:")
    print(summary.to_string(index=False)) 