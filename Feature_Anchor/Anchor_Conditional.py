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
        'results_conditional',
        'results_conditional/metrics',
        'results_conditional/plots'
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
    """Perform RFE while keeping anchored features"""
    non_anchored_features = [col for col in X.columns if col not in anchored_features]
    X_non_anchored = X[non_anchored_features]
    
    if n_features > 0 and len(non_anchored_features) > 0:
        estimator = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_non_anchored, y)
        
        selected_mask = rfe.support_
        selected_features = X_non_anchored.columns[selected_mask]
        final_model = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
        final_model.fit(X_non_anchored[selected_features], y)
        
        importance_scores = np.abs(final_model.coef_)
        feature_importance_pairs = list(zip(selected_features, importance_scores))
        sorted_features = [f for f, _ in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        sorted_importance = [i for _, i in sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)]
        
        final_features = anchored_features + sorted_features
        final_importance = [0] * len(anchored_features) + sorted_importance
    else:
        final_features = anchored_features
        final_importance = [0] * len(anchored_features)
    
    return final_features, final_importance

def create_performance_plot(all_actual, all_predictions, condition, n_features, final_r2, final_mse):
    """Create and save performance plot"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    # Set non-negative range for axes
    min_val = 0
    max_val = max(max(all_actual), max(all_predictions)) * 1.1  # Add 10% padding
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    condition_text = "< 10" if condition == "low" else "≥ 10"
    title = f"ACE-km Prediction (ACE-km {condition_text})\nUsing {n_features} Additional Features"
    metrics = f'R² = {final_r2:.4f}\nMSE = {final_mse:.4f}'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'results_conditional/plots/results_ACE_km_{condition}_{n_features}features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_plot(best_low, best_high):
    """Create combined plot for best configurations of both conditions"""
    plt.figure(figsize=(10, 8))
    
    # Plot low ACE-km points
    sns.scatterplot(x=best_low['Actual'], y=best_low['Predictions'], 
                   label='ACE-km < 10', alpha=0.6)
    
    # Plot high ACE-km points
    sns.scatterplot(x=best_high['Actual'], y=best_high['Predictions'], 
                   label='ACE-km ≥ 10', alpha=0.6)
    
    # Set non-negative range for axes
    min_val = 0
    max_val = max(
        max(best_low['Actual']), max(best_low['Predictions']),
        max(best_high['Actual']), max(best_high['Predictions'])
    ) * 1.1  # Add 10% padding
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Calculate combined R² and MSE
    all_actual = np.concatenate([best_low['Actual'], best_high['Actual']])
    all_predictions = np.concatenate([best_low['Predictions'], best_high['Predictions']])
    combined_r2 = r2_score(all_actual, all_predictions)
    combined_mse = mean_squared_error(all_actual, all_predictions)
    
    title = (f"Combined ACE-km Prediction\n"
            f"Low: {best_low['Additional_Features']} features, "
            f"High: {best_high['Additional_Features']} features")
    metrics = f'Combined R² = {combined_r2:.4f}\nCombined MSE = {combined_mse:.4f}'
    plt.title(f'{title}\n{metrics}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = 'results_conditional/plots/results_ACE_km_combined_best.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_r2, combined_mse

def create_detailed_summary(results_low, results_high):
    """Create detailed summary tables for both conditions"""
    # Create detailed summary for low ACE-km
    low_summary = pd.DataFrame({
        'Additional_Features': [],
        'R2': [],
        'MSE': [],
        'RMSE': [],
        'Selected_Features': [],
        'Feature_Importance': [],
        'Num_Total_Features': []
    })
    
    # Process low ACE-km results
    for n_features, result in results_low.items():
        features_str = ', '.join(result['Features'])
        importance_str = ', '.join([f"{imp:.4f}" for imp in result['Importance']])
        
        low_summary = pd.concat([low_summary, pd.DataFrame({
            'Additional_Features': [n_features],
            'R2': [result['R2']],
            'MSE': [result['MSE']],
            'RMSE': [result['RMSE']],
            'Selected_Features': [features_str],
            'Feature_Importance': [importance_str],
            'Num_Total_Features': [len(result['Features'])]
        })], ignore_index=True)
    
    # Create detailed summary for high ACE-km
    high_summary = pd.DataFrame({
        'Additional_Features': [],
        'R2': [],
        'MSE': [],
        'RMSE': [],
        'Selected_Features': [],
        'Feature_Importance': [],
        'Num_Total_Features': []
    })
    
    # Process high ACE-km results
    for n_features, result in results_high.items():
        features_str = ', '.join(result['Features'])
        importance_str = ', '.join([f"{imp:.4f}" for imp in result['Importance']])
        
        high_summary = pd.concat([high_summary, pd.DataFrame({
            'Additional_Features': [n_features],
            'R2': [result['R2']],
            'MSE': [result['MSE']],
            'RMSE': [result['RMSE']],
            'Selected_Features': [features_str],
            'Feature_Importance': [importance_str],
            'Num_Total_Features': [len(result['Features'])]
        })], ignore_index=True)
    
    # Sort by Additional_Features
    low_summary = low_summary.sort_values('Additional_Features').reset_index(drop=True)
    high_summary = high_summary.sort_values('Additional_Features').reset_index(drop=True)
    
    # Save to CSV
    low_summary.to_csv('results_conditional/metrics/detailed_summary_low.csv', index=False)
    high_summary.to_csv('results_conditional/metrics/detailed_summary_high.csv', index=False)
    
    return low_summary, high_summary

def run_conditional_analysis(data_path, feature_ranges):
    """Run analysis for both conditions with different feature ranges"""
    create_directories()
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df = df[~df.isin(['x']).any(axis=1)]
    df = df.apply(pd.to_numeric)
    
    # Remove outliers where ACE-km > 40
    df = df[df['ACE-km'] <= 40]
    
    # Load feature lists
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    # Get features
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    
    # Split data by condition
    mask_low = df['ACE-km'] < 10
    mask_high = df['ACE-km'] >= 10
    
    results_low = {}
    results_high = {}
    
    # Process low ACE-km cases
    print("\nProcessing low ACE-km cases (< 10)...")
    for n_features in feature_ranges['low']:
        print(f"\nTesting with {n_features} additional features")
        results = process_condition(
            df[mask_low],
            all_features,
            'ACE-km',
            acetoclastic,  # Only acetoclastic features for low case
            n_features,
            'low'
        )
        results_low[n_features] = {
            'R2': results['R2'],
            'MSE': results['MSE'],
            'RMSE': results['RMSE'],
            'Features': results['Features'],
            'Importance': results['Importance'],
            'Predictions': results['Predictions'],
            'Actual': results['Actual']
        }
    
    # Process high ACE-km cases
    print("\nProcessing high ACE-km cases (≥ 10)...")
    for n_features in feature_ranges['high']:
        print(f"\nTesting with {n_features} additional features")
        results = process_condition(
            df[mask_high],
            all_features,
            'ACE-km',
            acetoclastic + hydrogenotrophic + syntrophic,  # All classified features for high case
            n_features,
            'high'
        )
        results_high[n_features] = {
            'R2': results['R2'],
            'MSE': results['MSE'],
            'RMSE': results['RMSE'],
            'Features': results['Features'],
            'Importance': results['Importance'],
            'Predictions': results['Predictions'],
            'Actual': results['Actual']
        }
    
    # Save results to separate CSV files
    save_results(results_low, 'low')
    save_results(results_high, 'high')
    
    # Find best configurations
    best_low = find_best_config(results_low)
    best_high = find_best_config(results_high)
    
    # Save combined best results
    save_combined_results(best_low, best_high)
    
    # Create combined plot for best configurations
    combined_r2, combined_mse = create_combined_plot(best_low, best_high)
    
    # Create detailed summary tables
    low_summary, high_summary = create_detailed_summary(results_low, results_high)
    
    print("\nDetailed summary tables have been saved to:")
    print("- results_conditional/metrics/detailed_summary_low.csv")
    print("- results_conditional/metrics/detailed_summary_high.csv")
    
    return best_low, best_high, combined_r2, combined_mse

def process_condition(data, all_features, target, anchored_features, n_features, condition):
    """Process each condition separately"""
    X = data[all_features]
    y = data[target]
    
    selected_features, importance = select_features_with_rfe(X, y, anchored_features, n_features)
    X = data[selected_features]
    
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
    
    r2 = r2_score(all_actual, all_predictions)
    mse = mean_squared_error(all_actual, all_predictions)
    rmse = np.sqrt(mse)
    
    # Create performance plot
    create_performance_plot(all_actual, all_predictions, condition, n_features, r2, mse)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Is_Anchored': [f in anchored_features for f in selected_features],
        'Feature_Importance': importance
    })
    feature_importance.to_csv(
        f'results_conditional/metrics/features_ACE_km_{condition}_{n_features}features.csv',
        index=False
    )
    
    return {
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Features': selected_features,
        'Importance': importance,
        'Predictions': all_predictions,
        'Actual': all_actual
    }

def save_results(results, condition):
    """Save results for each condition"""
    metrics = {
        'Additional_Features': [],
        'R2': [],
        'MSE': [],
        'RMSE': []
    }
    
    for n_features, result in results.items():
        metrics['Additional_Features'].append(n_features)
        metrics['R2'].append(result['R2'])
        metrics['MSE'].append(result['MSE'])
        metrics['RMSE'].append(result['RMSE'])
    
    df = pd.DataFrame(metrics)
    df.to_csv(f'results_conditional/metrics/results_summary_{condition}.csv', index=False)

def find_best_config(results):
    """Find best configuration based on R2 score"""
    best_r2 = -float('inf')
    best_config = None
    
    for n_features, result in results.items():
        if result['R2'] > best_r2:
            best_r2 = result['R2']
            best_config = {
                'Additional_Features': n_features,
                'R2': result['R2'],
                'MSE': result['MSE'],
                'RMSE': result['RMSE'],
                'Features': result['Features'],
                'Importance': result['Importance'],
                'Predictions': result['Predictions'],
                'Actual': result['Actual']
            }
    
    return best_config

def save_combined_results(best_low, best_high):
    # Implementation of save_combined_results function
    pass

if __name__ == "__main__":
    data_path = "../Data/New_data.csv"
    
    # Same feature ranges for both conditions
    feature_ranges = {
        'low': [50, 100, 200, 400, 800],
        'high': [50, 100, 200, 400, 800]
    }
    
    best_low, best_high, combined_r2, combined_mse = run_conditional_analysis(data_path, feature_ranges)
    
    print("\nBest Configuration for Low ACE-km (< 10):")
    print(f"Additional Features: {best_low['Additional_Features']}")
    print(f"R2: {best_low['R2']:.4f}")
    print(f"MSE: {best_low['MSE']:.4f}")
    
    print("\nBest Configuration for High ACE-km (≥ 10):")
    print(f"Additional Features: {best_high['Additional_Features']}")
    print(f"R2: {best_high['R2']:.4f}")
    print(f"MSE: {best_high['MSE']:.4f}")
    
    print("\nCombined Performance:")
    print(f"R2: {combined_r2:.4f}")
    print(f"MSE: {combined_mse:.4f}")