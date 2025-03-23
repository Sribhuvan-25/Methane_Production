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
    directories = [
        'results_anchor_no_outliers',
        'results_anchor_no_outliers/metrics',
        'results_anchor_no_outliers/plots'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_feature_lists():
    acetoclastic = pd.read_csv('features/acetoclastic_features.csv')['feature'].tolist()
    hydrogenotrophic = pd.read_csv('features/hydrogenotrophic_features.csv')['feature'].tolist()
    syntrophic = pd.read_csv('features/syntrophic_features.csv')['feature'].tolist()
    return acetoclastic, hydrogenotrophic, syntrophic

def select_features_with_rfe(X, y, anchored_features, n_features=5):
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

def create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_actual, y=all_predictions, alpha=0.6)
    
    min_val = 0
    max_val = max(max(all_actual), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
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
    
    filename = f'results_anchor_no_outliers/plots/results_{target}_{mode}_{n_features}features.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_linearsvr_cv(data_path, target="ACE-km", mode="ace_only", n_features=800):
    create_directories()
    
    df = pd.read_csv(data_path)
    
    # Remove rows containing 'x'
    df = df[~df.isin(['x']).any(axis=1)]
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)
    
    # Remove outliers only for ACE-km related predictions
    if target == "ACE-km":
        df = df[df['ACE-km'] <= 40]
    
    acetoclastic, hydrogenotrophic, syntrophic = load_feature_lists()
    
    target_columns = ["Average-Total-ISD-Cells", "ACE-Xi", "ACE-km", "ACE-Ks", 
                     "H2-Xi", "H2-km", "H2-Ks"]
    all_features = [col for col in df.columns if col not in target_columns]
    
    if mode == "h2":
        anchored_features = hydrogenotrophic
    elif mode == "ace_only":
        anchored_features = acetoclastic
    elif mode == "ace_all":
        anchored_features = acetoclastic + hydrogenotrophic + syntrophic
    elif mode == "ace_conditional":
        anchored_features = acetoclastic
    else:
        raise ValueError("Invalid mode specified")

    if mode != "ace_conditional":
        selected_features, importance = select_features_with_rfe(
            df[all_features], 
            df[target], 
            anchored_features, 
            n_features
        )
        X = df[selected_features]
    else:
        X = df[all_features]
    
    y = df[target]
    
    print(f"\nDataset shape after cleaning and removing outliers: {df.shape}")
    print(f"Number of features used: {len(X.columns)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = []
    all_actual = []
    
    model = LinearSVR(random_state=42, max_iter=2000, tol=1e-4)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if mode == "ace_conditional":
            mask_train_low = y_train < 10
            mask_train_high = y_train >= 10
            mask_val_low = y_val < 10
            mask_val_high = y_val >= 10
            
            print(f"\nFold {fold+1}:")
            print(f"Training samples with ACE-km < 10: {sum(mask_train_low)}")
            print(f"Training samples with ACE-km ≥ 10: {sum(mask_train_high)}")
            print(f"Validation samples with ACE-km < 10: {sum(mask_val_low)}")
            print(f"Validation samples with ACE-km ≥ 10: {sum(mask_val_high)}")
            
            if sum(mask_train_low) > 0:
                features_low, importance_low = select_features_with_rfe(
                    X_train[mask_train_low],
                    y_train[mask_train_low],
                    acetoclastic,
                    n_features
                )
                
                X_train_low = X_train[mask_train_low][features_low]
                scaler_low = StandardScaler()
                X_train_low_scaled = scaler_low.fit_transform(X_train_low)
                
                model_low = LinearSVR(random_state=42, max_iter=100000, tol=1e-4, dual=True)
                model_low.fit(X_train_low_scaled, y_train[mask_train_low])
                
                if sum(mask_val_low) > 0:
                    X_val_low = X_val[mask_val_low][features_low]
                    X_val_low_scaled = scaler_low.transform(X_val_low)
                    y_pred_low = model_low.predict(X_val_low_scaled)
                    all_predictions.extend(y_pred_low)
                    all_actual.extend(y_val[mask_val_low])
            
            if sum(mask_train_high) > 0:
                features_high, importance_high = select_features_with_rfe(
                    X_train[mask_train_high],
                    y_train[mask_train_high],
                    acetoclastic + hydrogenotrophic + syntrophic,
                    n_features
                )
                
                X_train_high = X_train[mask_train_high][features_high]
                scaler_high = StandardScaler()
                X_train_high_scaled = scaler_high.fit_transform(X_train_high)
                
                model_high = LinearSVR(random_state=42, max_iter=100, tol=1e-4, dual=True)
                model_high.fit(X_train_high_scaled, y_train[mask_train_high])
                
                if sum(mask_val_high) > 0:
                    X_val_high = X_val[mask_val_high][features_high]
                    X_val_high_scaled = scaler_high.transform(X_val_high)
                    y_pred_high = model_high.predict(X_val_high_scaled)
                    all_predictions.extend(y_pred_high)
                    all_actual.extend(y_val[mask_val_high])
        
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            all_predictions.extend(y_pred)
            all_actual.extend(y_val)
            
            print(f"Fold {fold+1} completed")

    final_r2 = r2_score(all_actual, all_predictions)
    final_mse = mean_squared_error(all_actual, all_predictions)
    final_rmse = np.sqrt(final_mse)

    print(f"\nResults for {target} ({mode} mode):")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"RMSE: {final_rmse:.4f}")

    if mode == "ace_conditional":
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
        
        feature_importance = pd.concat([feature_importance_low, feature_importance_high], 
                                     ignore_index=True)
    else:
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Is_Anchored': [f in anchored_features for f in selected_features],
            'Feature_Importance': importance
        })
    
    feature_importance.to_csv(f'results_anchor_no_outliers/metrics/features_{target}_{mode}_{n_features}features.csv', index=False)

    create_performance_plot(all_actual, all_predictions, target, mode, final_r2, final_mse, n_features)

    results_df = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predictions
    })
    results_df.to_csv(f'results_anchor_no_outliers/metrics/results_{target}_{mode}_{n_features}features.csv', index=False)

    return final_r2, final_mse, final_rmse, results_df

if __name__ == "__main__":
    create_directories()
    
    data_path = "../Data/New_data.csv"
    additional_features = [50, 100, 200, 400, 800]
    all_results = {}
    
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
    
    print("\nRunning ACE-km conditional predictions...")
    for n_features in additional_features:
        r2_ace3, mse_ace3, rmse_ace3, results_ace3 = run_linearsvr_cv(
            data_path, 
            target="ACE-km",
            mode="ace_conditional",
            n_features=n_features
        )
        all_results[f'ACE_km_conditional_{n_features}'] = {
            'R2': r2_ace3,
            'MSE': mse_ace3,
            'RMSE': rmse_ace3,
            'Additional_Features': n_features
        }
    
    results_df = pd.DataFrame(all_results).T
    results_df.index = results_df.index.astype(str)
    results_df.index.name = 'Model_Configuration'
    results_df.to_csv('results_anchor_no_outliers/metrics/feature_comparison_results.csv')
    
    print("\nBest configurations based on R2 score:")
    for target in ['H2_km', 'ACE_km_aceonly', 'ACE_km_all', 'ACE_km_conditional']:
        target_results = results_df[results_df.index.str.startswith(target)]
        if not target_results.empty:
            best_config = target_results.loc[target_results['R2'].idxmax()]
            print(f"\n{target}:")
            print(f"Best R2: {best_config['R2']:.4f}")
            print(f"MSE: {best_config['MSE']:.4f}")
            print(f"Additional features: {best_config['Additional_Features']}") 