from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analyze_feature_importance(X, y, model):
    """Analyze feature importance using various methods."""
    # Train the model
    model.fit(X, y)
    
    # Get feature importance (works for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        # For linear models, use coefficients
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
    
    return importance

def create_polynomial_features(X, degree=2):
    """Create polynomial features."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly, columns=feature_names)

def handle_outliers(df, columns, n_std=3):
    """Remove outliers using z-score method."""
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < n_std]
    return df

def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation with multiple metrics."""
    metrics = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error'
    }
    
    results = {}
    for metric_name, scoring in metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    return results

def tune_hyperparameters(model, param_grid, X, y):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }

def analyze_residuals(model, X, y):
    """Analyze model residuals."""
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create residual plots
    plt.figure(figsize=(12, 4))
    
    # Residuals vs Predicted
    plt.subplot(121)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    # Residual distribution
    plt.subplot(122)
    plt.hist(residuals, bins=30)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Additional residual statistics
    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }

def check_multicollinearity(X):
    """Check for multicollinearity using correlation matrix."""
    correlation_matrix = X.corr()
    
    # Find highly correlated features
    threshold = 0.8
    high_correlation = np.where(np.abs(correlation_matrix) > threshold)
    high_correlation = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                       for x, y in zip(*high_correlation) if x != y and x < y]
    
    return pd.DataFrame(high_correlation, columns=['Feature 1', 'Feature 2', 'Correlation'])

# Example hyperparameter grids for different models
param_grids = {
    'linear': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}
