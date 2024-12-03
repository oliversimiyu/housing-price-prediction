# Housing Price Prediction Analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('HousingDataCategorical.csv')

# Descriptive Analysis
def perform_descriptive_analysis(df):
    # Basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    print("\nNumerical Columns Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

# Data Cleaning
def clean_data(df):
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # For numeric columns, fill NA with median
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill NA with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Feature Selection and Model Training
def train_models(df):
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Evaluate
        results[name] = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
    
    return results

# Main execution
if __name__ == "__main__":
    # Load and analyze data
    df = pd.read_csv('HousingDataCategorical.csv')
    
    # Perform descriptive analysis
    df = perform_descriptive_analysis(df)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Train and evaluate models
    results = train_models(df_cleaned)
    
    # Print results
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Training R² Score: {metrics['train_r2']:.4f}")
        print(f"Testing R² Score: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Testing RMSE: {metrics['test_rmse']:.4f}")
