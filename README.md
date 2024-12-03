# Housing Price Prediction Analysis

This project implements various machine learning algorithms to analyze and predict housing prices using a categorical dataset. The analysis includes basic descriptive statistics, data preprocessing, feature engineering, and multiple regression models.

## Project Structure

```
├── HousingDataCategorical.csv    # Dataset containing housing information
├── housing_analysis.py           # Main analysis script with basic models
├── advanced_analysis.py          # Advanced analysis techniques and model optimization
├── housing_analysis.ipynb        # Jupyter notebook with interactive analysis
└── requirements.txt              # Project dependencies
```

## Features

- **Data Analysis and Visualization**
  - Descriptive statistics
  - Feature correlation analysis
  - Data distribution visualization using seaborn and matplotlib

- **Data Preprocessing**
  - Handling missing values
  - Feature scaling
  - Label encoding for categorical variables
  - Outlier detection and handling

- **Machine Learning Models**
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor

- **Advanced Analysis Techniques**
  - Feature importance analysis
  - Polynomial feature creation
  - Cross-validation
  - Hyperparameter tuning using GridSearchCV
  - Residual analysis
  - Multicollinearity checking

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis
Run the main analysis script:
```bash
python housing_analysis.py
```

### Advanced Analysis
For advanced analysis and model optimization:
```bash
python advanced_analysis.py
```

### Interactive Analysis
Open the Jupyter notebook for interactive analysis:
```bash
jupyter notebook housing_analysis.ipynb
```

## Interactive Analysis in Google Colab

You can run the analysis interactively in Google Colab using this link:
[Open in Google Colab](https://colab.research.google.com/drive/12tkLBHQRK8m2SxCVGHyPpaNlouij69GH#scrollTo=aZzMMu6ZQLK3)

The Colab notebook includes:
- Complete data analysis pipeline
- Interactive visualizations
- Advanced machine learning techniques
- Real-time model training and evaluation

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- notebook

## Results

The analysis provides:
- Model performance metrics (MSE, R² score)
- Feature importance rankings
- Visualization of predictions vs actual values
- Detailed residual analysis
- Model optimization results

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.
