# Scientific Advice for Time Series Forecasting Submissions

## Overview
This document provides comprehensive scientific guidance on metrics, evaluation methodologies, and visualization best practices for academic time series forecasting competitions, specifically tailored for your multi-horizon financial time series forecasting project.

## Key Metrics for Time Series Forecasting

### Primary Competition Metric: Weighted RMSE
**Why it's important:**
- The official Kaggle metric for this competition
- Accounts for different weights across time periods and horizons
- Penalizes larger errors more heavily due to squaring
- Standard in financial forecasting where large errors have significant impact

**Interpretation:**
- Lower values indicate better performance
- Values typically range from 0.01 to 0.10 for financial time series
- Sensitive to outliers and extreme predictions

### Secondary Metrics for Academic Rigor

#### 1. Pearson Correlation
**Purpose:** Measures linear relationship between predictions and actual values
**Why include:**
- Complements RMSE by capturing directional accuracy
- Robust to scale differences
- Standard in financial econometrics
- Values range from -1 to 1 (higher is better)

#### 2. R² (Coefficient of Determination)
**Purpose:** Explains proportion of variance in target variable
**Why include:**
- Fundamental regression metric
- Indicates model explanatory power
- Values typically 0.001 to 0.050 for financial data (low is normal)

#### 3. Directional Accuracy
**Purpose:** Measures percentage of correct direction predictions
**Why include:**
- Critical for trading applications
- Captures practical utility beyond numerical accuracy
- Values around 0.50-0.60 are typical (0.50 = random)

#### 4. Mean Absolute Error (MAE)
**Purpose:** Average absolute prediction error
**Why include:**
- More robust to outliers than RMSE
- Provides interpretable error magnitude
- Complementary to RMSE

#### 5. MAPE and SMAPE
**Purpose:** Percentage-based error metrics
**Why include:**
- Scale-independent error measurement
- Useful for comparing across different price levels
- Note: Can be misleading with near-zero targets

## Recommended Evaluation Methodology

### 1. Multi-Horizon Analysis
**Approach:** Evaluate each horizon (H1, H3, H10, H25) separately
**Rationale:**
- Different horizons have different prediction difficulty
- Short-term vs long-term forecasting requires different considerations
- Allows identification of model strengths across time scales

### 2. Statistical Significance Testing
**Recommended Tests:**
- **Diebold-Mariano Test:** Compare forecast accuracy between models
- **Wilcoxon Signed-Rank Test:** Non-parametric comparison of prediction errors
- **Bootstrap Confidence Intervals:** Assess uncertainty in performance metrics

### 3. Cross-Validation Strategy
**For Time Series:**
- **Walk-Forward Validation:** Most appropriate for financial data
- **Time Series Split:** Preserves temporal ordering
- **Avoid:** Random cross-validation (breaks temporal dependencies)

## Visualization Best Practices

### 1. Performance Heatmaps
**Purpose:** Visual comparison across models, horizons, and metrics
**Best Practices:**
- Use diverging colormaps (RdYlBu) for balanced visualization
- Annotate cells with exact values
- Include color bars with clear labels
- Use consistent scaling across similar metrics

### 2. Prediction Distribution Plots
**Purpose:** Understand prediction characteristics and uncertainty
**Recommended Visualizations:**
- **Violin Plots:** Show distribution shape and quartiles
- **Box Plots:** Highlight outliers and central tendency
- **Histogram Overlays:** Compare prediction vs actual distributions
- **Q-Q Plots:** Assess normality of prediction errors

### 3. Learning Curves
**Purpose:** Track model convergence and training efficiency
**Key Elements:**
- Training vs validation loss curves
- Early stopping points
- Convergence speed comparison
- Overfitting indicators

### 4. Feature Importance Visualizations
**Purpose:** Explain model decisions and feature contributions
**Methods:**
- **SHAP Summary Plots:** Global and local feature importance
- **Feature Importance Bar Charts:** Traditional tree-based importance
- **Correlation Heatmaps:** Feature-target relationships

### 5. Time Series Plots
**Purpose:** Visual inspection of prediction quality
**Essential Elements:**
- Actual vs predicted values over time
- Prediction intervals (confidence bands)
- Residual plots over time
- Cumulative error analysis

## Academic Submission Structure

### 1. Abstract and Introduction
- Clearly state forecasting horizons
- Emphasize multi-horizon approach
- Highlight methodological innovations

### 2. Methodology Section
**Essential Elements:**
- Data preprocessing and feature engineering
- Model architecture and hyperparameters
- Training procedure and validation strategy
- Feature selection methodology (SHAP analysis)

### 3. Results Section
**Required Tables:**
- Performance metrics by horizon
- Statistical significance tests
- Feature importance rankings
- Training efficiency comparison

**Required Figures:**
- Performance heatmaps
- Prediction distribution comparisons
- Learning curves
- Time series prediction examples

### 4. Discussion Section
**Key Points:**
- Interpretation of multi-horizon performance differences
- Comparison with baseline methods
- Practical implications for financial forecasting
- Limitations and future work

## Specific Recommendations for Your Project

### 1. Model Comparison Strategy
**Recommended Approach:**
- Primary focus on Weighted RMSE (competition metric)
- Secondary emphasis on Pearson correlation and Directional Accuracy
- Include R² for explanatory power assessment
- Report all metrics with confidence intervals

### 2. Feature Engineering Documentation
**Essential to Document:**
- SHAP feature selection process
- Target encoding methodology
- Time feature engineering
- BNN integration approach

### 3. Model Naming Convention
**Clear Naming:**
- Baseline LGBM (raw features)
- LGBM SHAP-10 (top 10 SHAP features)
- LGBM All+SHAP (raw + engineered SHAP-10)
- LGBM BNN-SHAP10 (BNN predictions + SHAP-10)
- LGBM BNN-Aggregated (BNN aggregated predictions)
- Algorithm variants (XGBoost, CatBoost)
- Walk-forward variants (Trio models)

### 4. Statistical Analysis
**Recommended Tests:**
```python
# Diebold-Mariano test for forecast comparison
from scipy import stats
import numpy as np

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for forecast accuracy
    e1, e2: forecast error series
    h: forecast horizon
    """
    d = e1**2 - e2**2  # Loss differential
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = d_bar / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
```

### 5. Visualization Package
**Recommended Python Libraries:**
- `matplotlib` + `seaborn`: Core plotting
- `plotly`: Interactive visualizations
- `shap`: Feature importance plots
- `statsmodels`: Statistical analysis plots

## Submission Quality Checklist

### Metrics Reporting
- [ ] Weighted RMSE for all models and horizons
- [ ] Pearson correlation with p-values
- [ ] R² with confidence intervals
- [ ] Directional accuracy
- [ ] MAE and MAPE for completeness
- [ ] Statistical significance test results

### Visualizations
- [ ] Performance heatmaps for all metrics
- [ ] Prediction distribution comparisons
- [ ] Learning curves showing convergence
- [ ] Feature importance visualizations
- [ ] Time series prediction examples
- [ ] Residual analysis plots

### Documentation
- [ ] Clear model descriptions
- [ ] Feature engineering methodology
- [ ] Training and validation procedures
- [ ] Hyperparameter settings
- [ ] Computational requirements
- [ ] Reproducibility information

## Final Recommendations

### 1. Primary Focus Areas
- **Weighted RMSE optimization** for competition success
- **Multi-horizon analysis** for comprehensive evaluation
- **Statistical significance** for scientific rigor
- **Clear visualization** for effective communication

### 2. Secondary Considerations
- **Computational efficiency** for practical deployment
- **Feature interpretability** for domain understanding
- **Model robustness** across different market conditions
- **Reproducibility** for academic standards

### 3. Presentation Style
- **Professional academic tone**
- **Clear, concise writing**
- **Well-structured tables and figures**
- **Comprehensive methodological detail**
- **Thorough discussion of limitations**

This scientific advice framework should guide your submission preparation and ensure academic rigor while maximizing competition performance.
