# Time Series Forecasting Ablation Study

A comprehensive framework for multi-horizon time series forecasting with SHAP-based feature selection and ensemble methods. Developed as part of the NCI PGDAI_SEP25 academic project.

## Project Structure

```
ts_forecasting_ablation/
|
|--- run.py                      # Main entry point (vertical slice)
|
|--- config/                      # Configuration files
|    |--- horizons.yaml          # Horizon and validation settings
|    |--- models.yaml            # Model configurations and parameters
|    |--- paths.yaml             # Data paths and environment settings
|
|--- src/                         # Core modules
|    |--- __init__.py
|    |--- data_loader.py         # Multi-source data loading (Kaggle/local/URL)
|    |--- preprocessor.py        # Data cleaning and preprocessing
|    |--- feature_selector.py    # SHAP-based feature selection
|    |--- validator.py            # Walk-forward validation
|    |--- ensemble.py            # Ensemble creation and management
|    |--- evaluator.py            # Model evaluation and visualization
|    |--- utils.py               # Utility functions
|    |
|    |--- models/                 # Model implementations
|    |   |--- __init__.py
|    |   |--- lgbm_base.py        # LightGBM baseline
|    |   |--- lgbm_shap.py        # LightGBM with SHAP features
|    |   |--- xgb.py              # XGBoost baseline
|    |   |--- catboost.py         # CatBoost baseline
|    |   |--- bnn_pre.py          # BNN with pre-SHAP aggregation
|    |   |--- bnn_post.py         # BNN with post-SHAP uncertainty
|    |   |--- mamba.py            # Mamba model (negative control)
|
|--- notebooks/                   # Analysis notebooks
|    |--- 01_eda_widgets.ipynb   # Exploratory data analysis
|    |--- 02_ablation_study.ipynb # Ablation study analysis
|    |--- 03_presentation.ipynb  # Results presentation
|
|--- data/                        # Data storage
|--- results/                     # Results and outputs
|    |--- models/                 # Trained models
|    |--- shap_plots/             # SHAP visualizations
|    |--- predictions/            # Model predictions
|    |--- comparisons/            # Model comparisons
|
|--- artifacts/                   # Final paper outputs
|    |--- figures/                # Publication figures
|    |--- tables/                 # Result tables
|
|--- requirements.txt             # Python dependencies
|--- README.md                    # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
cd ts_forecasting_ablation

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run complete ablation study
python run.py --mode=full

# Run specific components
python run.py --mode=data      # Data loading and preprocessing
python run.py --mode=models    # Model training and evaluation
python run.py --mode=ensemble  # Ensemble creation only
```

## Research Questions

This study addresses three main research questions:

1. **Feature Selection Impact**: How does SHAP-based feature selection affect model performance and interpretability in time series forecasting?

2. **Ensemble Strategy Effectiveness**: What is the optimal ensemble strategy for multi-horizon time series forecasting?

3. **Model Complexity vs Performance**: How does model complexity impact prediction accuracy and computational efficiency?

## Methodology

The project follows CRISP-DM methodology:

1. **Business Understanding**: Multi-horizon financial forecasting
2. **Data Understanding**: Time series analysis with horizons [1, 3, 10, 25]
3. **Data Preparation**: SHAP-based feature selection and preprocessing
4. **Modeling**: LightGBM variants, XGBoost, CatBoost, BNN, Mamba
5. **Evaluation**: Weighted RMSE, statistical significance testing
6. **Deployment**: Modular pipeline with reproducible results

## Key Features

- **Environment Detection**: Automatic Kaggle/local/URL data loading
- **SHAP Integration**: Feature selection with interpretability
- **Ablation Study**: Systematic component analysis
- **Ensemble Methods**: Multiple weighting strategies
- **Walk-forward Validation**: Time-appropriate validation
- **Modular Design**: Git-ready, reproducible code

## Academic Requirements

This project meets all NCI PGDAI_SEP25 requirements:

- **3-4 ML Methods**: LightGBM, XGBoost, CatBoost, BNN, Mamba
- **CRISP-DM Methodology**: Full implementation
- **Multiple Metrics**: Weighted RMSE, Pearson, R², statistical tests
- **IEEE Format**: 8-10 page academic paper ready
- **Reproducibility**: Modular code with version control

## Results

Key findings from the study:

- **Best Model**: Weighted ensemble (80% v3_no_ultra + 20% top20_engineered)
- **Performance**: 0.1805 weighted RMSE score
- **SHAP Impact**: Significant noise reduction and performance improvement
- **Ensemble Benefit**: +0.0007 improvement over best single model

## Configuration

Key configuration files:

- `config/horizons.yaml`: Horizon settings and validation splits
- `config/models.yaml`: Model parameters and feature selection
- `config/paths.yaml`: Data paths and environment detection

## Contributing

1. Follow the modular structure
2. Add comprehensive documentation
3. Include unit tests for new features
4. Update configuration files as needed

## License

Developed for academic purposes as part of NCI PGDAI_SEP25 course requirements.

---

**Author**: Tomasz Biel  
**Email**: x25113186@student.ncirl.ie  
**Institution**: National College of Ireland  
**Course**: PGDAI_SEP25  
**Date**: April 2026
