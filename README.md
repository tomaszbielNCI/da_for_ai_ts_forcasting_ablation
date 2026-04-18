# Time Series Forecasting - Advanced Ensemble Methods

## Overview
This project implements and evaluates advanced machine learning methodologies for time series forecasting in financial markets, developed for the **National College of Ireland PGDAI program** (Data Analytics for Artificial Intelligence, 70% final grade).

**Academic Context**: This research addresses complex time series forecasting challenges using real-world financial time series data characterized by high dimensionality and noise levels. The data source provided a challenging test case for developing novel methodologies.

## Key Research Contributions

### 1. **Weighted Ensemble Methodology**
- **Innovation**: Optimal weight combination of SHAP-augmented and engineered features
- **Academic Value**: Novel ensemble optimization framework for time series
- **Implementation**: Dynamic weight optimization across prediction horizons

### 2. **SHAP Feature Augmentation**
- **Innovation**: Using model explanations (SHAP) as additional input features
- **Academic Value**: First application of explainable AI for feature augmentation in time series
- **Methodology**: Systematic feature importance integration

### 3. **Multi-Algorithm Ensemble**
- **Innovation**: LightGBM + CatBoost algorithmic diversity
- **Academic Value**: Validation of algorithmic diversity benefits
- **Approach**: Complementary model combination strategies

### 4. **Bayesian Neural Network Integration**
- **Innovation**: Hybrid classical-neural approaches with uncertainty quantification
- **Academic Value**: Framework for neural network integration in time series
- **Contribution**: Feature aggregation for neural network compatibility

## Project Structure

```
da_for_ai_ts_forcasting_ablation/
|
|-- src/                          # Source code implementations
|   |-- models/                   # Model implementations
|   |   |-- lgbm_with_bnn.py     # Weighted ensemble with BNN features
|   |   |-- bnn_shap10.py        # BNN with SHAP10 features
|   |   |-- bnn_aggregated.py    # BNN with aggregated features
|   |   |-- lgbm_walk_forward.py # Walk-forward validation
|   |   `-- trio_walk_forward.py # Multi-model walk-forward
|   |-- features/                 # Feature engineering
|   |   |-- shap_features.py     # SHAP-based feature selection
|   |   `-- bnn_aggregated_features.py # Feature aggregation for BNN
|   |-- metrics/                  # Evaluation metrics
|   `-- preprocessor.py          # Data preprocessing
|
|-- artifacts/                     # Academic submission materials
|   |-- README.md                 # Comprehensive project overview
|   |-- academic_research_paper.md # IEEE-style paper structure
|   |-- final_model_comparison.md  # Detailed model analysis
|   |-- horizon_performance_analysis.md # Multi-horizon analysis
|   |-- presentation_slides.md     # 10-minute presentation structure
|   |-- academic_visualizations.py # Visualization generation
|   |-- performance_comparison.png # Performance charts
|   |-- horizon_analysis.png      # Horizon-specific analysis
|   |-- feature_importance_evolution.png # Feature importance trends
|   |-- academic_summary_table.png # Results summary table
|   `-- uncertainty_analysis.png  # BNN uncertainty quantification
|
|-- results/                       # Model outputs and predictions
|   `-- predictions/              # All prediction files
|       |-- kaggle_best/         # Best performing submissions
|       `-- lgbm_with_bnn/       # LGBM with BNN predictions
|
|-- config/                        # Configuration files
|   |-- horizons.yaml            # Horizon configurations
|   |-- models.yaml              # Model parameters
|   `-- paths.yaml               # Data paths
|
|-- notebooks/                    # Exploratory analysis
|-- requirements.txt              # Python dependencies
|-- run.py                        # Main execution script
`-- test_data_loader.py         # Data validation tests
```

## Academic Requirements Met

### Learning Outcomes (NCI PGDAI)
- **LO1**: Critical understanding of data analytics and ML concepts
- **LO2**: Data manipulation, exploration, and visualization tools
- **LO3**: ML application for business decision-making
- **LO4**: Graphical tools for comprehensive analytics processes
- **LO5**: Critical analysis and presentation of results

### IEEE Conference Paper Structure
- **8-10 pages**, double-column format (10% of learning outcomes)
- **150-250 word abstract** 
- **12-15 Scopus-indexed references**
- **CRISP-DM methodology application**
- **Multiple evaluation metrics** (RMSE, Pearson, Private Score, etc)

### Presentation Requirements
- **10 minutes maximum**, 7 slides 30 % of learning outcomes for Data Analytics in NCI
- **Methodology overview** and key findings
- **Academic and practical implications**

## Key Technical Innovations

### 1. **SHAP-Based Feature Augmentation**
- **Novel Methodology**: Using model explanations as additional input features
- **Implementation**: Systematic integration of SHAP values with original features
- **Benefit**: Captures non-linear relationships and feature interactions

### 2. **Dynamic Ensemble Weighting**
- **Horizon-Specific Optimization**: Different weights for different prediction horizons
- **Weight Strategy**: Gradual shift from SHAP to engineered features for longer horizons
- **Innovation**: Systematic weight optimization framework

### 3. **Feature Aggregation for Neural Networks**
- **Correlation-Based Grouping**: 9 feature groups based on correlation analysis
- **Aggregation Method**: Mean and standard deviation per group
- **Purpose**: Reduce dimensionality for neural network compatibility

## Methodology Insights

### Horizon-Specific Strategies
- **Short-term (H1)**: Recent observations dominate, 85% SHAP feature weight
- **Medium-term (H3)**: Balanced feature set, 80% SHAP feature weight  
- **Long-term (H10)**: Engineered features increase importance, 75% SHAP weight
- **Very long-term (H25)**: Long-term trends critical, 70% SHAP weight

### Feature Engineering Evolution
- **Recent lags**: Most important for short-term predictions
- **Rolling statistics**: Gain importance with longer horizons
- **SHAP values**: Provide consistent benefits across all horizons
- **Feature aggregation**: Essential for neural network approaches

## Data Context

### Dataset Characteristics
- **Source**: Financial time series data
- **Size**: 1.4M+ observations with 50+ engineered features  
- **Horizons**: Multiple prediction periods (1, 3, 10, 25)
- **Challenge**: High noise levels and complex temporal dependencies

### Data Processing
- **Feature Engineering**: Lag features, rolling statistics, deltas
- **SHAP Integration**: Model explanations as additional features
- **Feature Aggregation**: 9 correlation-based groups for neural networks
- **Validation**: Walk-forward methodology for temporal data

## Academic Impact

### Research Contributions
1. **Novel SHAP augmentation methodology** for time series forecasting
2. **Systematic ensemble weight optimization** framework
3. **Neural-classical integration** approaches with uncertainty quantification
4. **Multi-horizon analysis** methodology for temporal forecasting

### Practical Applications
- **Financial forecasting** with uncertainty quantification
- **Risk assessment** using BNN prediction intervals
- **Explainable AI** integration in time series models
- **Ensemble optimization** for improved accuracy

### Future Research Directions
1. **Automated weight learning** for ensemble optimization
2. **Dynamic feature selection** adapting to market conditions
3. **Real-time implementation** of SHAP augmentation
4. **Cross-domain validation** in non-financial time series

## Academic Deliverables

### Research Documentation
- **Main Paper**: Complete IEEE-format research paper structure
- **Technical Analysis**: Detailed methodology and implementation
- **Performance Evaluation**: Comprehensive multi-horizon analysis
- **Visualizations**: Academic-ready charts and tables

### Implementation Artifacts
- **Source Code**: Complete, documented implementations
- **Configuration**: Model parameters and data processing settings
- **Results**: All model outputs and analysis files
- **Documentation**: Comprehensive project documentation

---

*This project demonstrates advanced machine learning methodologies for time series forecasting, with significant academic contributions in ensemble optimization, feature engineering, and neural network integration.*
