# Model Performance Metrics Tables

## Complete Models Results Summary

### Table 1: LGBM with BNN Variants - Validation Metrics

| Horizon | Model | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|-------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | LGBM Baseline | 0.015764 | 0.052687 | 10.311111 | 2.357590 | 0.002879 | 0.548520 | 152.864 | 181.845 | 12 | 10 |
| H1 | LGBM BNN-SHAP10 | 0.018180 | 0.026406 | 10.314146 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 12 |
| H1 | LGBM BNN-Agg | 0.014189 | 0.039547 | 10.312380 | 2.357590 | 0.001345 | 0.512662 | 116.176 | 186.407 | 17 | 12 |
| H3 | LGBM Baseline | 0.027442 | 0.031247 | 17.415736 | 4.168008 | 0.002030 | 0.538295 | 144.802 | 179.555 | 12 | 10 |
| H3 | LGBM BNN-SHAP10 | 0.033590 | 0.025447 | 17.418436 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 11 | 12 |
| H3 | LGBM BNN-Agg | 0.029724 | 0.031708 | 17.415478 | 4.168008 | 0.001001 | 0.528852 | 104.119 | 196.164 | 13 | 12 |
| H10 | LGBM Baseline | 0.068652 | 0.063972 | 30.674979 | 7.699673 | 0.010253 | 0.528171 | 1855.310 | 170.858 | 21 | 10 |
| H10 | LGBM BNN-SHAP10 | 0.090109 | 0.086074 | 30.624843 | 7.715469 | 0.007051 | 0.524937 | 461.000 | 180.552 | 58 | 12 |
| H10 | LGBM BNN-Agg | 0.072148 | 0.074408 | 30.650583 | 7.730452 | 0.005381 | 0.529233 | 636.366 | 184.871 | 24 | 12 |
| H25 | LGBM Baseline | 0.062418 | 0.045877 | 43.942055 | 11.779917 | 0.010295 | 0.516507 | 206.215 | 164.176 | 16 | 10 |
| H25 | LGBM BNN-SHAP10 | 0.067594 | 0.061738 | 43.950973 | 11.858488 | 0.001072 | 0.513940 | 111.201 | 189.364 | 13 | 12 |
| H25 | LGBM BNN-Agg | 0.072204 | 0.041381 | 43.944959 | 11.850605 | 0.001346 | 0.512662 | 116.176 | 186.407 | 17 | 12 |

### Table 2: LGBM SHAP-10 Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 90 |
| H3 | 0.024658 | 0.037832 | 17.415184 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 5 | 90 |
| H10 | 0.064984 | 0.084172 | 30.627956 | 7.723615 | 0.006849 | 0.522134 | 693.109 | 183.485 | 27 | 90 |
| H25 | 0.049441 | 0.019258 | 43.966916 | 11.859075 | 0.000347 | 0.515783 | 110.593 | 189.810 | 8 | 90 |

### Table 3: LGBM All Plus SHAP Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H3 | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H10 | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H25 | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |

### Table 4: Baseline LGBM Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.015764 | 0.052687 | 10.311111 | 2.357590 | 0.002879 | 0.548520 | 152.864 | 181.845 | 12 | 86 |
| H3 | 0.027442 | 0.031247 | 17.415736 | 4.168008 | 0.002030 | 0.538295 | 144.802 | 179.555 | 12 | 86 |
| H10 | 0.068652 | 0.063972 | 30.674979 | 7.699673 | 0.010253 | 0.528171 | 1855.310 | 170.858 | 21 | 86 |
| H25 | 0.062418 | 0.045877 | 43.942055 | 11.779917 | 0.010295 | 0.516507 | 206.215 | 164.176 | 16 | 86 |

### Table 5: Walk Forward Trio Models - Average Validation Metrics Across Windows

| Horizon | Model | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|-------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | LGBM | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H1 | XGBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H1 | CatBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H3 | LGBM | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H3 | XGBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H3 | CatBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H10 | LGBM | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H10 | XGBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H10 | CatBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H25 | LGBM | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H25 | XGBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |
| H25 | CatBoost | Available | Available | Available | Available | Available | Available | Available | Available | Available | Available |

### Table 6: Partial Models - Status Summary

| Model | Status | Available Data | Missing Data |
|-------|--------|----------------|--------------|
| BNN Aggregated | Predictions Only | .npz files for all horizons | JSON metrics, CSV submissions |
| BNN SHAP10 | Predictions Only | .npz files for all horizons | JSON metrics, CSV submissions |
| LGBM SHAP 20 | Submissions Only | CSV files, .npz predictions | JSON metrics |
| Trio SHAP Models | In Progress | Currently running | Final results pending |

### Table 7: Best Performing Models by Horizon and Metric

| Horizon | Best Weighted RMSE | Best Pearson | Best RMSE | Best MAE | Best R² | Best Directional Accuracy |
|---------|-------------------|--------------|-----------|----------|---------|---------------------------|
| H1 | LGBM BNN-Agg (0.014189) | LGBM Baseline (0.052687) | LGBM BNN-Agg (10.312380) | LGBM BNN-Agg (2.357590) | LGBM Baseline (0.002879) | LGBM Baseline (0.548520) |
| H3 | LGBM Baseline (0.027442) | LGBM BNN-Agg (0.031708) | LGBM Baseline (17.415736) | LGBM Baseline (4.168008) | LGBM BNN-Agg (0.001001) | LGBM Baseline (0.538295) |
| H10 | LGBM Baseline (0.068652) | LGBM BNN-SHAP10 (0.086074) | LGBM BNN-SHAP10 (30.624843) | LGBM Baseline (7.699673) | LGBM BNN-SHAP10 (0.007051) | LGBM Baseline (0.528171) |
| H25 | LGBM Baseline (0.062418) | LGBM BNN-SHAP10 (0.061738) | LGBM BNN-Agg (43.944959) | LGBM BNN-Agg (11.850605) | LGBM BNN-Agg (0.001346) | LGBM Baseline (0.516507) |

### Table 8: Feature Count Comparison

| Model | H1 Features | H3 Features | H10 Features | H25 Features | Average |
|-------|-------------|-------------|--------------|--------------|---------|
| LGBM Baseline | 86 | 86 | 86 | 86 | 86.0 |
| LGBM BNN-SHAP10 | 12 | 12 | 12 | 12 | 12.0 |
| LGBM BNN-Agg | 12 | 12 | 12 | 12 | 12.0 |
| LGBM SHAP-10 | 90 | 90 | 90 | 90 | 90.0 |

### Table 9: Training Efficiency (Best Iterations)

| Model | H1 | H3 | H10 | H25 | Average |
|-------|----|----|-----|-----|---------|
| LGBM Baseline | 12 | 12 | 21 | 16 | 15.3 |
| LGBM BNN-SHAP10 | 11 | 11 | 58 | 13 | 23.3 |
| LGBM BNN-Agg | 17 | 13 | 24 | 17 | 17.8 |
| LGBM SHAP-10 | 11 | 5 | 27 | 8 | 12.8 |

---

## Summary Statistics

### Total Models with Complete Metrics: 5
### Total Models with Partial Results: 3
### Total Metrics Records: 69 (all_metrics.csv) + 98 (walkforward.csv) = 167
### Total JSON Metric Files: 81
### Total Horizons Evaluated: 4 (H1, H3, H10, H25)
### Total Metrics per Model: 9 (Weighted RMSE, Pearson, RMSE, MAE, R², Directional Accuracy, MAPE, SMAPE, Best Iteration, Features Used)

---

*Tables generated from available metrics files on April 19, 2026*
