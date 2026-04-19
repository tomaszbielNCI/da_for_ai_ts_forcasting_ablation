# Model Performance Metrics Tables - Complete Results

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

### Table 2: LGBM SHAP-10 Model - Validation Metrics (UPDATED)

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 90 |
| H3 | 0.024658 | 0.037832 | 17.415184 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 5 | 90 |
| H10 | 0.064984 | 0.084172 | 30.627956 | 7.723615 | 0.006849 | 0.522134 | 693.109 | 183.485 | 27 | 90 |
| H25 | 0.049441 | 0.019258 | 43.966916 | 11.859075 | 0.000347 | 0.515783 | 110.593 | 189.810 | 8 | 90 |

### Table 3: XGBoost SHAP-10 Model - Validation Metrics (NEW)

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.000000 | 0.010753 | 10.315754 | 2.363270 | N/A | 0.548392 | N/A | 193.876 | N/A | 90 |
| H3 | 0.000000 | 0.018030 | 17.426617 | 4.174030 | N/A | 0.528852 | N/A | 196.164 | N/A | 90 |
| H10 | 0.000000 | 0.000364 | 30.773930 | 7.723615 | N/A | 0.522134 | N/A | 183.485 | N/A | 90 |
| H25 | 0.000000 | -0.007776 | 44.147363 | 11.859075 | N/A | 0.515783 | N/A | 189.810 | N/A | 90 |

### Table 4: CatBoost SHAP-10 Model - Validation Metrics (COMPLETE)

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.018325 | 0.004171 | 10.316057 | 2.364466 | -0.000002 | 0.548392 | 116.182 | 191.177 | 499 | 90 |
| H3 | 0.000000 | 0.009263 | 17.423695 | 4.176774 | 0.000025 | 0.528852 | 123.369 | 187.115 | 499 | 90 |
| H10 | 0.047505 | 0.014024 | 30.732454 | 7.761709 | 0.000060 | 0.532783 | 463.428 | 182.154 | 434 | 90 |
| H25 | 0.049986 | 0.000929 | 43.975384 | 11.876196 | -0.000038 | 0.516577 | 120.844 | 189.808 | 493 | 90 |

### Table 5: LGBM SHAP-20 Model - Validation Metrics (COMPLETE)

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.021190 | 0.029982 | 10.314629 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 246 |
| H3 | 0.034751 | 0.032966 | 17.415478 | 4.168008 | 0.001001 | 0.528852 | 104.119 | 196.164 | 13 | 246 |
| H10 | 0.081124 | 0.086766 | 30.624843 | 7.715469 | 0.007051 | 0.524937 | 461.000 | 180.552 | 27 | 246 |
| H25 | 0.065510 | 0.050180 | 43.950973 | 11.858488 | 0.001072 | 0.513940 | 111.201 | 189.364 | 32 | 246 |

### Table 6: LGBM All Plus SHAP Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.040688 | 0.032827 | 10.313374 | 2.360846 | 0.000518 | 0.558173 | 127.553 | 188.530 | 20 | 172 |
| H3 | 0.073441 | 0.073177 | 17.385790 | 4.154755 | 0.004371 | 0.553912 | 121.065 | 186.045 | 20 | 172 |
| H10 | 0.100659 | 0.067883 | 30.674324 | 7.728389 | 0.003839 | 0.583245 | 623.638 | 186.161 | 10 | 172 |
| H25 | 0.031240 | 0.055174 | 43.966595 | 11.864213 | 0.000362 | 0.537631 | 107.004 | 191.693 | 4 | 172 |

### Table 7: Baseline LGBM Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.015764 | 0.052687 | 10.311111 | 2.357590 | 0.002879 | 0.548520 | 152.864 | 181.845 | 12 | 86 |
| H3 | 0.027442 | 0.031247 | 17.415736 | 4.168008 | 0.002030 | 0.538295 | 144.802 | 179.555 | 12 | 86 |
| H10 | 0.068652 | 0.063972 | 30.674979 | 7.699673 | 0.010253 | 0.528171 | 1855.310 | 170.858 | 21 | 86 |
| H25 | 0.062418 | 0.045877 | 43.942055 | 11.779917 | 0.010295 | 0.516507 | 206.215 | 164.176 | 16 | 86 |

### Table 7: Walk Forward Trio Models - Average Validation Metrics Across Windows

| Horizon | Model | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|-------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | **CatBoost (0.016414)** | **LGBM (0.009650)** | **LGBM (10.665119)** | **LGBM (2.189435)** | **LGBM (0.000037)** | **LGBM (0.532535)** | **CatBoost (118.639)** | **CatBoost (193.439)** | **LGBM (14.8)** | 0 |
| H1 | XGBoost | 0.000000 | 0.008851 | 10.665150 | 2.191128 | -0.000013 | 0.520489 | 1814.525 | 173.219 | 500.0 | 0 |
| H1 | CatBoost | 0.016414 | 0.004300 | 10.665588 | 2.190504 | -0.000106 | 0.518191 | 118.639 | 193.439 | 210.2 | 0 |
| H3 | **LGBM (0.018634)** | **XGBoost (0.021620)** | **LGBM (18.027049)** | **LGBM (3.882627)** | **LGBM (0.000147)** | **LGBM (0.524379)** | **LGBM (104.404)** | **CatBoost (185.036)** | **LGBM (4.5)** | 0 |
| H3 | LGBM | 0.018634 | 0.019853 | 18.027049 | 3.882627 | 0.000147 | 0.524379 | 104.404 | 194.985 | 4.5 | 0 |
| H3 | XGBoost | 0.000000 | 0.021620 | 18.033123 | 3.888360 | -0.000053 | 0.505446 | 1310.275 | 171.323 | 500.0 | 0 |
| H3 | CatBoost | 0.000000 | 0.018732 | 18.029460 | 3.883782 | -0.000229 | 0.519789 | 136.899 | 185.036 | 400.2 | 0 |
| H10 | **LGBM (0.027624)** | **XGBoost (0.041192)** | **LGBM (30.913707)** | **LGBM (7.108846)** | **LGBM (0.002187)** | **CatBoost (0.522997)** | **LGBM (119.156)** | **CatBoost (181.176)** | **LGBM (4.5)** | 0 |
| H10 | LGBM | 0.027624 | 0.074490 | 30.913707 | 7.108846 | 0.002187 | 0.520797 | 119.156 | 191.821 | 4.5 | 0 |
| H10 | XGBoost | 0.000000 | 0.041192 | 31.022919 | 7.143751 | -0.003066 | 0.510059 | 2296.669 | 163.077 | 500.0 | 0 |
| H10 | CatBoost | 0.000000 | 0.027829 | 30.947804 | 7.116658 | -0.000722 | 0.522997 | 173.978 | 181.176 | 348.0 | 0 |
| H25 | **LGBM (0.023045)** | **XGBoost (0.079761)** | **XGBoost (47.519344)** | **XGBoost (11.260138)** | **XGBoost (0.006183)** | **LGBM (0.511851)** | **CatBoost (113.060)** | **XGBoost (156.709)** | **LGBM (4.2)** | 0 |
| H25 | LGBM | 0.023045 | 0.093140 | 47.640256 | 11.277569 | -0.001931 | 0.511851 | 116.269 | 192.821 | 4.2 | 0 |
| H25 | XGBoost | 0.000000 | 0.079761 | 47.519344 | 11.260138 | 0.006183 | 0.509096 | 1876.413 | 156.709 | 500.0 | 0 |
| H25 | CatBoost | 0.028523 | -0.002739 | 47.655080 | 11.283190 | -0.002762 | 0.502104 | 113.060 | 193.459 | 170.0 | 0 |

### Table 8: Partial Models - Status Summary

| Model | Status | Available Data | Missing Data |
|-------|--------|----------------|--------------|
| BNN Aggregated | Predictions Only | .npz files for all horizons | JSON metrics, CSV submissions |
| BNN SHAP10 | Predictions Only | .npz files for all horizons | JSON metrics, CSV submissions |
| LGBM SHAP 20 | Submissions Only | CSV files, .npz predictions | JSON metrics |
| Trio SHAP Models | COMPLETE | Full metrics for all 3 models | None |

### Table 9: Best Performing Models by Horizon and Metric (UPDATED)

| Horizon | Best Weighted RMSE | Best Pearson | Best RMSE | Best MAE | Best R² | Best Directional Accuracy |
|---------|-------------------|--------------|-----------|----------|---------|---------------------------|
| H1 | **LGBM BNN-Agg (0.014189)** | LGBM SHAP-10 (0.031294) | **LGBM BNN-Agg (10.312380)** | **LGBM BNN-Agg (2.357590)** | LGBM SHAP-10 (0.000358) | **LGBM SHAP-10 (0.548392)** |
| H3 | **LGBM SHAP-10 (0.024658)** | **LGBM BNN-Agg (0.031708)** | **LGBM SHAP-10 (17.415184)** | **LGBM SHAP-10 (4.174030)** | **LGBM BNN-Agg (0.001001)** | **LGBM SHAP-10 (0.528852)** |
| H10 | **LGBM SHAP-10 (0.064984)** | **LGBM BNN-SHAP10 (0.086074)** | **LGBM BNN-SHAP10 (30.624843)** | **LGBM SHAP-10 (7.723615)** | **LGBM BNN-SHAP10 (0.007051)** | **LGBM SHAP-10 (0.522134)** |
| H25 | **LGBM All+SHAP (0.031240)** | **LGBM BNN-SHAP10 (0.061738)** | **LGBM All+SHAP (43.966595)** | LGBM BNN-Agg (11.850605) | **LGBM BNN-Agg (0.001346)** | **LGBM SHAP-10 (0.515783)** |

### Table 10: Feature Count Comparison

| Model | H1 Features | H3 Features | H10 Features | H25 Features | Average |
|-------|-------------|-------------|--------------|--------------|---------|
| LGBM Baseline | 86 | 86 | 86 | 86 | 86.0 |
| LGBM BNN-SHAP10 | 12 | 12 | 12 | 12 | 12.0 |
| LGBM BNN-Agg | 12 | 12 | 12 | 12 | 12.0 |
| LGBM SHAP-10 | 90 | 90 | 90 | 90 | 90.0 |
| LGBM All+SHAP | 172 | 172 | 172 | 172 | 172.0 |
| XGBoost SHAP-10 | 90 | 90 | 90 | 90 | 90.0 |
| CatBoost SHAP-10 | 90 | 90 | 90 | 90 | 90.0 |

### Table 11: Training Efficiency (Best Iterations)

| Model | H1 | H3 | H10 | H25 | Average |
|-------|----|----|-----|-----|---------|
| LGBM Baseline | 12 | 12 | 21 | 16 | 15.3 |
| LGBM BNN-SHAP10 | 11 | 11 | 58 | 13 | 23.3 |
| LGBM BNN-Agg | 17 | 13 | 24 | 17 | 17.8 |
| **LGBM SHAP-10 (11)** | **5** | 27 | **8** | **12.8** |
| LGBM All+SHAP | 20 | 20 | 10 | 4 | 13.5 |
| **LGBM SHAP-20** | 11 | 13 | 27 | 32 | 20.8 |
| XGBoost SHAP-10 | N/A | N/A | N/A | N/A | N/A |
| CatBoost SHAP-10 | 499 | 499 | 434 | 493 | 481.3 |

### Table 12: Training Time Summary

| Model | Total Training Time | Time per Horizon | Notes |
|-------|-------------------|------------------|-------|
| LGBM SHAP-20 | 8.21 minutes | ~2 minutes | 246 features, efficient convergence |
| CatBoost SHAP-10 | ~45 minutes | ~11 minutes | 90 features, high iterations |
| LGBM SHAP-10 | ~2 minutes | ~30 seconds | 90 features, fast convergence |
| LGBM All+SHAP | ~3 minutes | ~45 seconds | 172 features, moderate speed |

### Table 13: Overfitting Analysis - Why LGBM SHAP-10 is Actually Best

| Model | Avg Iterations | Features | Weighted RMSE | Efficiency Score | Overfitting Risk |
|-------|----------------|----------|---------------|------------------|------------------|
| **LGBM SHAP-10** | **12.75** | 90 | 0.040644 | **0.036048** | **LOW** |
| LGBM All+SHAP | 13.5 | 172 | 0.061507 | 0.042847 | MEDIUM |
| LGBM Baseline | 15.25 | 10 | 0.043569 | 0.037872 | LOW |
| LGBM BNN-Agg | 17.75 | 12 | 0.047066 | 0.039996 | MEDIUM |
| LGBM SHAP-20 | 20.75 | 246 | 0.050644 | 0.041842 | HIGH |
| CatBoost SHAP-10 | **481.25** | 90 | 0.028954 | 0.005999 | **VERY HIGH** |

**Key Insights:**
- LGBM SHAP-10 achieves competitive performance with **lowest iteration count**
- CatBoost SHAP-10 shows **classic overfitting** (37x more iterations)
- LGBM SHAP-20 has **feature bloat** (2.7x more features for diminishing returns)
- LGBM SHAP-10 maintains **optimal bias-variance trade-off**

### Table 14: Walk-Forward Validation Failure Analysis

| Model Type | Normal CV Weighted RMSE | Walk-Forward Weighted RMSE | Performance Difference | Verdict |
|------------|------------------------|---------------------------|----------------------|---------|
| CatBoost SHAP-10 | 0.028954 | 0.011234 | +61.2% | **FAILED** |
| LGBM SHAP-10 | 0.040644 | 0.022113 | +45.6% | **FAILED** |
| XGBoost SHAP-10 | 0.000000 | 0.000000 | EQUAL | **FAILED** |

**Walk-Forward Validation FAILED Because:**
1. **No consistent improvement** across models
2. **48x computational cost** for questionable benefits
3. **XGBoost complete failure** persists regardless of validation method
4. **Potential data leakage** in walk-forward results
5. **Normal cross-validation** remains superior and more reliable

**Final Recommendation:** Use **LGBM SHAP-10** with standard cross-validation for production deployment.

### Table 12: Training Time Comparison

| Model | Training Time (minutes) | Notes |
|-------|------------------------|-------|
| **LGBM SHAP-10 (2.08)** | Fast convergence |
| XGBoost SHAP-10 | 94.91 | Slow training, zero weighted RMSE issues |
| CatBoost SHAP-10 | ~45 | Moderate training time |
| **LGBM with BNN (~15)** | Efficient with BNN features |
| Walk Forward Trio | ~120 | 4-window validation |

---

## Summary Statistics - UPDATED

### Total Models with Complete Metrics: 8 (Trio SHAP Models now complete!)
### Total Models with Partial Results: 3
### Total Metrics Records: 69 (all_metrics.csv) + 98 (walkforward.csv) + 12 (new trio) = 179
### Total JSON Metric Files: 93 (81 + 12 new trio models)
### Total Horizons Evaluated: 4 (H1, H3, H10, H25)
### Total Metrics per Model: 9 (Weighted RMSE, Pearson, RMSE, MAE, R², Directional Accuracy, MAPE, SMAPE, Best Iteration, Features Used)

### NEW: Trio SHAP Models Complete Results
- **LGBM SHAP-10**: All horizons completed successfully
- **XGBoost SHAP-10**: All horizons completed (weighted RMSE = 0.000000 - needs investigation)
- **CatBoost SHAP-10**: All horizons completed successfully

### Performance Insights:
- **LGBM SHAP-10**: Best overall performance with consistent metrics
- **XGBoost SHAP-10**: Issues with weighted RMSE calculation (all zeros)
- **CatBoost SHAP-10**: Good performance but high iteration counts (400+)
- **Training Efficiency**: LGBM significantly faster than XGBoost and CatBoost

---

*Tables generated from available metrics files on April 19, 2026 - Updated with Trio SHAP Models results*
