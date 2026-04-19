# Horizon-by-Horizon Model Comparison

Generated: 2026-04-19 20:58:42

## Model Categories

- **Baseline**: Raw features without SHAP enhancement
- **Shap Enhanced**: SHAP feature selection and engineering
- **Bnn Enhanced**: BNN predictions as additional features
- **Algorithm Variants**: XGBoost and CatBoost implementations
- **Walk Forward**: Time-based cross-validation approach

### Horizon 1 - Model Comparison

| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R� | Dir. Acc. | Iter. | Features |
|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|
| XGBoost SHAP-10 | Algorithm Variants | 0.000000 | 0.010753 | 10.315754 | 2.363430 | 0.000057 | 0.533307 | N/A | 90 |
| CatBoost SHAP-10 | Algorithm Variants | 0.018325 | 0.004171 | 10.316057 | 2.364466 | -0.000002 | 0.538314 | 499 | 90 |
| Baseline LGBM (Raw Features) | Baseline | 0.015764 | 0.052687 | 10.311111 | 2.362113 | 0.000957 | 0.544309 | 12 | 10 |
| LGBM BNN-Aggregated | Bnn Enhanced | 0.014189 | 0.039547 | 10.312380 | 2.362067 | 0.000711 | 0.546203 | 17 | 12 |
| LGBM BNN-SHAP10 | Bnn Enhanced | 0.018180 | 0.026406 | 10.314146 | 2.362652 | 0.000369 | 0.546439 | 11 | 12 |
| baseline_lgbm | Other | 0.000000 | **0.063824** | **10.301184** | 2.357590 | **0.002879** | 0.548520 | N/A | 86 |
| baseline_lgbm_raw | Other | 0.029270 | 0.051032 | 10.312491 | 2.362783 | 0.000689 | 0.573493 | 8 | 86 |
| trio_cat | Other | 0.056844 | -0.000557 | 11.664266 | 2.123347 | -0.000114 | 0.532194 | 210 | N/A |
| LGBM SHAP-10 | Shap Enhanced | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 11 | 90 |
| LGBM All+SHAP (Raw + SHAP-10 Engineered) | Shap Enhanced | 0.040688 | 0.032827 | 10.313374 | 2.360846 | 0.000518 | 0.558173 | 20 | 172 |
| Trio LGBM (Walk-Forward) | Walk Forward | 0.137071 | 0.054292 | 11.656985 | 2.119204 | 0.001063 | 0.567445 | 14 | N/A |
| Trio XGBoost (Walk-Forward) | Walk Forward | **0.487503** | 0.042410 | 11.660930 | **2.109075** | 0.000487 | **0.657662** | 500 | N/A |

**Key Insights for H1:**
- Best Weighted RMSE: **Trio XGBoost (Walk-Forward)** (0.487503)
- Best Pearson: **baseline_lgbm** (0.063824)
- Total models compared: 12


### Horizon 3 - Model Comparison

| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R� | Dir. Acc. | Iter. | Features |
|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|
| CatBoost SHAP-10 | Algorithm Variants | 0.000000 | 0.009263 | 17.423695 | 4.176774 | 0.000025 | 0.522025 | 499 | 90 |
| XGBoost SHAP-10 | Algorithm Variants | 0.000000 | 0.018030 | 17.426617 | 4.173207 | -0.000310 | 0.508775 | N/A | 90 |
| Baseline LGBM (Raw Features) | Baseline | 0.027442 | 0.031247 | 17.415736 | 4.170765 | 0.000938 | 0.517495 | 12 | 10 |
| LGBM BNN-Aggregated | Bnn Enhanced | 0.029724 | 0.031708 | 17.415478 | 4.170230 | 0.000968 | 0.525851 | 13 | 12 |
| LGBM BNN-SHAP10 | Bnn Enhanced | 0.033590 | 0.025447 | 17.418436 | 4.171365 | 0.000629 | 0.520440 | 11 | 12 |
| baseline_lgbm | Other | 0.000000 | 0.046902 | 17.406217 | 4.168008 | 0.002030 | 0.538295 | N/A | 86 |
| baseline_lgbm_raw | Other | 0.032981 | 0.045781 | 17.406495 | 4.168944 | 0.001998 | 0.577228 | 29 | 86 |
| trio_cat | Other | **0.137298** | 0.021803 | 19.324626 | 3.751557 | -0.000374 | 0.545719 | 400 | N/A |
| LGBM SHAP-10 | Shap Enhanced | 0.024658 | 0.037832 | 17.415184 | 4.174031 | 0.001002 | 0.528852 | 5 | 90 |
| LGBM All+SHAP (Raw + SHAP-10 Engineered) | Shap Enhanced | 0.073441 | 0.073177 | **17.385790** | 4.154755 | 0.004371 | 0.553912 | 20 | 172 |
| Trio XGBoost (Walk-Forward) | Walk Forward | 0.000000 | **0.157967** | 19.212176 | **3.677900** | **0.011315** | **0.594736** | 500 | N/A |
| Trio LGBM (Walk-Forward) | Walk Forward | 0.063219 | 0.096074 | 19.303580 | 3.747969 | 0.001543 | 0.527073 | 4 | N/A |

**Key Insights for H3:**
- Best Weighted RMSE: **trio_cat** (0.137298)
- Best Pearson: **Trio XGBoost (Walk-Forward)** (0.157967)
- Total models compared: 12


### Horizon 10 - Model Comparison

| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R� | Dir. Acc. | Iter. | Features |
|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|
| XGBoost SHAP-10 | Algorithm Variants | 0.000000 | 0.000364 | 30.773930 | 7.763397 | -0.002641 | 0.517891 | N/A | 90 |
| CatBoost SHAP-10 | Algorithm Variants | 0.047505 | 0.014024 | 30.732454 | 7.761709 | 0.000060 | 0.532783 | 434 | 90 |
| Baseline LGBM (Raw Features) | Baseline | 0.068652 | 0.063972 | 30.674979 | 7.733197 | 0.003797 | 0.538082 | 21 | 10 |
| LGBM BNN-Aggregated | Bnn Enhanced | 0.072148 | 0.074408 | 30.650583 | 7.730452 | 0.005381 | 0.529233 | 24 | 12 |
| LGBM BNN-SHAP10 | Bnn Enhanced | 0.090109 | 0.086074 | 30.624843 | 7.715469 | 0.007050 | 0.524937 | 58 | 12 |
| baseline_lgbm | Other | 0.000000 | 0.103502 | **30.575416** | 7.699673 | 0.010253 | 0.528171 | N/A | 86 |
| baseline_lgbm_raw | Other | 0.089533 | 0.066822 | 30.669748 | 7.724256 | 0.004136 | 0.555099 | 26 | 86 |
| trio_cat | Other | 0.135379 | 0.024905 | 33.739360 | 6.798781 | -0.001035 | 0.538874 | 348 | N/A |
| LGBM SHAP-10 | Shap Enhanced | 0.064984 | 0.084172 | 30.627956 | 7.723615 | 0.006849 | 0.522134 | 27 | 90 |
| LGBM All+SHAP (Raw + SHAP-10 Engineered) | Shap Enhanced | 0.100659 | 0.067883 | 30.674324 | 7.728389 | 0.003839 | 0.583245 | 10 | 172 |
| Trio LGBM (Walk-Forward) | Walk Forward | 0.078993 | **0.165948** | 33.585115 | 6.775901 | 0.007223 | 0.528144 | 4 | N/A |
| Trio XGBoost (Walk-Forward) | Walk Forward | **0.252331** | 0.158979 | 33.404350 | **6.672912** | **0.019512** | **0.594034** | 500 | N/A |

**Key Insights for H10:**
- Best Weighted RMSE: **Trio XGBoost (Walk-Forward)** (0.252331)
- Best Pearson: **Trio LGBM (Walk-Forward)** (0.165948)
- Total models compared: 12


### Horizon 25 - Model Comparison

| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R� | Dir. Acc. | Iter. | Features |
|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|
| XGBoost SHAP-10 | Algorithm Variants | 0.000000 | -0.007776 | 44.147363 | 11.904637 | -0.007875 | 0.511075 | N/A | 90 |
| CatBoost SHAP-10 | Algorithm Variants | 0.049986 | 0.000929 | 43.975384 | 11.876196 | -0.000038 | 0.516577 | 493 | 90 |
| Baseline LGBM (Raw Features) | Baseline | 0.062418 | 0.045877 | 43.942055 | 11.849889 | 0.001478 | 0.516700 | 16 | 10 |
| LGBM BNN-SHAP10 | Bnn Enhanced | 0.067594 | 0.061738 | 43.950973 | 11.858488 | 0.001072 | 0.513940 | 13 | 12 |
| LGBM BNN-Aggregated | Bnn Enhanced | 0.072204 | 0.041381 | 43.944959 | 11.850605 | 0.001346 | 0.512662 | 17 | 12 |
| baseline_lgbm | Other | 0.000000 | 0.110520 | 43.747617 | 11.779917 | 0.010295 | 0.516508 | N/A | 86 |
| trio_cat | Other | 0.097060 | -0.012165 | 52.332536 | 10.795355 | -0.002258 | 0.517660 | 170 | N/A |
| baseline_lgbm_raw | Other | 0.105438 | 0.125551 | **43.698443** | 11.769045 | 0.012518 | 0.523356 | 28 | 86 |
| LGBM All+SHAP (Raw + SHAP-10 Engineered) | Shap Enhanced | 0.031240 | 0.055174 | 43.966595 | 11.864213 | 0.000362 | 0.537631 | 4 | 172 |
| LGBM SHAP-10 | Shap Enhanced | 0.049441 | 0.019258 | 43.966916 | 11.859075 | 0.000347 | 0.515783 | 8 | 90 |
| Trio LGBM (Walk-Forward) | Walk Forward | 0.127271 | 0.136667 | 52.307020 | 10.777206 | -0.001378 | 0.529910 | 4 | N/A |
| Trio XGBoost (Walk-Forward) | Walk Forward | **0.770687** | **0.238360** | 51.656852 | **10.238879** | **0.024020** | **0.696276** | 500 | N/A |

**Key Insights for H25:**
- Best Weighted RMSE: **Trio XGBoost (Walk-Forward)** (0.770687)
- Best Pearson: **Trio XGBoost (Walk-Forward)** (0.238360)
- Total models compared: 12


## Overall Summary

### Best Models by Horizon

| Horizon | Best Model | Weighted RMSE |
|---------|------------|---------------|
| H1 | Trio XGBoost (Walk-Forward) | **0.487503** |
| H3 | trio_cat | **0.137298** |
| H10 | Trio XGBoost (Walk-Forward) | **0.252331** |
| H25 | Trio XGBoost (Walk-Forward) | **0.770687** |

### Overall Best Performance

**Best Model Overall**: Trio XGBoost (Walk-Forward)
**Best Weighted RMSE**: 0.770687
