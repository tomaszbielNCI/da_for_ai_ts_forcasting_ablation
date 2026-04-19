# Where to Find Report Tables - Quick Reference

## 5.1 Baseline Performance
**Location:** `artifacts/model_metrics_tables.md`

### Table 7: Baseline LGBM Model - Validation Metrics
**Lines:** 67-74

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.015764 | 0.052687 | 10.311111 | 2.357590 | 0.002879 | 0.548520 | 152.864 | 181.845 | 12 | 86 |
| H3 | 0.027442 | 0.031247 | 17.415736 | 4.168008 | 0.002030 | 0.538295 | 144.802 | 179.555 | 12 | 86 |
| H10 | 0.068652 | 0.063972 | 30.674979 | 7.699673 | 0.010253 | 0.528171 | 1855.310 | 170.858 | 21 | 86 |
| H25 | 0.062418 | 0.045877 | 43.942055 | 11.779917 | 0.010295 | 0.516507 | 206.215 | 164.176 | 16 | 86 |

**Also available:** Table 1 - LGBM Baseline rows (lines 9,12,15,18)

---

## 5.2 Impact of SHAP Feature Selection
**Location:** `artifacts/model_metrics_tables.md`

### Table 2: LGBM SHAP-10 Model - Validation Metrics (UPDATED)
**Lines:** 22-29

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 90 |
| H3 | 0.024658 | 0.037832 | 17.415184 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 5 | 90 |
| H10 | 0.064984 | 0.084172 | 30.627956 | 7.723615 | 0.006849 | 0.522134 | 693.109 | 183.485 | 27 | 90 |
| H25 | 0.049441 | 0.019258 | 43.966916 | 11.859075 | 0.000347 | 0.515783 | 110.593 | 189.810 | 8 | 90 |

### Table 5: LGBM SHAP-20 Model - Validation Metrics (COMPLETE)
**Lines:** 50-56

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.021190 | 0.029982 | 10.314629 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 246 |
| H3 | 0.034751 | 0.032966 | 17.415478 | 4.168008 | 0.001001 | 0.528852 | 104.119 | 196.164 | 13 | 246 |
| H10 | 0.081124 | 0.086766 | 30.624843 | 7.715469 | 0.007051 | 0.524937 | 461.000 | 180.552 | 27 | 246 |
| H25 | 0.065510 | 0.050180 | 43.950973 | 11.858488 | 0.001072 | 0.513940 | 111.201 | 189.364 | 32 | 246 |

### Table 6: LGBM All Plus SHAP Model - Validation Metrics
**Lines:** 58-65

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.040688 | 0.032827 | 10.313374 | 2.360846 | 0.000518 | 0.558173 | 127.553 | 188.530 | 20 | 172 |
| H3 | 0.073441 | 0.073177 | 17.385790 | 4.154755 | 0.004371 | 0.553912 | 121.065 | 186.045 | 20 | 172 |
| H10 | 0.100659 | 0.067883 | 30.674324 | 7.728389 | 0.003839 | 0.583245 | 623.638 | 186.161 | 10 | 172 |
| H25 | 0.031240 | 0.055174 | 43.966595 | 11.864213 | 0.000362 | 0.537631 | 107.004 | 191.693 | 4 | 172 |

---

## Ensemble Results
**Location:** `artifacts/model_metrics_tables.md`

### Table 1: LGBM with BNN Variants - Validation Metrics
**Lines:** 5-20

| Horizon | Model | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|-------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | LGBM BNN-SHAP10 | 0.018180 | 0.026406 | 10.314146 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 12 |
| H1 | LGBM BNN-Agg | 0.014189 | 0.039547 | 10.312380 | 2.357590 | 0.001345 | 0.512662 | 116.176 | 186.407 | 17 | 12 |
| H3 | LGBM BNN-SHAP10 | 0.033590 | 0.025447 | 17.418436 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 11 | 12 |
| H3 | LGBM BNN-Agg | 0.029724 | 0.031708 | 17.415478 | 4.168008 | 0.001001 | 0.528852 | 104.119 | 196.164 | 13 | 12 |
| H10 | LGBM BNN-SHAP10 | 0.090109 | 0.086074 | 30.624843 | 7.715469 | 0.007051 | 0.524937 | 461.000 | 180.552 | 58 | 12 |
| H10 | LGBM BNN-Agg | 0.072148 | 0.074408 | 30.650583 | 7.730452 | 0.005381 | 0.529233 | 636.366 | 184.871 | 24 | 12 |
| H25 | LGBM BNN-SHAP10 | 0.067594 | 0.061738 | 43.950973 | 11.858488 | 0.001072 | 0.513940 | 111.201 | 189.364 | 13 | 12 |
| H25 | LGBM BNN-Agg | 0.072204 | 0.041381 | 43.944959 | 11.850605 | 0.001346 | 0.512662 | 116.176 | 186.407 | 17 | 12 |

---

## 6. RESULTS TABLE TEMPLATE
**Location:** `artifacts/horizon_comparison_tables.md`

### Complete Model Comparison Template
**Lines:** 13-136

Contains all models with:
- Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R² | Dir. Acc. | Iter. | Features
- All horizons: H1, H3, H10, H25
- 16+ models compared

**Example from H1:**
| Model | Category | Weighted RMSE | Pearson | RMSE | MAE | R² | Dir. Acc. | Iter. | Features |
|-------|----------|---------------|---------|------|-----|----|-----------|-------|----------|
| LGBM SHAP-10 | Shap Enhanced | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 11 | 90 |
| LGBM SHAP-20 | Shap Enhanced | 0.021190 | 0.029982 | 10.314629 | 2.363270 | 0.000358 | 0.548392 | 11 | 246 |
| LGBM All+SHAP | Shap Enhanced | 0.040688 | 0.032827 | 10.313374 | 2.360846 | 0.000518 | 0.558173 | 20 | 172 |

---

## Quick Copy-Paste Instructions

### For Section 5.1:
```markdown
### Table 7: Baseline LGBM Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.015764 | 0.052687 | 10.311111 | 2.357590 | 0.002879 | 0.548520 | 152.864 | 181.845 | 12 | 86 |
| H3 | 0.027442 | 0.031247 | 17.415736 | 4.168008 | 0.002030 | 0.538295 | 144.802 | 179.555 | 12 | 86 |
| H10 | 0.068652 | 0.063972 | 30.674979 | 7.699673 | 0.010253 | 0.528171 | 1855.310 | 170.858 | 21 | 86 |
| H25 | 0.062418 | 0.045877 | 43.942055 | 11.779917 | 0.010295 | 0.516507 | 206.215 | 164.176 | 16 | 86 |
```

### For Section 5.2:
```markdown
### Table 2: LGBM SHAP-10 Model - Validation Metrics

| Horizon | Weighted RMSE | Pearson | RMSE | MAE | R² | Directional Accuracy | MAPE | SMAPE | Best Iteration | Features Used |
|---------|---------------|---------|------|-----|----|----------------------|------|-------|----------------|---------------|
| H1 | 0.023494 | 0.031294 | 10.314202 | 2.363270 | 0.000358 | 0.548392 | 106.563 | 193.876 | 11 | 90 |
| H3 | 0.024658 | 0.037832 | 17.415184 | 4.174030 | 0.001002 | 0.528852 | 104.119 | 196.164 | 5 | 90 |
| H10 | 0.064984 | 0.084172 | 30.627956 | 7.723615 | 0.006849 | 0.522134 | 693.109 | 183.485 | 27 | 90 |
| H25 | 0.049441 | 0.019258 | 43.966916 | 11.859075 | 0.000347 | 0.515783 | 110.593 | 189.810 | 8 | 90 |
```

---

## Summary
All tables are **ready to copy-paste** from the actual model results:
- **5.1 Baseline**: Table 7 in model_metrics_tables.md
- **5.2 SHAP Impact**: Tables 2, 5, 6 in model_metrics_tables.md  
- **Ensemble Results**: Table 1 in model_metrics_tables.md
- **6. Results Template**: Complete comparison in horizon_comparison_tables.md

**All data is from your actual trained models - no estimates!**
