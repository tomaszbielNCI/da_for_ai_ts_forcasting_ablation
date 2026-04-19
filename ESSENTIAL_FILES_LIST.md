# Essential Files for NCI Minimal Submission

## Complete src/ Directory Structure Needed

### Core Files (5 files)
1. `src/__init__.py` - Package initialization
2. `src/data_loader.py` - Data loading utilities (13.6 KB)
3. `src/preprocessor.py` - Data preprocessing (15.3 KB)
4. `src/run_shap_10_direct.py` - Direct SHAP execution (15.3 KB)
5. `requirements.txt` - Python dependencies

### Features Directory (3 files)
6. `src/features/__init__.py` - Features package init
7. `src/features/shap_features.py` - SHAP feature engineering (11.6 KB)
8. `src/features/bnn_aggregated_features.py` - BNN feature aggregation (5.5 KB)

### Metrics Directory (2 files)
9. `src/metrics/__init__.py` - Metrics package init
10. `src/metrics/evaluation.py` - Evaluation metrics (13.1 KB)

### Models Directory (15 files)
11. `src/models/__init__.py` - Models package init
12. `src/models/lgbm_with_bnn.py` - **CHAMPION MODEL** (16.1 KB)
13. `src/models/bnn_shap10.py` - BNN SHAP model (17.9 KB)
14. `src/models/bnn_aggregated.py` - BNN aggregated model (9.9 KB)
15. `src/models/lgbm_walk_forward.py` - Walk-forward validation (13.1 KB)
16. `src/models/trio_walk_forward.py` - Trio walk-forward (21.3 KB)
17. `src/models/shap_analyzer.py` - SHAP analysis (16.5 KB)
18. `src/models/ensemble_shap.py` - Ensemble SHAP (9.1 KB)
19. `src/models/lgbm_shap_10.py` - LGBM SHAP10 (15.1 KB)
20. `src/models/lgbm_shap_20.py` - LGBM SHAP20 (14.5 KB)
21. `src/models/lgbm_all_plus_shap.py` - LGBM all SHAP (14.7 KB)
22. `src/models/trio_shap_models.py` - Trio SHAP models (1.7 KB)
23. `src/models/baseline_lgbm.py` - Baseline LGBM (17.5 KB)
24. `src/models/catboost_model.py` - CatBoost model (14.4 KB)
25. `src/models/xgb_model.py` - XGBoost model (13.8 KB)

### Configuration Files (3 files)
26. `config/models.yaml` - Model configurations
27. `config/horizons.yaml` - Horizon settings
28. `config/paths.yaml` - Data paths

### Documentation (2 files)
29. `README.md` - Project overview
30. `utils/kaggle_context.md` - Data context

### Academic Artifacts (2 files)
31. `artifacts/academic_research_paper.md` - IEEE paper
32. `artifacts/presentation_slides.md` - Presentation

### Main Execution (1 file)
33. `run.py` - Main execution script (4.8 KB)

---

## Total: 33 files



## Solution: Priority Selection

### High Priority (Must Have - 15 files)
1. `src/__init__.py`
2. `src/data_loader.py`
3. `src/preprocessor.py`
4. `src/features/__init__.py`
5. `src/features/shap_features.py`
6. `src/features/bnn_aggregated_features.py`
7. `src/metrics/__init__.py`
8. `src/metrics/evaluation.py`
9. `src/models/__init__.py`
10. `src/models/lgbm_with_bnn.py` 
11. `src/models/bnn_shap10.py`
12. `src/models/bnn_aggregated.py`
13. `src/models/lgbm_walk_forward.py`
14. `requirements.txt`
15. `README.md`

### Medium Priority (Add if space - 5 files)
16. `config/models.yaml`
17. `config/horizons.yaml`
18. `src/models/shap_analyzer.py`
19. `src/models/ensemble_shap.py`
20. `run.py`

## Final Strategy
- **Submit 15-20 essential files**
- **Reference GitHub repo for complete code**
- **Add notebooks separately if allowed**
- **Request limit increase tomorrow**
