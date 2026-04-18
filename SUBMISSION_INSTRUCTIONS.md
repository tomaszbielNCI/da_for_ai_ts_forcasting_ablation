# NCI PGDAI Submission Instructions

## Submission Package Ready! 

**File**: `nci_submission.zip` (427 KB - well under 500MB limit)

## What's Included in Submission

### Core Academic Components
- **Complete source code** (`src/`) - All model implementations
- **Academic artifacts** (`artifacts/`) - Research paper, presentations, visualizations
- **Interactive notebooks** (`notebooks/`) - Data analysis and exploration
- **Configuration files** (`config/`) - Model parameters and settings
- **Documentation** (`README.md`) - Comprehensive project overview

### Key Files for Evaluation
- `README.md` - Project overview and academic contributions
- `artifacts/academic_research_paper.md` - IEEE-style research paper
- `artifacts/presentation_slides.md` - 10-minute presentation structure
- `notebooks/unified_sanity_check_analysis_final.ipynb` - Complete data analysis
- `src/models/lgbm_with_bnn.py` - Champion model implementation
- `requirements.txt` - Python dependencies

## What's Excluded (Large Data Files)
- Original datasets (12GB+)
- Model predictions and intermediate results
- Cached files and temporary data
- Downloaded data files

## Submission Steps

### 1. Download the Package
```bash
# From GitHub repository
git clone https://github.com/tomaszbielNCI/da_for_ai_ts_forcasting_ablation.git
cd da_for_ai_ts_forcasting_ablation

# Or download the prepared zip file
# nci_submission.zip (427 KB)
```

### 2. Verify Package Contents
```bash
# Check key files exist
ls README.md
ls artifacts/academic_research_paper.md
ls notebooks/unified_sanity_check_analysis_final.ipynb
ls src/models/lgbm_with_bnn.py
```

### 3. Installation (if needed)
```bash
pip install -r requirements.txt
```

### 4. Run Key Demonstrations
```bash
# Run main model
python run.py

# Run data analysis
jupyter notebook notebooks/unified_sanity_check_analysis_final.ipynb
```

## Academic Evaluation Checklist

### Learning Outcomes Coverage
- **LO1**: Critical understanding of ML concepts - Covered in research paper
- **LO2**: Data analytics tools - Demonstrated in notebooks and code
- **LO3**: ML application for decision-making - Shown in model implementations
- **LO4**: Graphical tools for analytics - Available in artifacts/notebooks
- **LO5**: Critical analysis and presentation - Complete in research paper

### IEEE Paper Requirements
- **8-10 pages**: `artifacts/academic_research_paper.md`
- **150-250 word abstract**: Included in paper
- **12-15 Scopus references**: Framework provided
- **CRISP-DM methodology**: Applied throughout project

### Presentation Requirements
- **10 minutes, 7 slides**: `artifacts/presentation_slides.md`
- **Team coordination**: Structure provided for group presentation
- **Methodology overview**: Complete in slides

## Technical Specifications

### Package Size: 427 KB (well under 500MB limit)
### Files: 38 source files + documentation
### Languages: Python, Markdown
### Dependencies: Listed in requirements.txt

## Key Academic Contributions

1. **Weighted Ensemble Methodology** - Novel ensemble optimization
2. **SHAP Feature Augmentation** - Explainable AI integration
3. **Multi-Algorithm Ensemble** - Algorithmic diversity validation
4. **BNN Integration** - Neural-classical hybrid approaches
5. **Multi-Horizon Analysis** - Systematic temporal evaluation

## Data Context Note
The project uses financial time series data for methodology demonstration. All large datasets are excluded from submission - only code, documentation, and analysis tools are included.

## Contact Information
**Course**: Data Analytics for Artificial Intelligence (PGDAI_SEP25)
**Institution**: National College of Ireland
**Weight**: 70% of final grade
**Submission Date**: April 17, 2026

---

**This submission package contains all necessary academic work for evaluation while staying well within the 500MB size limit.**
