# Engagement-Based Influencer Marketing Effectiveness: A Multi-Tier Analysis

**MSc Computer Science Dissertation**  
**Trinity College Dublin | 2025-2026**


## Overview

This repository contains the complete code, data processing pipelines, and analysis scripts for my MSc dissertation investigating influencer marketing effectiveness across different scales using machine learning.

**Key Findings:**
- Mid-tier YouTube influencers (100K–1M) achieved **3.29× higher engagement** than mega-tier Instagram influencers (50M–500M)
- Machine learning achieved **R²=0.39** for mid-tier prediction but **failed (R²=-0.26)** for mega-tier
- First systematic examination of **scale-dependent predictability** in influencer engagement

## Research Questions

**RQ1:** What is the magnitude of engagement rate differences between mid-tier and mega-tier influencers?

**RQ2:** Which quantitative metrics best predict engagement at different scales?

**RQ3:** Does ML prediction accuracy vary systematically by influencer scale?

**RQ4:** What are the practical implications for influencer selection strategies?


```

##  Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- (Optional) YouTube Data API v3 key for data collection

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dissertation-influencer-ml.git
cd dissertation-influencer-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Step 1: Data collection (requires API key)
python src/data/youtube_collector.py

# Step 2: Feature engineering
python src/data/feature_engineer.py

# Step 3: Run baseline models
python src/models/baseline.py

# Step 4: Train ML models
python src/models/ml_models.py

# Or run all notebooks in order (recommended for exploration)
jupyter notebook
```

##  Dependencies

### Core Libraries
- **Data Processing:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`, `shap`
- **Statistical Analysis:** `scipy`, `statsmodels`
- **Data Collection:** `google-api-python-client`

See `requirements.txt` for complete list with versions.

##  Key Results

### Engagement Rate Comparison

| Platform | Tier | n | Mean Engagement | Std Dev |
|----------|------|---|-----------------|---------|
| YouTube | Mid (100K–1M) | 45 | 3.65% | 1.70% |
| Instagram | Mega (50M–500M) | 40 | 1.11% | 0.60% |
| **Ratio** | - | - | **3.29×** | - |

**Statistical Test:** t(83)=6.75, p<0.001, Cohen's d=1.47

### Machine Learning Performance

| Model | YouTube R² | Instagram R² | Gap |
|-------|-----------|--------------|-----|
| Linear Regression | 0.08 | -0.15 | 0.23 |
| Random Forest (Conservative) | **0.39** | -0.26 | **0.65** |
| XGBoost | 0.32 | -0.79 | 1.11 |

### Top Predictive Features (YouTube)

1. Comment-to-view ratio (r=0.382)
2. Video count (r=0.347)
3. Recent activity rate (r=0.297)
4. Total views (r=0.274)
5. Subscribers per video (r=0.268)

##  Methodology

### Data Collection
- **YouTube:** YouTube Data API v3 (December 2024 – January 2025)
- **Instagram:** Kaggle "Top 200 Instagram Influencers" dataset
- **Sample:** n=85 (45 YouTube, 40 Instagram)

### Feature Engineering
- Initial features: 21 (YouTube), 17 (Instagram)
- Final features: 8 (YouTube), 9 (Instagram)
- Systematic leakage detection and removal
- Samples-per-feature ratios: 5.6 (YouTube), 4.4 (Instagram)

### Models Evaluated
- **Baselines:** Mean predictor, single-feature regression, full linear regression
- **ML Models:** Linear Regression, Ridge Regression (α=1.0, 5.0, 10.0), Random Forest (2 configs), XGBoost
- **Validation:** 5-fold cross-validation

### Evaluation Metrics
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Train-test performance gap

##  Academic Context

**Institution:** Trinity College Dublin  
**School:** Computer Science and Statistics  
**Degree:** MSc in Computer Science  
**Course Code:** CS7CS6 - Dissertation  
**Supervisor:** Dr. Van-Dinh Nguyen  
**Submission:** August 2026

##  Citation

If you use this code or reference this work, please cite:

```bibtex
@mastersthesis{varma2026influencer,
  author = {Varma, Shaurya},
  title = {Engagement-Based Influencer Marketing Effectiveness: A Multi-Tier Analysis of YouTube and Instagram Using Machine Learning},
  school = {Trinity College Dublin},
  year = {2026},
  type = {MSc Dissertation},
  note = {Available at: https://github.com/YOUR_USERNAME/dissertation-influencer-ml}
}
```

## Data Access

**Note:** Raw data is **not included** in this repository due to:
- YouTube API Terms of Service (channel-level data)
- Kaggle dataset licensing (Instagram data)
- Privacy considerations

**To replicate:**
1. Obtain YouTube Data API v3 key from Google Cloud Console
2. Download Instagram dataset from [Kaggle](https://www.kaggle.com/datasets/...)
3. Place in `data/raw/` directory
4. Run data collection scripts

Processed features (with anonymized IDs) are available in `data/processed/`.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Supervisor:** Dr. Van-Dinh Nguyen for guidance and feedback
- **Trinity College Dublin** for research resources
- **YouTube Data API v3** for data access
- **Kaggle community** for Instagram dataset

## Contact

**Shaurya Varma**  
MSc Computer Science Student  
Trinity College Dublin  
Email: [varmasl@tcd.ie]  
GitHub: [shauvar](https://github.com/shauvar)



---

**Last Updated:** April 2026  
**Status:** ✅ Complete - Dissertation Submitted
