# House Price Prediction

Predict house prices from property features (area, rooms, location signals, furnishing, etc.) using exploratory data analysis plus multiple regression techniques  OLS via statsmodels, Linear / Ridge / ElasticNet, PCA-based regression, RFE feature selection, and polynomial-degree comparison.

## Dataset

`Housing.csv` - 545 rows × 13 columns (546 lines including header).

| Column | Type | Description |
|---|---|---|
| `price` | int (target) | House sale price |
| `area` | int | Plot / built-up area |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `stories` | int | Number of stories |
| `mainroad` | yes/no | Adjacent to main road |
| `guestroom` | yes/no | Has guest room |
| `basement` | yes/no | Has basement |
| `hotwaterheating` | yes/no | Hot-water heating |
| `airconditioning` | yes/no | Air conditioning |
| `parking` | int | Parking spaces |
| `prefarea` | yes/no | Preferred area |
| `furnishingstatus` | furnished / semi-furnished / unfurnished | Furnishing level |

No missing values. Duplicate rows are detected and removed, and numeric outliers are inspected with the IQR rule before modelling.

## What the notebook does (`notebook.ipynb`, 31 cells)

1. **Load & inspect** — `pd.read_csv`, `df.info()`, `nunique()`, `describe()`.
2. **EDA with plots** — price histogram (KDE), categorical count plots, numeric distributions, boxplots, pairplot + KDE upper triangle, correlation heatmap, strong (`|r| ≥ 0.30`) vs weak price correlations.
3. **Cleaning & encoding**
   - Drop duplicates.
   - IQR outlier inspection on numeric features.
   - Binary yes/no → 1/0; `furnishingstatus` one-hot encoded (drop-first → `furnishingstatus_semi_furnished`, `furnishingstatus_unfurnished`), giving **13 model features**.
4. **Train/test split** — 80/20 via `train_test_split(random_state=100)`, indexes reset.
5. **Scaling** — `StandardScaler` fit on train numerics (`area, bedrooms, bathrooms, stories, parking`) and applied to test; categorical 0/1 columns passed through unchanged, column order preserved.
6. **Statistical modelling (statsmodels OLS)** — formula `price ~ <13 features>` on standardized train data; full summary (R², adj-R², F, p-values, AIC/BIC, Omnibus/JB, Durbin-Watson, condition number) plus a plain-English interpretation of each diagnostic.
7. **Dimensionality & selection experiments**
   - PCA scree plot (explained + cumulative variance, 90% line ≈ 7–8 components) and LinearRegression-on-PCA RMSE vs number of components.
   - RFE (Recursive Feature Elimination) ranking of features with LinearRegression.
8. **Predictive models (scikit-learn) with shared `Evaluate(n, pred1, pred2)`**
   - Multiple Linear Regression, Ridge, ElasticNet — each reports coefficients, intercept, train/test R², RSS, MSE, RMSE, actual-vs-predicted scatter and residual distribution plots, logged into a 5×8 comparison matrix (`Train/Test × R2, RSS, MSE, RMSE`).
   - Polynomial regression (degrees 2–6) train vs test RMSE curves showing overfitting at high degrees.

## Results (current notebook outputs)

- OLS on train (n = 436): **R² ≈ 0.678, adj-R² ≈ 0.668, F ≈ 68.4, p ≈ 3.5e-95**.
- Multiple Linear Regression: **train R² ≈ 0.678, test R² ≈ 0.681; train RMSE ≈ 1,058,055, test RMSE ≈ 1,061,662** (same currency units as `price`).
- Significant positive drivers: `area`, `bathrooms`, `stories`, `airconditioning`, `parking`, `prefarea`; `bedrooms` not significant after controlling for other features (p ≈ 0.275); `semi-furnished` not distinguishable from `furnished` (p ≈ 0.439) while `unfurnished` is significantly negative.
- Residuals are right-skewed (≈ 0.89) with heavy tails — not perfectly normal (see Omnibus/Jarque-Bera), Durbin-Watson ≈ 2.09 (no clear autocorrelation), condition number ≈ 7.66 (no severe multicollinearity flag).
- Re-run the notebook to refresh the full Ridge / ElasticNet / polynomial comparison matrix and plots.

## Getting started

### Prerequisites

- Python 3.10+ (project `.venv` was created with the repo's Python)
- Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`

### Setup

```bash
# clone
git clone https://github.com/PIYUSH-NEXTGEN/House-Price-Prediction.git
cd House-Price-Prediction

# create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
```

### Run the analysis

```bash
# option 1: Jupyter
jupyter notebook notebook.ipynb

# option 2: VS Code - open notebook.ipynb and run cells top-to-bottom
```

> Run cells in order: preprocessing (encoding → split → scaling) must execute before the OLS / PCA / RFE / regression cells. `Housing.csv` must sit next to `notebook.ipynb`.

## Project structure

```
House-Price-Prediction/
├── Housing.csv      # raw dataset (545 rows × 13 cols)
├── notebook.ipynb   # full EDA + preprocessing + OLS + sklearn models
├── README.md        # this file
└── .gitignore
```

## Tech stack

- **Data & EDA:** pandas, numpy, matplotlib, seaborn
- **Modelling:** scikit-learn (LinearRegression, Ridge, ElasticNet, StandardScaler, PCA, RFE, PolynomialFeatures, train_test_split, r2_score, mean_squared_error), statsmodels (OLS formula API)
