# Walmart Sales Forecasting

End-to-end Data Science project on the [Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) dataset.  
Focus: reproducible feature engineering, structured model comparison, and experiment tracking with MLflow.

---

## Results

| Model | CV WMAE | CV RMSE |
|---|---|---|
| Ridge (Baseline) | 0.1508 ± 0.0086 | 5681 ± 1066 |
| GBM | 0.0987 ± 0.0068 | 3691 ± 924 |
| LightGBM | 0.0908 ± 0.0062 | 3377 ± 932 |

---


## Dataset

Three CSV files from the Kaggle dataset:

| File | Rows | Description |
|---|---|---|
| `train.csv` | 421,570 | Weekly sales per store/department |
| `stores.csv` | 45 | Store type (A/B/C) and size |
| `features.csv` | 8,190 | Temperature, fuel price, markdowns, CPI, unemployment |

---

## Feature Engineering (`src/features.py`)

All three datasets are merged and the following feature groups are built:

- **Temporal** — year, month, week, quarter + cyclical sin/cos encodings for week and month
- **Holiday** — binary flags for Super Bowl, Labor Day, Thanksgiving, Christmas weeks
- **Markdown** — total spend, active count, binary activity flag (replaces 5 sparse raw columns)
- **Store** — one-hot encoded store type, store size
- **Department** — historical mean and std of weekly sales per (Store, Dept)
- **Lag features** — weekly sales lagged by 1, 2, 4, 8, 12, 52 weeks
- **Rolling features** — rolling mean and std over 4, 8, 12-week windows

Leakage prevention: all lag and rolling features use `.shift(1)` so no future sales
are visible at prediction time.

Final feature matrix: **48 features**, 421,570 rows.

---

## Experiment Tracking (`src/train.py`)

Models are compared using **walk-forward time-series cross-validation** (`TimeSeriesSplit`, 5 folds)
to respect the temporal structure of the data.

Primary metric: **WMAE** (Weighted Mean Absolute Error) — Walmart's official competition metric,
which up-weights holiday weeks.

Each MLflow run logs:
- Hyperparameters
- CV metrics (WMAE, RMSE, MAE) — mean and std across folds
- Training metrics on full dataset
- Model artifact


---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/yourname/walmart-forecasting.git
cd walmart-forecasting
pip install -r requirements.txt
```

**2. Download the data**
```bash
python3 -c "
import kagglehub
path = kagglehub.dataset_download('aslanahmedov/walmart-sales-forecast')
print(path)
"
# Copy train.csv, stores.csv, features.csv into data/raw/
```

**3. Start the MLflow tracking server**
```bash
mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.db
```

**4. Run feature engineering smoke test**
```bash
python3 -m src.features
```

**5. Train all models**
```bash
python3 -m src.train              # all three models
python3 -m src.train --model lightgbm   # single model
```

**6. View experiments**

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
mlflow
kagglehub
```

---
