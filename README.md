# Expression Cleaning Streamlit App

This is a local Streamlit app for gene expression preprocessing & cleaning (GTEx-style, but generalizable).

## What it does
- Load gene-by-sample expression matrix (TSV/CSV; genes in rows, samples in columns)
- log2(x+1) transform
- Low-expression + low-variance filtering
- Optional quantile normalization
- PCA on samples (Altair scatter; optional metadata coloring)
- Outlier detection (Mahalanobis in PCA space, hierarchical Ward clusters, IQR on PCs)
- Optional confounder regression (multi-output linear regression → residuals)
- Export cleaned matrix (CSV or Parquet), plus optional save to disk

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Tips
- For large matrices, prefer **Local path** input instead of upload.
- If the matrix is huge, start by disabling quantile normalization, and/or reduce the number of genes before loading.
- To use metadata:
  - Provide a metadata TSV/CSV and set the sample ID column name (default: `SAMPID`).
  - Then you can color PCA by a metadata column and regress out covariates.
