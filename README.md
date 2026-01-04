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

## Preprocessing Pipeline (ssGSEA-ready)

This app implements a comprehensive preprocessing workflow suitable for downstream analyses like ssGSEA (single-sample Gene Set Enrichment Analysis). The preprocessing steps follow best practices for gene expression data normalization and quality control:

### Step 1: Data Loading
- Supports TSV/CSV formats with flexible column configurations
- Handles GTEx-style files with multiple header rows
- Loads gene annotations and sample metadata
- Merges subject-level phenotypes with sample attributes

### Step 2: Log Transformation & Filtering
**Log2(x+1) Transformation:**
- Transforms raw TPM/count values to log2 scale
- Stabilizes variance across expression ranges
- Makes data more normally distributed for downstream analysis

**Low-Expression Filtering:**
- Removes genes with low expression across samples
- Configurable threshold: minimum TPM value in minimum fraction of samples
- Reduces noise and improves statistical power

**Low-Variance Filtering:**
- Removes genes with low variance across samples
- Applied on log2-transformed data
- Configurable quantile threshold (e.g., keep top 80% most variable genes)
- Removes non-informative genes

### Step 3: Quantile Normalization (Optional)
- Makes sample distributions identical
- Corrects for technical variation between samples
- Uses rank-based method: sort → average by rank → reassign
- Particularly important when samples come from different batches or processing runs

### Step 4: Quality Control with PCA
- Principal Component Analysis on samples
- Visualizes sample relationships and batch effects
- Enables coloring by metadata attributes (e.g., RIN, tissue type, batch)
- Gene-attribute correlation analysis to identify confounders

### Step 5: Outlier Detection
Three complementary methods for robustness:

1. **Mahalanobis Distance in PCA space:**
   - Measures distance from center in multi-dimensional PC space
   - Chi-squared threshold for outlier definition
   - Most sensitive to multivariate outliers

2. **Hierarchical Clustering (Ward linkage):**
   - Identifies samples in small isolated clusters
   - Good for detecting batch-specific outliers
   - Multiple linkage methods available

3. **IQR Method on PCs:**
   - Detects outliers in individual principal components
   - Union of outliers across first N PCs
   - Simple and interpretable

### Step 6: Confounder Regression (Optional)
- Removes technical variation while preserving biological signal
- Multi-output linear regression on selected covariates
- Returns residuals (variation not explained by confounders)
- Supports both numeric and categorical confounders
- Visualizes PCA before/after regression to verify effect

### Final Output
- Fully preprocessed expression matrix ready for downstream analysis
- Compatible with ssGSEA, GSEA, differential expression, etc.
- Export as CSV or Parquet format
- Includes gene annotations and filtered samples only

## Why These Steps?

These preprocessing steps are essential for robust gene expression analysis:

1. **Log transformation:** Required for most statistical methods that assume normality
2. **Filtering:** Removes noise and reduces multiple testing burden
3. **Quantile normalization:** Removes systematic technical variation between samples
4. **QC/Outliers:** Identifies problematic samples that could bias results
5. **Batch correction:** Removes known technical confounders while preserving biology

For ssGSEA specifically, proper preprocessing ensures:
- Consistent gene expression distributions across samples
- Removal of technical artifacts that could create false pathway signals
- Comparable enrichment scores between samples

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
