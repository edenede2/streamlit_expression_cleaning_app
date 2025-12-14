
"""
Streamlit app: Expression Matrix Preprocessing & Cleaning

Designed to be GENERALIZABLE:
- Works with any gene-by-sample matrix in TSV/CSV (genes in rows, samples in columns)
- Optionally joins a sample metadata table for PCA coloring + confounder regression
- Implements common steps from the provided notebook:
  log2(x+1) transform, low-expression + low-variance filtering, quantile normalization,
  PCA visualization, outlier detection, and confounder regression (residualization).

Run:
  streamlit run app.py

Notes:
- For very large matrices, prefer "Local path" over upload.
- Quantile normalization can be slow for very large matrices; start with a subset / skip it.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2, pearsonr


def quantile_normalize_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile normalization using the exact method from the notebook.
    Input: genes x samples DataFrame
    Output: quantile normalized genes x samples DataFrame
    """
    # Transpose to samples x genes, normalize, transpose back
    df_t = df.T
    df_sorted = pd.DataFrame(
        np.sort(df_t.values, axis=0),
        index=df_t.index,
        columns=df_t.columns
    )
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn = df_t.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return df_qn.T


def compute_gene_attribute_correlations(
    expression_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_id_col: str,
    attributes: List[str],
    max_genes: int = 5000
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute Pearson correlations between gene expression and sample attributes.
    Returns a DataFrame with correlations and a summary dict.
    """
    sample_ids = list(expression_df.columns)
    
    # Pre-encode categorical attributes
    batch_attributes_encoded = {}
    for attr in attributes:
        if attr not in meta_df.columns:
            continue
        attr_data = meta_df.set_index(sample_id_col).reindex(sample_ids)[attr]
        if not pd.api.types.is_numeric_dtype(attr_data):
            batch_attributes_encoded[attr] = pd.Categorical(attr_data).codes.astype(float)
            # Replace -1 (missing) with NaN
            batch_attributes_encoded[attr] = np.where(
                batch_attributes_encoded[attr] == -1, np.nan, batch_attributes_encoded[attr]
            )
        else:
            batch_attributes_encoded[attr] = attr_data.values.astype(float)
    
    # Limit genes for speed
    genes_to_use = expression_df.index[:max_genes].tolist()
    
    results = []
    for gene in genes_to_use:
        gene_data = expression_df.loc[gene, :].values.astype(float)
        row = {'gene': gene}
        
        for attr in attributes:
            if attr not in batch_attributes_encoded:
                row[f'{attr}_correlation'] = np.nan
                row[f'{attr}_pvalue'] = np.nan
                continue
                
            attr_data = batch_attributes_encoded[attr]
            
            # Remove NaN values
            mask = ~(np.isnan(gene_data) | np.isnan(attr_data))
            if mask.sum() >= 2:
                corr, pval = pearsonr(gene_data[mask], attr_data[mask])
                row[f'{attr}_correlation'] = corr
                row[f'{attr}_pvalue'] = pval
            else:
                row[f'{attr}_correlation'] = np.nan
                row[f'{attr}_pvalue'] = np.nan
        
        results.append(row)
    
    corr_df = pd.DataFrame(results)
    
    # Summary statistics
    summary = {}
    for attr in attributes:
        col = f'{attr}_correlation'
        if col in corr_df.columns:
            vals = corr_df[col].dropna()
            if len(vals) > 0:
                summary[attr] = {
                    'n_genes': len(vals),
                    'mean': vals.mean(),
                    'median': vals.median(),
                    'std': vals.std(),
                    'min': vals.min(),
                    'max': vals.max()
                }
    
    return corr_df, summary


@dataclass(frozen=True)
class ExpressionData:
    """In-memory expression matrix with optional gene annotations."""
    genes: pd.DataFrame  # gene annotation columns (may be empty)
    X: pd.DataFrame      # gene x sample numeric matrix
    gene_id_col: str     # a display column name for genes (may be empty string)


# ---------- Helpers ----------
def _save_upload_to_temp(uploaded: "st.runtime.uploaded_file_manager.UploadedFile") -> str:
    suffix = os.path.splitext(uploaded.name)[1] or ".txt"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _infer_sep(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in [".tsv", ".txt", ".gct"]:
        return "\t"
    return ","


@st.cache_data(show_spinner=False)
def read_expression(
    path: str,
    sep: str,
    skip_rows: int,
    gene_cols: Tuple[str, ...],
    n_preview_rows: int = 8,
    infer_schema_length: int = 2000,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Reads the expression file and returns:
      - preview dataframe (first rows)
      - column list
      - numeric candidate columns (best-effort)
    """
    lf = pl.scan_csv(
        path,
        separator=sep,
        skip_rows=skip_rows,
        has_header=True,
        infer_schema_length=infer_schema_length,
        null_values=["NA", "NaN", "nan", ""],
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    # Collect a small preview early
    # Use polars native methods to avoid numpy/pyarrow compatibility issues
    pl_preview = lf.head(n_preview_rows).collect()
    cols = pl_preview.columns
    # Convert via dict to avoid pyarrow->numpy issues
    preview = pd.DataFrame(pl_preview.to_dict())

    # Best-effort numeric detection from preview:
    numeric_candidates: List[str] = []
    for c in cols:
        if c in gene_cols:
            continue
        ser = pd.to_numeric(preview[c], errors="coerce")
        # treat as numeric if most values coerce successfully (or are missing)
        ok = ser.notna().mean()
        if ok >= 0.8:
            numeric_candidates.append(c)

    return preview, cols, numeric_candidates


@st.cache_data(show_spinner=False)
def load_expression_to_memory(
    path: str,
    sep: str,
    skip_rows: int,
    gene_cols: Tuple[str, ...],
    sample_cols: Tuple[str, ...],
    cast_float: bool = True,
    infer_schema_length: int = 2000,
) -> ExpressionData:
    """
    Loads the expression matrix into memory as pandas DataFrames.
    Returns gene annotation table and numeric matrix (genes x samples).
    """
    lf = pl.scan_csv(
        path,
        separator=sep,
        skip_rows=skip_rows,
        has_header=True,
        infer_schema_length=infer_schema_length,
        null_values=["NA", "NaN", "nan", ""],
        ignore_errors=True,
        truncate_ragged_lines=True,
    )

    # Ensure sample columns are numeric
    if cast_float:
        lf = lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in sample_cols])

    pl_df = lf.select(list(gene_cols) + list(sample_cols)).collect()
    # Convert via dict to avoid pyarrow->numpy compatibility issues
    df = pd.DataFrame(pl_df.to_dict())

    genes = df[list(gene_cols)].copy() if len(gene_cols) else pd.DataFrame(index=df.index)
    X = df[list(sample_cols)].copy()

    # Choose a display gene column (best guess)
    gene_id_col = ""
    for cand in ("Name", "gene", "Gene", "gene_id", "id"):
        if cand in genes.columns:
            gene_id_col = cand
            break

    return ExpressionData(genes=genes, X=X, gene_id_col=gene_id_col)


def log2p1(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df.astype(float) + 1.0)


def filter_low_expression_and_variance(
    X_raw: pd.DataFrame,
    min_tpm: float,
    min_samples: int,
    var_quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean masks:
      keep_expr: gene passes expression filter on raw scale
      keep_var:  gene passes variance filter on log2(x+1) scale
    """
    Xv = X_raw.to_numpy(dtype=float, copy=False)
    keep_expr = (Xv >= min_tpm).sum(axis=1) >= int(min_samples)

    Xlog = np.log2(Xv + 1.0)
    variances = np.var(Xlog, axis=1, ddof=1)
    thr = np.quantile(variances, var_quantile)
    keep_var = variances >= thr

    return keep_expr, keep_var


def quantile_normalize_fast(X: np.ndarray) -> np.ndarray:
    """
    Fast quantile normalization (approx tie handling).
    X shape: (n_genes, n_samples) or (n_rows, n_cols)

    Algorithm:
      1) sort each column
      2) average across columns by sorted rank
      3) assign rank-means back to original order per column

    Note: ties in a column get the mean of their order positions (approx).
    """
    # Sort each column
    order = np.argsort(X, axis=0)
    X_sorted = np.take_along_axis(X, order, axis=0)
    mean_sorted = np.mean(X_sorted, axis=1)

    X_qn = np.empty_like(X_sorted)
    # place mean_sorted at sorted positions, then invert ordering
    for j in range(X.shape[1]):
        X_qn[order[:, j], j] = mean_sorted
    return X_qn


def pca_on_samples(X_gene_by_sample: pd.DataFrame, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """
    PCA on samples (columns). Input is genes x samples.
    Returns (scores: samples x n_components, fitted PCA).
    """
    # samples x genes
    Y = X_gene_by_sample.T.to_numpy(dtype=float)
    scaler = StandardScaler()
    Yz = scaler.fit_transform(Y)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(Yz)
    return scores, pca


def detect_outliers_mahalanobis(
    scores: np.ndarray,
    df: int,
    chi2_quantile: float,
) -> np.ndarray:
    """
    Outlier detection by Mahalanobis distance in PCA space.
    Returns boolean mask (samples,).
    """
    center = np.mean(scores[:, :df], axis=0)
    cov = np.cov(scores[:, :df].T)
    inv_cov = np.linalg.pinv(cov)
    d = np.array([mahalanobis(s[:df], center, inv_cov) for s in scores])
    thr = float(chi2.ppf(chi2_quantile, df=df))
    return d > thr


def detect_outliers_hierarchical(
    scores: np.ndarray,
    cut_fraction_of_max: float,
    min_cluster_fraction: float,
    method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Outliers as samples in small clusters after hierarchical linkage in PCA space.
    Returns (boolean mask, linkage matrix, cut height).
    """
    Z = linkage(scores, method=method)
    max_d = float(cut_fraction_of_max * np.max(Z[:, 2]))
    clusters = fcluster(Z, max_d, criterion="distance")

    counts = pd.Series(clusters).value_counts()
    min_size = max(1, int(min_cluster_fraction * len(clusters)))
    small = set(counts[counts < min_size].index.tolist())

    outlier_mask = np.array([c in small for c in clusters], dtype=bool)
    return outlier_mask, Z, max_d


def detect_outliers_iqr(
    scores: np.ndarray,
    n_pcs: int,
    iqr_k: float,
) -> np.ndarray:
    """
    Outliers by IQR in first n_pcs PCs. Union across PCs.
    Returns boolean mask (samples,).
    """
    n_pcs = int(min(n_pcs, scores.shape[1]))
    out = np.zeros(scores.shape[0], dtype=bool)
    for k in range(n_pcs):
        v = scores[:, k]
        q1, q3 = np.quantile(v, [0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        out |= (v < lo) | (v > hi)
    return out


def build_covariate_matrix(
    meta: pd.DataFrame,
    sample_ids: List[str],
    sample_id_col: str,
    covariates: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Builds a design matrix X for regression with one-hot encoding for categorical covariates.
    Returns (X, feature_names). Rows aligned to sample_ids order.
    """
    m = meta.set_index(sample_id_col).loc[sample_ids, covariates].copy()

    # Separate numeric vs categorical
    numeric_cols = []
    cat_cols = []
    for c in covariates:
        if pd.api.types.is_numeric_dtype(m[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    parts = []
    feature_names: List[str] = []

    if numeric_cols:
        parts.append(m[numeric_cols].astype(float))
        feature_names += numeric_cols

    if cat_cols:
        dummies = pd.get_dummies(m[cat_cols].astype("string"), dummy_na=True, drop_first=False)
        parts.append(dummies.astype(float))
        feature_names += dummies.columns.astype(str).tolist()

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sample_ids)
    # Drop rows with NaNs (after dummy_na, remaining NaNs mostly from numeric covariates)
    mask = ~X.isna().any(axis=1)
    X_clean = X.loc[mask].to_numpy(dtype=float)
    kept = X.index[mask].tolist()
    return X_clean, kept


def regress_out_confounders(
    X_gene_by_sample: pd.DataFrame,
    meta: pd.DataFrame,
    sample_id_col: str,
    covariates: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Regresses covariates from expression (multi-output linear regression).
    Input: genes x samples. Output: residuals genes x samples (for samples kept without NaNs).
    """
    sample_ids = list(X_gene_by_sample.columns)
    Xcov, kept_samples = build_covariate_matrix(meta, sample_ids, sample_id_col, covariates)

    # Align expression to kept_samples
    Y = X_gene_by_sample[kept_samples].T.to_numpy(dtype=float)  # samples x genes

    model = LinearRegression(fit_intercept=True)
    model.fit(Xcov, Y)
    Yhat = model.predict(Xcov)
    resid = Y - Yhat  # samples x genes

    resid_df = pd.DataFrame(resid.T, index=X_gene_by_sample.index, columns=kept_samples)
    return resid_df, kept_samples


def sample_values_for_hist(X: pd.DataFrame, n: int = 50000, seed: int = 0) -> np.ndarray:
    v = X.to_numpy(dtype=float, copy=False).ravel()
    if v.size <= n:
        return v
    rng = np.random.default_rng(seed)
    idx = rng.choice(v.size, size=n, replace=False)
    return v[idx]


def compute_expression_filter_stats(
    X_raw: pd.DataFrame,
    min_tpm_values: np.ndarray,
    min_samples: int
) -> pd.DataFrame:
    """
    Compute how many genes pass expression filter at different thresholds.
    Returns DataFrame with threshold and genes_kept columns.
    """
    Xv = X_raw.to_numpy(dtype=float, copy=False)
    results = []
    for tpm in min_tpm_values:
        keep = (Xv >= tpm).sum(axis=1) >= min_samples
        results.append({'min_tpm': tpm, 'genes_kept': int(keep.sum()), 'genes_removed': int((~keep).sum())})
    return pd.DataFrame(results)


def compute_variance_filter_stats(
    X_raw: pd.DataFrame,
    var_quantiles: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute how many genes pass variance filter at different quantile cutoffs.
    Returns DataFrame with quantile and genes_kept columns, plus the variance array.
    """
    Xlog = np.log2(X_raw.to_numpy(dtype=float, copy=False) + 1.0)
    variances = np.var(Xlog, axis=1, ddof=1)
    
    results = []
    for q in var_quantiles:
        thr = np.quantile(variances, q)
        keep = variances >= thr
        results.append({
            'quantile': q, 
            'variance_threshold': thr,
            'genes_kept': int(keep.sum()), 
            'genes_removed': int((~keep).sum())
        })
    return pd.DataFrame(results), variances


def plotly_expression_threshold_explorer(
    filter_stats: pd.DataFrame,
    current_threshold: float,
    title: str = "Genes Kept vs Expression Threshold"
) -> go.Figure:
    """
    Interactive plot showing how many genes are kept at different expression thresholds.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Genes kept line
    fig.add_trace(
        go.Scatter(
            x=filter_stats['min_tpm'],
            y=filter_stats['genes_kept'],
            mode='lines+markers',
            name='Genes Kept',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Genes removed line
    fig.add_trace(
        go.Scatter(
            x=filter_stats['min_tpm'],
            y=filter_stats['genes_removed'],
            mode='lines+markers',
            name='Genes Removed',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Current threshold marker
    current_row = filter_stats[filter_stats['min_tpm'] == current_threshold]
    if len(current_row) > 0:
        fig.add_vline(x=current_threshold, line_dash="dash", line_color="green",
                      annotation_text=f"Current: {current_threshold}")
    
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='Min Expression Threshold (TPM)')
    fig.update_yaxes(title_text='Genes Kept', secondary_y=False)
    fig.update_yaxes(title_text='Genes Removed', secondary_y=True)
    
    return fig


def plotly_variance_distribution(
    variances: np.ndarray,
    current_quantile: float,
    title: str = "Gene Variance Distribution (log2 scale)"
) -> go.Figure:
    """
    Histogram of gene variances with quantile markers.
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=variances,
        nbinsx=60,
        marker_color='steelblue',
        opacity=0.75,
        name='Variances'
    ))
    
    # Add quantile markers
    quantiles_to_show = [0.1, 0.2, 0.3, 0.5]
    colors = ['orange', 'green', 'purple', 'brown']
    for q, color in zip(quantiles_to_show, colors):
        thr = np.quantile(variances, q)
        fig.add_vline(x=thr, line_dash="dot", line_color=color,
                      annotation_text=f"Q{int(q*100)}: {thr:.3f}")
    
    # Current threshold
    current_thr = np.quantile(variances, current_quantile)
    fig.add_vline(x=current_thr, line_dash="solid", line_color="red", line_width=3,
                  annotation_text=f"Current ({current_quantile:.0%}): {current_thr:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title='Variance',
        yaxis_title='Count',
        height=400,
        template='plotly_white'
    )
    return fig


def plotly_variance_quantile_explorer(
    filter_stats: pd.DataFrame,
    current_quantile: float,
    title: str = "Genes Kept vs Variance Quantile Cutoff"
) -> go.Figure:
    """
    Interactive plot showing genes kept at different variance quantile cutoffs.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Genes kept
    fig.add_trace(
        go.Scatter(
            x=filter_stats['quantile'] * 100,
            y=filter_stats['genes_kept'],
            mode='lines+markers',
            name='Genes Kept',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Variance threshold
    fig.add_trace(
        go.Scatter(
            x=filter_stats['quantile'] * 100,
            y=filter_stats['variance_threshold'],
            mode='lines+markers',
            name='Variance Threshold',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Current quantile marker
    fig.add_vline(x=current_quantile * 100, line_dash="dash", line_color="green",
                  annotation_text=f"Current: {current_quantile:.0%}")
    
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='Variance Quantile Cutoff (%)')
    fig.update_yaxes(title_text='Genes Kept', secondary_y=False)
    fig.update_yaxes(title_text='Variance Threshold', secondary_y=True)
    
    return fig


def plotly_expression_vs_variance(
    X_raw: pd.DataFrame,
    n_sample: int = 5000,
    title: str = "Mean Expression vs Variance"
) -> go.Figure:
    """
    Scatter plot of mean expression vs variance for each gene.
    Helps identify relationship and potential outliers.
    """
    Xv = X_raw.to_numpy(dtype=float, copy=False)
    Xlog = np.log2(Xv + 1.0)
    
    means = np.mean(Xlog, axis=1)
    variances = np.var(Xlog, axis=1, ddof=1)
    
    # Sample if too many genes
    if len(means) > n_sample:
        idx = np.random.choice(len(means), n_sample, replace=False)
        means = means[idx]
        variances = variances[idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means,
        y=variances,
        mode='markers',
        marker=dict(size=4, color='steelblue', opacity=0.5),
        name='Genes'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Mean Expression (log2)',
        yaxis_title='Variance (log2)',
        height=400,
        template='plotly_white'
    )
    return fig


def plotly_samples_expressing_gene(
    X_raw: pd.DataFrame,
    min_tpm: float,
    title: str = "Distribution: Samples Expressing Each Gene"
) -> go.Figure:
    """
    Histogram showing how many samples express each gene above threshold.
    """
    Xv = X_raw.to_numpy(dtype=float, copy=False)
    samples_expressing = (Xv >= min_tpm).sum(axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples_expressing,
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.75
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f'Number of Samples with Expression ≥ {min_tpm}',
        yaxis_title='Number of Genes',
        height=350,
        template='plotly_white'
    )
    return fig


def plotly_combined_filter_preview(
    X_raw: pd.DataFrame,
    min_tpm: float,
    min_samples: int,
    var_quantile: float,
    title: str = "Filter Preview: Expression vs Variance with Filter Regions"
) -> go.Figure:
    """
    Scatter plot showing genes colored by whether they pass filters.
    """
    Xv = X_raw.to_numpy(dtype=float, copy=False)
    Xlog = np.log2(Xv + 1.0)
    
    # Compute metrics
    samples_expressing = (Xv >= min_tpm).sum(axis=1)
    variances = np.var(Xlog, axis=1, ddof=1)
    var_threshold = np.quantile(variances, var_quantile)
    
    # Filter status
    pass_expr = samples_expressing >= min_samples
    pass_var = variances >= var_threshold
    pass_both = pass_expr & pass_var
    
    # Sample for visualization
    n_sample = 5000
    if len(variances) > n_sample:
        idx = np.random.choice(len(variances), n_sample, replace=False)
    else:
        idx = np.arange(len(variances))
    
    fig = go.Figure()
    
    # Genes that fail both
    mask = ~pass_expr[idx] & ~pass_var[idx]
    if mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=samples_expressing[idx][mask],
            y=variances[idx][mask],
            mode='markers',
            marker=dict(size=4, color='lightgray', opacity=0.5),
            name='Fail Both'
        ))
    
    # Genes that pass only expression
    mask = pass_expr[idx] & ~pass_var[idx]
    if mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=samples_expressing[idx][mask],
            y=variances[idx][mask],
            mode='markers',
            marker=dict(size=4, color='orange', opacity=0.6),
            name='Pass Expression Only'
        ))
    
    # Genes that pass only variance
    mask = ~pass_expr[idx] & pass_var[idx]
    if mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=samples_expressing[idx][mask],
            y=variances[idx][mask],
            mode='markers',
            marker=dict(size=4, color='purple', opacity=0.6),
            name='Pass Variance Only'
        ))
    
    # Genes that pass both
    mask = pass_both[idx]
    if mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=samples_expressing[idx][mask],
            y=variances[idx][mask],
            mode='markers',
            marker=dict(size=4, color='green', opacity=0.6),
            name='Pass Both (Kept)'
        ))
    
    # Add threshold lines
    fig.add_vline(x=min_samples, line_dash="dash", line_color="red",
                  annotation_text=f"Min samples: {min_samples}")
    fig.add_hline(y=var_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Var threshold: {var_threshold:.4f}")
    
    fig.update_layout(
        title=title,
        xaxis_title=f'Samples with Expression ≥ {min_tpm}',
        yaxis_title='Variance (log2 scale)',
        height=500,
        template='plotly_white'
    )
    return fig


def plotly_hist(values: np.ndarray, title: str, color: str = "steelblue") -> go.Figure:
    """Create a histogram using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=60,
        marker_color=color,
        opacity=0.75
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        height=300,
        template="plotly_white"
    )
    return fig


def plotly_pca_scatter(
    scores: np.ndarray,
    sample_ids: List[str],
    color: Optional[pd.Series],
    title: str,
) -> go.Figure:
    """Create a PCA scatter plot using Plotly."""
    fig = go.Figure()
    
    if color is not None:
        color_vals = color.values
        is_numeric = pd.api.types.is_numeric_dtype(color_vals)
        
        if is_numeric:
            fig.add_trace(go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_vals,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color.name),
                    opacity=0.8
                ),
                text=[f"Sample: {s}<br>{color.name}: {c:.2f}" if isinstance(c, (int, float)) else f"Sample: {s}<br>{color.name}: {c}"
                      for s, c in zip(sample_ids, color_vals)],
                hoverinfo='text'
            ))
        else:
            # Categorical - use discrete colors
            unique_vals = pd.unique(color_vals)
            colors = px.colors.qualitative.Set1[:len(unique_vals)]
            color_map = {v: colors[i % len(colors)] for i, v in enumerate(unique_vals)}
            
            for val in unique_vals:
                mask = color_vals == val
                fig.add_trace(go.Scatter(
                    x=scores[mask, 0],
                    y=scores[mask, 1],
                    mode='markers',
                    marker=dict(size=10, color=color_map[val], opacity=0.8),
                    name=str(val),
                    text=[f"Sample: {s}<br>{color.name}: {val}" for s in np.array(sample_ids)[mask]],
                    hoverinfo='text'
                ))
    else:
        fig.add_trace(go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1],
            mode='markers',
            marker=dict(size=10, color='steelblue', opacity=0.8),
            text=[f"Sample: {s}" for s in sample_ids],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=450,
        template='plotly_white'
    )
    return fig


def plotly_distribution_comparison(
    before: np.ndarray,
    after: np.ndarray,
    before_title: str = "Before",
    after_title: str = "After",
    main_title: str = "Distribution Comparison"
) -> go.Figure:
    """Create side-by-side histogram comparison using Plotly subplots."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(before_title, after_title),
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Histogram(x=before, nbinsx=50, marker_color='steelblue', opacity=0.75, name='Before'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=after, nbinsx=50, marker_color='green', opacity=0.75, name='After'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=main_title,
        height=350,
        showlegend=False,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='Value', row=1, col=1)
    fig.update_xaxes(title_text='Value', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    
    return fig


def plotly_outlier_detection(
    scores: np.ndarray,
    sample_ids: List[str],
    mahal_distances: np.ndarray,
    threshold: float,
    outlier_mask: np.ndarray
) -> go.Figure:
    """Create Mahalanobis distance histogram + PCA scatter with outliers marked."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Mahalanobis Distance Distribution', 'PC1 vs PC2 with Outliers Marked'),
        horizontal_spacing=0.12
    )
    
    # Histogram of distances
    fig.add_trace(
        go.Histogram(x=mahal_distances, nbinsx=30, marker_color='skyblue', name='Distances'),
        row=1, col=1
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold:.2f}", row=1, col=1)
    
    # PCA scatter with outliers
    colors = ['red' if o else 'blue' for o in outlier_mask]
    fig.add_trace(
        go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1],
            mode='markers',
            marker=dict(size=8, color=colors, opacity=0.7),
            text=[f"Sample: {s}<br>Distance: {d:.2f}" for s, d in zip(sample_ids, mahal_distances)],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Mahalanobis Distance', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='PC1', row=1, col=2)
    fig.update_yaxes(title_text='PC2', row=1, col=2)
    
    fig.update_layout(
        height=450,
        title_text='PCA-based Outlier Detection (Mahalanobis)',
        template='plotly_white'
    )
    return fig


def plotly_mahal_qq_plot(
    mahal_distances: np.ndarray,
    df: int,
    title: str = "Chi-Squared Q-Q Plot for Mahalanobis Distances"
) -> go.Figure:
    """
    Create a Q-Q plot comparing Mahalanobis distances to chi-squared distribution.
    If points follow the diagonal, distances are approximately chi-squared distributed.
    """
    # Sort observed distances
    sorted_d2 = np.sort(mahal_distances ** 2)  # Squared distances follow chi2
    n = len(sorted_d2)
    
    # Theoretical quantiles from chi-squared distribution
    theoretical_quantiles = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=df)
    
    fig = go.Figure()
    
    # Q-Q scatter
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sorted_d2,
        mode='markers',
        marker=dict(size=6, color='steelblue', opacity=0.6),
        name='Samples'
    ))
    
    # Reference line (y = x)
    max_val = max(theoretical_quantiles.max(), sorted_d2.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Reference (y=x)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f'Theoretical Quantiles (χ² df={df})',
        yaxis_title='Observed Squared Mahalanobis Distances',
        height=400,
        template='plotly_white'
    )
    return fig


def plotly_mahal_pca_ellipse(
    scores: np.ndarray,
    sample_ids: List[str],
    mahal_distances: np.ndarray,
    outlier_mask: np.ndarray,
    center: np.ndarray,
    cov_2d: np.ndarray,
    chi2_quantile: float,
    df: int = 2,
    title: str = "PCA with Mahalanobis Confidence Ellipse"
) -> go.Figure:
    """
    Create PCA scatter plot with Mahalanobis confidence ellipse.
    The ellipse shows the boundary for the given chi-squared quantile.
    """
    fig = go.Figure()
    
    # Draw confidence ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    chi2_val = chi2.ppf(chi2_quantile, df=df)
    
    # Eigenvalue decomposition for ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Ellipse radii
    a = np.sqrt(chi2_val * eigenvalues[0])
    b = np.sqrt(chi2_val * eigenvalues[1])
    
    # Rotation angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Ellipse points
    x_ellipse = center[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y_ellipse = center[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    
    # Add ellipse
    fig.add_trace(go.Scatter(
        x=x_ellipse,
        y=y_ellipse,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'{chi2_quantile*100:.0f}% Confidence Ellipse'
    ))
    
    # Add center point
    fig.add_trace(go.Scatter(
        x=[center[0]],
        y=[center[1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name='Center'
    ))
    
    # Scatter plot with coloring by distance
    fig.add_trace(go.Scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=mahal_distances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Mahal. Dist.'),
            opacity=0.7
        ),
        text=[f"Sample: {s}<br>Distance: {d:.2f}<br>Outlier: {o}" 
              for s, d, o in zip(sample_ids, mahal_distances, outlier_mask)],
        hoverinfo='text',
        name='Samples'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=500,
        template='plotly_white'
    )
    return fig


def plotly_mahal_pc_contributions(
    scores: np.ndarray,
    sample_ids: List[str],
    center: np.ndarray,
    inv_cov: np.ndarray,
    outlier_mask: np.ndarray,
    n_pcs: int = 5,
    title: str = "Per-PC Contribution to Mahalanobis Distance"
) -> go.Figure:
    """
    Show how much each PC contributes to the Mahalanobis distance for each sample.
    Helps identify which PCs are driving outlier status.
    """
    n_pcs = min(n_pcs, scores.shape[1], len(center))
    
    # Calculate per-PC contribution (squared deviation weighted by inverse covariance)
    deviations = scores[:, :n_pcs] - center[:n_pcs]
    
    # For visualization, show absolute weighted deviations per PC
    # Use diagonal of inv_cov as approximation for independent contributions
    contributions = np.abs(deviations) * np.sqrt(np.abs(np.diag(inv_cov)[:n_pcs]))
    
    fig = go.Figure()
    
    # Create grouped bar chart for outliers vs non-outliers
    pc_labels = [f'PC{i+1}' for i in range(n_pcs)]
    
    # Mean contribution for non-outliers
    non_outlier_contrib = contributions[~outlier_mask].mean(axis=0) if (~outlier_mask).sum() > 0 else np.zeros(n_pcs)
    fig.add_trace(go.Bar(
        x=pc_labels,
        y=non_outlier_contrib,
        name='Non-Outliers (mean)',
        marker_color='steelblue'
    ))
    
    # Mean contribution for outliers
    if outlier_mask.sum() > 0:
        outlier_contrib = contributions[outlier_mask].mean(axis=0)
        fig.add_trace(go.Bar(
            x=pc_labels,
            y=outlier_contrib,
            name='Outliers (mean)',
            marker_color='red'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Component',
        yaxis_title='Mean Contribution to Mahalanobis Distance',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    return fig


def plotly_mahal_distance_vs_pcs(
    scores: np.ndarray,
    sample_ids: List[str],
    mahal_distances: np.ndarray,
    outlier_mask: np.ndarray,
    n_pcs: int = 4,
    title: str = "Mahalanobis Distance vs Individual PCs"
) -> go.Figure:
    """
    Create scatter plots showing relationship between Mahalanobis distance and each PC.
    """
    n_pcs = min(n_pcs, scores.shape[1])
    cols = min(2, n_pcs)
    rows = (n_pcs + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'PC{i+1}' for i in range(n_pcs)],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['red' if o else 'steelblue' for o in outlier_mask]
    
    for i in range(n_pcs):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=scores[:, i],
                y=mahal_distances,
                mode='markers',
                marker=dict(size=6, color=colors, opacity=0.6),
                text=[f"Sample: {s}<br>PC{i+1}: {p:.2f}<br>Distance: {d:.2f}" 
                      for s, p, d in zip(sample_ids, scores[:, i], mahal_distances)],
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text=f'PC{i+1}', row=row, col=col)
        fig.update_yaxes(title_text='Mahal. Distance', row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title_text=title,
        template='plotly_white'
    )
    return fig


def plotly_mahal_ranked_samples(
    sample_ids: List[str],
    mahal_distances: np.ndarray,
    threshold: float,
    n_show: int = 30,
    title: str = "Ranked Samples by Mahalanobis Distance"
) -> go.Figure:
    """
    Bar chart showing top N samples ranked by Mahalanobis distance.
    """
    # Sort by distance
    sorted_idx = np.argsort(mahal_distances)[::-1]
    n_show = min(n_show, len(sample_ids))
    
    top_samples = [sample_ids[i][:20] for i in sorted_idx[:n_show]]
    top_distances = mahal_distances[sorted_idx[:n_show]]
    
    # Color by outlier status
    colors = ['red' if d > threshold else 'steelblue' for d in top_distances]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_samples,
        y=top_distances,
        marker_color=colors,
        text=[f'{d:.2f}' for d in top_distances],
        textposition='outside'
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold:.2f}")
    
    fig.update_layout(
        title=title,
        xaxis_title='Sample',
        yaxis_title='Mahalanobis Distance',
        height=450,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    return fig


def plotly_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    title: str = "Hierarchical Clustering Dendrogram"
) -> go.Figure:
    """Create a dendrogram using Plotly based on scipy dendrogram output."""
    # Get dendrogram data
    dend = dendrogram(linkage_matrix, labels=labels, no_plot=True)
    
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    fig = go.Figure()
    
    # Draw the dendrogram lines
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='blue', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add labels at the bottom
    for i, label in enumerate(dend['ivl']):
        fig.add_annotation(
            x=5 + i * 10,
            y=0,
            text=label[:15] + '...' if len(label) > 15 else label,
            showarrow=False,
            textangle=-90,
            font=dict(size=6)
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Samples',
        yaxis_title='Distance',
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plotly_pca_comparison(
    scores_before: np.ndarray,
    scores_after: np.ndarray,
    samples_before: List[str],
    samples_after: List[str],
    color_before: Optional[np.ndarray] = None,
    color_after: Optional[np.ndarray] = None,
    color_name: str = "Value"
) -> go.Figure:
    """Compare PCA before and after (e.g., outlier removal or regression)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Before', 'After'),
        horizontal_spacing=0.1
    )
    
    # Before
    if color_before is not None:
        fig.add_trace(
            go.Scatter(
                x=scores_before[:, 0],
                y=scores_before[:, 1],
                mode='markers',
                marker=dict(size=8, color=color_before, colorscale='Viridis',
                           showscale=True, colorbar=dict(title=color_name, x=0.45)),
                text=[f"Sample: {s}<br>{color_name}: {v:.2f}" if isinstance(v, (int, float, np.floating)) else f"Sample: {s}"
                      for s, v in zip(samples_before, color_before)],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=scores_before[:, 0],
                y=scores_before[:, 1],
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.7),
                text=[f"Sample: {s}" for s in samples_before],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # After
    if color_after is not None:
        fig.add_trace(
            go.Scatter(
                x=scores_after[:, 0],
                y=scores_after[:, 1],
                mode='markers',
                marker=dict(size=8, color=color_after, colorscale='Viridis',
                           showscale=True, colorbar=dict(title=color_name, x=1.0)),
                text=[f"Sample: {s}<br>{color_name}: {v:.2f}" if isinstance(v, (int, float, np.floating)) else f"Sample: {s}"
                      for s, v in zip(samples_after, color_after)],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=scores_after[:, 0],
                y=scores_after[:, 1],
                mode='markers',
                marker=dict(size=8, color='steelblue', opacity=0.7),
                text=[f"Sample: {s}" for s in samples_after],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text='PC1', row=1, col=1)
    fig.update_xaxes(title_text='PC1', row=1, col=2)
    fig.update_yaxes(title_text='PC2', row=1, col=1)
    fig.update_yaxes(title_text='PC2', row=1, col=2)
    
    fig.update_layout(
        height=450,
        title_text='PCA Comparison',
        template='plotly_white'
    )
    return fig


def plotly_boxplot_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    n_genes: int = 5,
    title: str = "Boxplot Comparison Before and After Normalization"
) -> go.Figure:
    """Create boxplots comparing distributions before and after normalization for first n genes."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Before', 'After'),
        horizontal_spacing=0.1
    )
    
    genes = before_df.index[:n_genes].tolist()
    colors = px.colors.qualitative.Set2
    
    for i, gene in enumerate(genes):
        gene_label = f"Gene {i+1}"
        fig.add_trace(
            go.Box(y=before_df.loc[gene].values, name=gene_label, 
                   marker_color=colors[i % len(colors)], showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=after_df.loc[gene].values, name=gene_label,
                   marker_color=colors[i % len(colors)], showlegend=False),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        title_text=title,
        template='plotly_white'
    )
    fig.update_yaxes(title_text='Log2(TPM + 1)', row=1, col=1)
    fig.update_yaxes(title_text='Log2(TPM + 1)', row=1, col=2)
    
    return fig


def plotly_sample_boxplots(
    df: pd.DataFrame,
    n_samples: int = 20,
    title: str = "Sample Distribution Boxplots"
) -> go.Figure:
    """Create boxplots showing distribution of gene expression per sample."""
    samples = df.columns[:n_samples].tolist()
    
    fig = go.Figure()
    for sample in samples:
        fig.add_trace(go.Box(
            y=df[sample].values,
            name=sample[:15] + '...' if len(sample) > 15 else sample,
            marker_color='steelblue',
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sample',
        yaxis_title='Expression',
        height=400,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    return fig


def plotly_correlation_heatmap(
    corr_df: pd.DataFrame,
    attributes: List[str],
    n_top_genes: int = 20,
    title: str = "Top Correlated Genes Heatmap"
) -> go.Figure:
    """Create a heatmap of top correlated genes with attributes."""
    # Get top genes by max absolute correlation across attributes
    corr_cols = [f'{attr}_correlation' for attr in attributes if f'{attr}_correlation' in corr_df.columns]
    
    if not corr_cols:
        return go.Figure().update_layout(title="No correlation data available")
    
    corr_df['max_abs_corr'] = corr_df[corr_cols].abs().max(axis=1)
    top_genes = corr_df.nlargest(n_top_genes, 'max_abs_corr')
    
    # Build heatmap data
    z_data = []
    y_labels = []
    for _, row in top_genes.iterrows():
        gene = row['gene']
        z_row = [row.get(f'{attr}_correlation', np.nan) for attr in attributes]
        z_data.append(z_row)
        y_labels.append(gene[:20] + '...' if len(str(gene)) > 20 else str(gene))
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=attributes,
        y=y_labels,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Attribute',
        yaxis_title='Gene',
        height=500,
        template='plotly_white'
    )
    return fig


def plotly_correlation_distribution(
    corr_df: pd.DataFrame,
    attributes: List[str],
    title: str = "Distribution of Gene-Attribute Correlations"
) -> go.Figure:
    """Create histograms showing distribution of correlations for each attribute."""
    n_attrs = len(attributes)
    cols = min(2, n_attrs)
    rows = (n_attrs + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=attributes[:n_attrs],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, attr in enumerate(attributes):
        col_name = f'{attr}_correlation'
        if col_name not in corr_df.columns:
            continue
        
        vals = corr_df[col_name].dropna()
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=40, marker_color=colors[i % len(colors)], 
                        opacity=0.75, name=attr, showlegend=False),
            row=row, col=col
        )
        fig.update_xaxes(title_text='Correlation', row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title_text=title,
        template='plotly_white'
    )
    return fig


def plotly_scatter_with_regression(
    expression_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_id_col: str,
    attribute: str,
    gene: str,
    title: str = "Gene Expression vs Attribute"
) -> go.Figure:
    """Create scatter plot with regression line for a gene vs attribute."""
    sample_ids = list(expression_df.columns)
    gene_data = expression_df.loc[gene, :].values.astype(float)
    
    attr_data = meta_df.set_index(sample_id_col).reindex(sample_ids)[attribute]
    if not pd.api.types.is_numeric_dtype(attr_data):
        attr_data = pd.Categorical(attr_data).codes.astype(float)
        attr_data = np.where(attr_data == -1, np.nan, attr_data)
    else:
        attr_data = attr_data.values.astype(float)
    
    # Remove NaN
    mask = ~(np.isnan(gene_data) | np.isnan(attr_data))
    x = attr_data[mask]
    y = gene_data[mask]
    
    # Compute correlation
    if len(x) >= 2:
        corr, pval = pearsonr(x, y)
    else:
        corr, pval = np.nan, np.nan
    
    fig = go.Figure()
    
    # Scatter
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.6),
        name='Samples'
    ))
    
    # Regression line
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Fit (r={corr:.3f}, p={pval:.2e})'
        ))
    
    fig.update_layout(
        title=f"{title}<br>{gene} vs {attribute}",
        xaxis_title=attribute,
        yaxis_title='Expression (log2)',
        height=400,
        template='plotly_white'
    )
    return fig


def plotly_variance_explained(pca_model: PCA, n_pcs: int = 10) -> go.Figure:
    """Create a bar chart of variance explained by each PC."""
    n_pcs = min(n_pcs, len(pca_model.explained_variance_ratio_))
    ev = pca_model.explained_variance_ratio_[:n_pcs] * 100
    cumulative = np.cumsum(ev)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(n_pcs)], y=ev,
               name='Variance Explained', marker_color='steelblue'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=[f'PC{i+1}' for i in range(n_pcs)], y=cumulative,
                   mode='lines+markers', name='Cumulative',
                   line=dict(color='red', width=2)),
        secondary_y=True
    )
    
    fig.update_layout(
        title='PCA Variance Explained',
        height=350,
        template='plotly_white'
    )
    fig.update_xaxes(title_text='Principal Component')
    fig.update_yaxes(title_text='Variance Explained (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative (%)', secondary_y=True)
    
    return fig


def plotly_pca_grid(
    scores: np.ndarray,
    sample_ids: List[str],
    meta_df: pd.DataFrame,
    sample_id_col: str,
    covariates: List[str]
) -> go.Figure:
    """Create 2x2 grid of PCA plots colored by different covariates."""
    n_covs = min(4, len(covariates))
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=covariates[:n_covs],
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )
    
    aligned = meta_df.set_index(sample_id_col)
    
    for i, cov in enumerate(covariates[:n_covs]):
        row = i // 2 + 1
        col = i % 2 + 1
        
        cov_data = aligned.reindex(sample_ids)[cov]
        if not pd.api.types.is_numeric_dtype(cov_data):
            cov_vals = pd.Categorical(cov_data).codes.astype(float)
        else:
            cov_vals = cov_data.values.astype(float)
        
        fig.add_trace(
            go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=cov_vals,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=cov[:10],
                        x=1.0 if col == 2 else 0.45,
                        y=0.75 if row == 1 else 0.25,
                        len=0.4
                    )
                ),
                text=[f"Sample: {s}<br>{cov}: {v}" for s, v in zip(sample_ids, cov_data)],
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text='PC1', row=row, col=col)
        fig.update_yaxes(title_text='PC2', row=row, col=col)
    
    fig.update_layout(
        height=700,
        title_text='PCA Colored by Sample Attributes',
        template='plotly_white'
    )
    return fig


# ---------- Streamlit App ----------
st.set_page_config(page_title="Expression Cleaning (GTEx-style)", layout="wide")

st.title("Expression Data Preprocessing & Cleaning")
st.caption(
    "Load a gene-by-sample expression matrix, filter & normalize, visualize PCA, detect outliers, "
    "and optionally regress out technical confounders."
)

with st.sidebar:
    st.header("Input")

    mode = st.radio("Expression input", ["Local path", "Upload"], horizontal=True)

    expr_path = ""
    uploaded_expr = None
    if mode == "Local path":
        expr_path = st.text_input(
            "Expression file path (TSV/CSV)",
            value="",
            placeholder="/path/to/expression.tsv",
        )
    else:
        uploaded_expr = st.file_uploader("Upload expression file", type=["tsv", "csv", "txt", "gct"])

    default_sep = "\t" if (expr_path.endswith(".tsv") or expr_path.endswith(".txt") or expr_path.endswith(".gct")) else ","
    sep_options = ["\t", ",", ";"]
    sep_labels = {"\t": "Tab (\\t)", ",": "Comma (,)", ";": "Semicolon (;)"}
    sep = st.selectbox(
        "Separator",
        options=sep_options,
        index=sep_options.index(default_sep if default_sep in sep_options else "\t"),
        format_func=lambda x: sep_labels.get(x, x)
    )
    skip_rows = st.number_input("Skip rows at top (e.g., GTEx TPM has 2)", min_value=0, value=0, step=1)

    st.subheader("Gene columns")
    st.caption("Columns to keep as gene annotations (non-numeric). Others treated as sample columns.")
    gene_cols_text = st.text_input("Comma-separated gene columns", value="Name,Description,id")
    gene_cols = tuple([c.strip() for c in gene_cols_text.split(",") if c.strip()])

    st.subheader("Metadata (optional)")
    meta_mode = st.radio("Metadata input", ["None", "Local path", "Upload"], horizontal=True, index=0)
    meta_path = ""
    uploaded_meta = None
    if meta_mode == "Local path":
        meta_path = st.text_input("Metadata file path (TSV/CSV)", value="", placeholder="/path/to/metadata.tsv")
    elif meta_mode == "Upload":
        uploaded_meta = st.file_uploader("Upload metadata file", type=["tsv", "csv", "txt"])

    meta_sep_options = ["\t", ",", ";"]
    meta_sep_labels = {"\t": "Tab (\\t)", ",": "Comma (,)", ";": "Semicolon (;)"}
    meta_sep = st.selectbox(
        "Metadata separator",
        options=meta_sep_options,
        index=0,
        format_func=lambda x: meta_sep_labels.get(x, x)
    )
    sample_id_col = st.text_input("Sample ID column in metadata", value="SAMPID")
    
    st.subheader("Subject Phenotypes (optional)")
    st.caption("Load additional subject-level data (e.g., age, sex, death circumstances). Will be merged with sample attributes.")
    pheno_mode = st.radio("Subject phenotypes input", ["None", "Local path", "Upload"], horizontal=True, index=0, key="pheno_mode")
    pheno_path = ""
    uploaded_pheno = None
    if pheno_mode == "Local path":
        pheno_path = st.text_input("Subject phenotypes file path", value="", placeholder="/path/to/phenotypes.tsv")
    elif pheno_mode == "Upload":
        uploaded_pheno = st.file_uploader("Upload subject phenotypes file", type=["tsv", "csv", "txt"], key="pheno_upload")
    
    pheno_sep = st.selectbox(
        "Phenotypes separator",
        options=meta_sep_options,
        index=0,
        format_func=lambda x: meta_sep_labels.get(x, x),
        key="pheno_sep"
    )
    subject_id_col = st.text_input("Subject ID column in phenotypes", value="SUBJID")
    st.caption("For GTEx: Subject ID is extracted from Sample ID (e.g., GTEX-1117F from GTEX-1117F-0226-SM-5GZZ7)")

    st.divider()
    st.header("Cleaning parameters")

    min_tpm = st.number_input("Min expression (raw scale)", min_value=0.0, value=1.0, step=0.1)
    min_samples_fraction = st.slider("Min fraction of samples expressing gene", 0.0, 1.0, 0.2, 0.05)
    min_samples_floor = st.number_input("Min samples floor", min_value=1, value=10, step=1)
    var_quantile = st.slider("Variance quantile cutoff (log2 scale)", 0.0, 0.9, 0.2, 0.05)

    do_quantile_norm = st.checkbox("Quantile normalize (after filtering)", value=True)
    qn_fast = st.checkbox("Use fast quantile normalization", value=True, help="Faster; approximate tie handling.")
    st.caption("Tip: disable quantile normalization first if you want speed.")

    st.divider()
    st.header("PCA / Outliers")

    n_pcs = st.slider("PCA components", 2, 50, 10)
    outlier_methods = st.multiselect(
        "Outlier methods",
        ["Mahalanobis (PCA)", "Hierarchical clusters (Ward)", "IQR on PCs"],
        default=["Mahalanobis (PCA)", "IQR on PCs"],
    )

    chi2_q = st.slider("Mahalanobis chi2 quantile", 0.80, 0.999, 0.95, 0.01)
    mahal_df = st.slider("Mahalanobis degrees of freedom (PC dims)", 2, 20, 10)

    st.markdown("**Hierarchical Clustering Options**")
    linkage_method = st.selectbox(
        "Linkage method",
        options=["ward", "complete", "average", "single", "centroid", "median", "weighted"],
        index=0,
        help="Ward: minimizes variance; Complete: max distance; Average: mean distance; Single: min distance"
    )
    cut_frac = st.slider("Hierarchical cut fraction of max height", 0.1, 0.99, 0.7, 0.05)
    min_cluster_frac = st.slider("Min cluster fraction (smaller => outlier)", 0.0, 0.2, 0.05, 0.01)
    show_dendrogram = st.checkbox("Show dendrogram", value=True)

    iqr_pcs = st.slider("IQR method: number of PCs", 1, 20, 5)
    iqr_k = st.slider("IQR multiplier", 0.5, 5.0, 1.5, 0.1)

    st.divider()
    st.header("Confounder regression")
    do_regress = st.checkbox("Regress out confounders (requires metadata)", value=False)


# ---------- Load data ----------
expr_real_path = None
if uploaded_expr is not None:
    expr_real_path = _save_upload_to_temp(uploaded_expr)
elif expr_path and os.path.exists(expr_path):
    expr_real_path = expr_path

meta_real_path = None
if uploaded_meta is not None:
    meta_real_path = _save_upload_to_temp(uploaded_meta)
elif meta_path and os.path.exists(meta_path):
    meta_real_path = meta_path

pheno_real_path = None
if uploaded_pheno is not None:
    pheno_real_path = _save_upload_to_temp(uploaded_pheno)
elif pheno_path and os.path.exists(pheno_path):
    pheno_real_path = pheno_path

tabs = st.tabs(["1) Load", "2) Transform & Filter", "3) Normalize", "4) PCA", "5) Outliers", "6) Confounders & Export"])

with tabs[0]:
    st.subheader("Step 1 — Load & preview")

    if not expr_real_path:
        st.info("Provide an expression file path or upload a file to begin.")
        st.stop()

    preview, cols, numeric_candidates = read_expression(
        expr_real_path,
        sep=sep,
        skip_rows=int(skip_rows),
        gene_cols=gene_cols,
    )
    st.write("Preview (first rows):")
    st.dataframe(preview, use_container_width=True)

    st.write(f"Detected **{len(cols)}** columns.")
    if gene_cols:
        st.write("Gene columns (requested):", list(gene_cols))
    st.write("Numeric sample candidates (from preview):", numeric_candidates[:40], "..." if len(numeric_candidates) > 40 else "")

    # Let user choose sample columns (default numeric candidates)
    sample_cols = st.multiselect(
        "Select sample columns",
        options=[c for c in cols if c not in gene_cols],
        default=numeric_candidates,
        help="For GTEx-style files, this should be all sample columns (numeric).",
    )
    if len(sample_cols) == 0:
        st.warning("Select at least one sample column.")
        st.stop()

    cast_float = st.checkbox("Cast sample columns to float", value=True)

    if st.button("Load expression matrix into memory", type="primary"):
        with st.spinner("Loading expression matrix..."):
            expr = load_expression_to_memory(
                expr_real_path,
                sep=sep,
                skip_rows=int(skip_rows),
                gene_cols=gene_cols,
                sample_cols=tuple(sample_cols),
                cast_float=cast_float,
            )
        st.session_state["expr"] = expr
        st.success(f"Loaded matrix: genes={expr.X.shape[0]:,}, samples={expr.X.shape[1]:,}")

    # Load metadata if provided
    if meta_real_path:
        try:
            meta_df = pd.read_csv(meta_real_path, sep=meta_sep)
            st.session_state["meta"] = meta_df
            st.write("**Sample Attributes** preview:")
            st.dataframe(meta_df.head(8), use_container_width=True)
            st.write(f"Sample attributes rows: {meta_df.shape[0]:,}, columns: {meta_df.shape[1]:,}")
        except Exception as e:
            st.error(f"Failed reading metadata: {e}")
    
    # Load subject phenotypes if provided
    if pheno_real_path:
        try:
            pheno_df = pd.read_csv(pheno_real_path, sep=pheno_sep)
            st.session_state["pheno"] = pheno_df
            st.write("**Subject Phenotypes** preview:")
            st.dataframe(pheno_df.head(8), use_container_width=True)
            st.write(f"Subject phenotypes rows: {pheno_df.shape[0]:,}, columns: {pheno_df.shape[1]:,}")
            
            # Merge with sample attributes if both are loaded
            if "meta" in st.session_state and sample_id_col in st.session_state["meta"].columns:
                meta_df = st.session_state["meta"]
                
                # Extract subject ID from sample ID (GTEx format: GTEX-XXXXX-... -> GTEX-XXXXX)
                def extract_subject_id(sample_id: str) -> str:
                    """Extract subject ID from GTEx-style sample ID."""
                    parts = str(sample_id).split("-")
                    if len(parts) >= 2:
                        return "-".join(parts[:2])  # GTEX-XXXXX
                    return sample_id
                
                meta_df["_extracted_subjid"] = meta_df[sample_id_col].apply(extract_subject_id)
                
                # Merge phenotypes
                if subject_id_col in pheno_df.columns:
                    # Avoid duplicate columns
                    pheno_cols_to_add = [c for c in pheno_df.columns if c not in meta_df.columns and c != subject_id_col]
                    merged_df = meta_df.merge(
                        pheno_df[[subject_id_col] + pheno_cols_to_add],
                        left_on="_extracted_subjid",
                        right_on=subject_id_col,
                        how="left"
                    )
                    # Drop helper columns
                    merged_df = merged_df.drop(columns=["_extracted_subjid"], errors="ignore")
                    if subject_id_col in merged_df.columns and subject_id_col != sample_id_col:
                        pass  # Keep SUBJID column for reference
                    
                    st.session_state["meta"] = merged_df
                    st.session_state["meta_merged"] = True
                    
                    st.success(f"✅ Merged sample attributes with subject phenotypes!")
                    st.write(f"**Combined metadata:** {merged_df.shape[0]:,} rows, {merged_df.shape[1]:,} columns")
                    st.write("New columns from phenotypes:", pheno_cols_to_add)
                    
                    # Show merged preview
                    with st.expander("Preview merged metadata"):
                        st.dataframe(merged_df.head(8), use_container_width=True)
                else:
                    st.warning(f"Subject ID column '{subject_id_col}' not found in phenotypes file.")
        except Exception as e:
            st.error(f"Failed reading subject phenotypes: {e}")

with tabs[1]:
    st.subheader("Step 2 — Log transform + filtering")

    if "expr" not in st.session_state:
        st.info("Go to **1) Load** and load the expression matrix first.")
        st.stop()

    expr: ExpressionData = st.session_state["expr"]
    X_raw = expr.X

    n_samples = X_raw.shape[1]
    n_genes = X_raw.shape[0]
    min_samples = max(int(min_samples_floor), int(np.floor(min_samples_fraction * n_samples)))
    
    # Summary metrics
    st.markdown("### Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Genes", f"{n_genes:,}")
    with col2:
        st.metric("Total Samples", f"{n_samples:,}")
    with col3:
        st.metric("Matrix Size", f"{n_genes * n_samples:,} values")
    
    st.write(f"**Current filter settings:** gene expressed ≥ **{min_tpm}** in at least **{min_samples} / {n_samples}** samples, variance ≥ {var_quantile:.0%} quantile.")

    # Distributions before
    vals_raw = sample_values_for_hist(X_raw, n=50000)
    st.plotly_chart(plotly_hist(vals_raw, "Raw values distribution (sampled)"), use_container_width=True)
    
    # ========== THRESHOLD EXPLORATION SECTION ==========
    st.markdown("---")
    st.markdown("### 🔍 Threshold Exploration Tools")
    st.caption("Use these tools to find optimal thresholds before applying filters.")
    
    with st.expander("📊 Expression Threshold Explorer", expanded=True):
        st.markdown("See how many genes pass the expression filter at different thresholds.")
        
        # Compute stats for different thresholds
        tpm_range = st.slider(
            "TPM range to explore",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 5.0),
            step=0.5,
            key="tpm_range_slider"
        )
        tpm_values = np.linspace(tpm_range[0], tpm_range[1], 20)
        
        if st.button("Compute expression threshold stats", key="btn_expr_stats"):
            with st.spinner("Computing..."):
                expr_stats = compute_expression_filter_stats(X_raw, tpm_values, min_samples)
            st.session_state["expr_filter_stats"] = expr_stats
        
        if "expr_filter_stats" in st.session_state:
            expr_stats = st.session_state["expr_filter_stats"]
            
            # Plot
            fig_expr = plotly_expression_threshold_explorer(expr_stats, float(min_tpm))
            st.plotly_chart(fig_expr, use_container_width=True)
            
            # Table
            st.dataframe(expr_stats, use_container_width=True, hide_index=True)
        
        # Distribution of samples expressing each gene
        st.markdown("##### Samples Expressing Each Gene")
        fig_samples = plotly_samples_expressing_gene(X_raw, float(min_tpm))
        st.plotly_chart(fig_samples, use_container_width=True)
    
    with st.expander("📈 Variance Threshold Explorer", expanded=True):
        st.markdown("See how many genes pass the variance filter at different quantile cutoffs.")
        
        if st.button("Compute variance statistics", key="btn_var_stats"):
            with st.spinner("Computing variances..."):
                var_quantiles = np.linspace(0, 0.5, 21)
                var_stats, variances = compute_variance_filter_stats(X_raw, var_quantiles)
            st.session_state["var_filter_stats"] = var_stats
            st.session_state["gene_variances"] = variances
        
        if "var_filter_stats" in st.session_state:
            var_stats = st.session_state["var_filter_stats"]
            variances = st.session_state["gene_variances"]
            
            # Variance distribution with quantile markers
            st.markdown("##### Variance Distribution")
            fig_var_dist = plotly_variance_distribution(variances, float(var_quantile))
            st.plotly_chart(fig_var_dist, use_container_width=True)
            
            # Genes kept at different quantiles
            st.markdown("##### Genes Kept vs Variance Quantile")
            fig_var_explore = plotly_variance_quantile_explorer(var_stats, float(var_quantile))
            st.plotly_chart(fig_var_explore, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Variance", f"{variances.min():.4f}")
            with col2:
                st.metric("Median Variance", f"{np.median(variances):.4f}")
            with col3:
                st.metric("Mean Variance", f"{variances.mean():.4f}")
            with col4:
                st.metric("Max Variance", f"{variances.max():.4f}")
            
            # Table
            st.dataframe(var_stats, use_container_width=True, hide_index=True)
    
    with st.expander("🎯 Mean Expression vs Variance", expanded=False):
        st.markdown("Scatter plot showing relationship between mean expression and variance for each gene.")
        fig_mean_var = plotly_expression_vs_variance(X_raw)
        st.plotly_chart(fig_mean_var, use_container_width=True)
    
    with st.expander("🔬 Combined Filter Preview", expanded=True):
        st.markdown("Preview which genes will be kept/removed with current settings.")
        
        # Interactive threshold adjustment for preview
        col1, col2 = st.columns(2)
        with col1:
            preview_min_tpm = st.number_input(
                "Preview: Min TPM", 
                min_value=0.0, 
                value=float(min_tpm), 
                step=0.1,
                key="preview_min_tpm"
            )
        with col2:
            preview_var_q = st.slider(
                "Preview: Variance quantile",
                min_value=0.0,
                max_value=0.5,
                value=float(var_quantile),
                step=0.05,
                key="preview_var_q"
            )
        
        preview_min_samples = max(int(min_samples_floor), int(np.floor(min_samples_fraction * n_samples)))
        
        # Compute preview stats
        Xv = X_raw.to_numpy(dtype=float, copy=False)
        Xlog = np.log2(Xv + 1.0)
        samples_expressing = (Xv >= preview_min_tpm).sum(axis=1)
        variances_preview = np.var(Xlog, axis=1, ddof=1)
        var_threshold_preview = np.quantile(variances_preview, preview_var_q)
        
        pass_expr = samples_expressing >= preview_min_samples
        pass_var = variances_preview >= var_threshold_preview
        pass_both = pass_expr & pass_var
        
        # Preview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pass Expression", f"{pass_expr.sum():,}", delta=f"{pass_expr.sum() - n_genes:,}")
        with col2:
            st.metric("Pass Variance", f"{pass_var.sum():,}", delta=f"{pass_var.sum() - n_genes:,}")
        with col3:
            st.metric("Pass Both (Kept)", f"{pass_both.sum():,}", delta=f"{pass_both.sum() - n_genes:,}")
        with col4:
            st.metric("Removal %", f"{(1 - pass_both.sum()/n_genes)*100:.1f}%")
        
        # Combined scatter plot
        fig_combined = plotly_combined_filter_preview(
            X_raw, preview_min_tpm, preview_min_samples, preview_var_q
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    
    # ========== APPLY FILTERS ==========
    st.markdown("---")
    st.markdown("### Apply Filters")
    
    if st.button("Apply log2(x+1) + filters", type="primary"):
        with st.spinner("Filtering genes..."):
            keep_expr, keep_var = filter_low_expression_and_variance(
                X_raw=X_raw,
                min_tpm=float(min_tpm),
                min_samples=int(min_samples),
                var_quantile=float(var_quantile),
            )
        keep = keep_expr & keep_var
        X_filt = X_raw.loc[keep].copy()
        genes_filt = expr.genes.loc[keep].copy() if not expr.genes.empty else pd.DataFrame(index=X_filt.index)
        st.session_state["X_filtered_raw"] = X_filt
        st.session_state["genes_filtered"] = genes_filt
        st.session_state["keep_mask"] = keep
        st.session_state["keep_expr_mask"] = keep_expr
        st.session_state["keep_var_mask"] = keep_var
        st.success(f"Kept genes: {int(keep.sum()):,} / {keep.size:,}")

    if "X_filtered_raw" in st.session_state:
        X_filt = st.session_state["X_filtered_raw"]
        keep_expr = st.session_state.get("keep_expr_mask", None)
        keep_var = st.session_state.get("keep_var_mask", None)
        
        st.markdown("### Filtering Results")
        
        # Summary of filtering
        if keep_expr is not None and keep_var is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Passed Expression Filter", f"{keep_expr.sum():,}")
            with col2:
                st.metric("Passed Variance Filter", f"{keep_var.sum():,}")
            with col3:
                st.metric("Final Kept (Both)", f"{X_filt.shape[0]:,}")
        
        vals_log = sample_values_for_hist(log2p1(X_filt), n=50000)
        st.plotly_chart(plotly_hist(vals_log, "log2(x+1) distribution after filtering (sampled)", color="green"), use_container_width=True)
        st.write("Filtered shape:", X_filt.shape)

with tabs[2]:
    st.subheader("Step 3 — Quantile normalization (optional)")

    if "X_filtered_raw" not in st.session_state:
        st.info("Run **Step 2** first (filter genes).")
        st.stop()

    X_filt = st.session_state["X_filtered_raw"]
    X_log = log2p1(X_filt)

    st.write("Input to this step: filtered, log2(x+1) matrix (genes x samples).")
    st.write("Shape:", X_log.shape)

    # Show before normalization - multiple views
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_hist(sample_values_for_hist(X_log), "Before normalization (log2) — sampled"), use_container_width=True)
    with col2:
        st.plotly_chart(plotly_sample_boxplots(X_log, n_samples=15, title="Sample distributions (before)"), use_container_width=True)

    if do_quantile_norm:
        st.markdown("""
        **Quantile Normalization Method** (from notebook):
        1. Transpose matrix to samples × genes
        2. Sort values in each column (gene)
        3. Calculate row means of sorted matrix
        4. Replace each value with the mean of its rank position
        5. Transpose back to genes × samples
        """)
        
        if st.button("Run quantile normalization", type="primary"):
            with st.spinner("Quantile normalizing (using notebook method)..."):
                # Use the exact method from the notebook
                X_qn_df = quantile_normalize_notebook(X_log)

            st.session_state["X_qn"] = X_qn_df
            st.session_state["X_log_before_qn"] = X_log
            st.success("Quantile normalization complete.")

        if "X_qn" in st.session_state:
            X_qn_df = st.session_state["X_qn"]
            X_log_before = st.session_state.get("X_log_before_qn", X_log)
            
            st.markdown("### Before vs After Comparison")
            
            # Side-by-side histograms
            fig_compare = plotly_distribution_comparison(
                sample_values_for_hist(X_log_before),
                sample_values_for_hist(X_qn_df),
                before_title="Before Normalization",
                after_title="After Normalization",
                main_title="Distribution Comparison"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Boxplot comparison for first few genes
            st.markdown("### Gene-level Boxplot Comparison")
            fig_boxplot = plotly_boxplot_comparison(
                X_log_before, X_qn_df, n_genes=5,
                title="Boxplot: First 5 Genes Before vs After Quantile Normalization"
            )
            st.plotly_chart(fig_boxplot, use_container_width=True)
            
            # Sample distributions after
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plotly_hist(sample_values_for_hist(X_qn_df), "After normalization — sampled", color="green"), use_container_width=True)
            with col2:
                st.plotly_chart(plotly_sample_boxplots(X_qn_df, n_samples=15, title="Sample distributions (after)"), use_container_width=True)
            
            st.write("Normalized matrix shape:", X_qn_df.shape)
    else:
        st.info("Quantile normalization disabled. PCA will use the filtered log2(x+1) matrix.")

with tabs[3]:
    st.subheader("Step 4 — PCA on samples")

    if "X_filtered_raw" not in st.session_state:
        st.info("Run **Step 2** first.")
        st.stop()

    X_filt = st.session_state["X_filtered_raw"]
    X_log = log2p1(X_filt)

    X_in = st.session_state.get("X_qn", X_log)
    st.write("PCA input:", "Quantile-normalized" if "X_qn" in st.session_state else "Filtered log2(x+1)")
    st.write("Shape:", X_in.shape)

    scores, pca_model = pca_on_samples(X_in, n_components=int(n_pcs))
    sample_ids = list(X_in.columns)

    # Variance explained chart
    st.markdown("### Variance Explained by Principal Components")
    fig_variance = plotly_variance_explained(pca_model, n_pcs=min(10, n_pcs))
    st.plotly_chart(fig_variance, use_container_width=True)
    
    # Table of variance
    ev = pca_model.explained_variance_ratio_[: min(10, len(pca_model.explained_variance_ratio_))]
    ev_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(ev))],
        "Variance Explained (%)": (ev * 100).round(2),
        "Cumulative (%)": (np.cumsum(ev) * 100).round(2)
    })
    st.dataframe(ev_df, use_container_width=True, hide_index=True)

    # Color by metadata column if available
    meta_df = st.session_state.get("meta", None)
    color_series = None
    
    st.markdown("### PCA Scatter Plot")
    if meta_df is not None and sample_id_col in meta_df.columns:
        aligned = meta_df.set_index(sample_id_col)
        common = [s for s in sample_ids if s in aligned.index]
        st.write(f"Metadata alignment: {len(common)}/{len(sample_ids)} samples found in metadata.")
        color_col = st.selectbox(
            "Color PCA by (metadata column)",
            options=["(none)"] + [c for c in meta_df.columns if c != sample_id_col],
            index=0,
        )
        if color_col != "(none)":
            # align order with sample_ids, keep NaNs
            color_series = aligned.reindex(sample_ids)[color_col]
            color_series.name = color_col

    chart = plotly_pca_scatter(scores, sample_ids, color_series, "PCA: samples in PC1–PC2")
    st.plotly_chart(chart, use_container_width=True)
    
    # PCA grid with covariates (if metadata available)
    if meta_df is not None and sample_id_col in meta_df.columns:
        st.markdown("### PCA Colored by Multiple Covariates")
        default_covs = [c for c in [ "SMRIN", "SMTSISCH"] if c in meta_df.columns]
        pca_covariates = st.multiselect(
            "Select covariates for PCA grid",
            options=[c for c in meta_df.columns if c != sample_id_col],
            default=default_covs[:4]
        )
        if len(pca_covariates) > 0:
            fig_grid = plotly_pca_grid(scores, sample_ids, meta_df, sample_id_col, pca_covariates)
            st.plotly_chart(fig_grid, use_container_width=True)

    st.session_state["pca_scores"] = scores
    st.session_state["pca_input_sample_ids"] = sample_ids
    st.session_state["pca_model"] = pca_model
    
    # Gene-Attribute Correlation Analysis
    if meta_df is not None and sample_id_col in meta_df.columns:
        st.markdown("---")
        st.markdown("### Gene-Attribute Correlation Analysis")
        st.write("Compute Pearson correlations between gene expression and sample attributes (phenotypes).")
        
        numeric_attrs = [c for c in meta_df.columns if c != sample_id_col]
        corr_attributes = st.multiselect(
            "Select attributes for correlation analysis",
            options=numeric_attrs,
            default=[c for c in ["SMRIN", "SMTSISCH"] if c in numeric_attrs][:4]
        )
        
        max_genes_corr = st.slider("Max genes to analyze (for speed)", 100, 10000, 2000, 100)
        
        if corr_attributes and st.button("Compute Gene-Attribute Correlations", type="secondary"):
            with st.spinner(f"Computing correlations for {max_genes_corr} genes..."):
                corr_df, corr_summary = compute_gene_attribute_correlations(
                    X_in, meta_df, sample_id_col, corr_attributes, max_genes=max_genes_corr
                )
            st.session_state["corr_df"] = corr_df
            st.session_state["corr_summary"] = corr_summary
            st.session_state["corr_attributes"] = corr_attributes
            st.success(f"Correlation analysis complete for {len(corr_df)} genes!")
        
        if "corr_df" in st.session_state:
            corr_df = st.session_state["corr_df"]
            corr_summary = st.session_state["corr_summary"]
            corr_attributes = st.session_state["corr_attributes"]
            
            # Summary statistics
            st.markdown("#### Summary Statistics")
            summary_data = []
            for attr, stats in corr_summary.items():
                summary_data.append({
                    "Attribute": attr,
                    "N Genes": stats['n_genes'],
                    "Mean Corr": f"{stats['mean']:.4f}",
                    "Median Corr": f"{stats['median']:.4f}",
                    "Std": f"{stats['std']:.4f}",
                    "Range": f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Correlation distribution histograms
            st.markdown("#### Distribution of Correlations")
            fig_corr_dist = plotly_correlation_distribution(corr_df, corr_attributes)
            st.plotly_chart(fig_corr_dist, use_container_width=True)
            
            # Heatmap of top correlated genes
            st.markdown("#### Top Correlated Genes Heatmap")
            n_top = st.slider("Number of top genes to show", 10, 50, 20)
            fig_heatmap = plotly_correlation_heatmap(corr_df, corr_attributes, n_top_genes=n_top)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Scatter plots for top genes
            st.markdown("#### Scatter Plots for Top Correlated Genes")
            attr_for_scatter = st.selectbox("Select attribute for scatter plot", corr_attributes)
            corr_col = f"{attr_for_scatter}_correlation"
            if corr_col in corr_df.columns:
                top_gene = corr_df.loc[corr_df[corr_col].abs().idxmax(), 'gene']
                fig_scatter = plotly_scatter_with_regression(
                    X_in, meta_df, sample_id_col, attr_for_scatter, top_gene,
                    title=f"Top Correlated Gene vs {attr_for_scatter}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

with tabs[4]:
    st.subheader("Step 5 — Outlier detection")

    if "pca_scores" not in st.session_state:
        st.info("Run **Step 4** first (PCA).")
        st.stop()

    scores = st.session_state["pca_scores"]
    sample_ids = st.session_state["pca_input_sample_ids"]

    masks = []
    labels = []

    hierarchical_linkage_matrix = None
    hierarchical_cut_height = None
    
    if "Mahalanobis (PCA)" in outlier_methods:
        df_use = int(min(mahal_df, scores.shape[1]))
        m = detect_outliers_mahalanobis(scores, df=df_use, chi2_quantile=float(chi2_q))
        masks.append(m)
        labels.append(f"Mahalanobis(df={df_use}, q={chi2_q:.3f})")
        
        # Compute Mahalanobis distances and related values
        center = np.mean(scores[:, :df_use], axis=0)
        cov = np.cov(scores[:, :df_use].T)
        inv_cov = np.linalg.pinv(cov)
        mahal_distances = np.array([mahalanobis(s[:df_use], center, inv_cov) for s in scores])
        threshold = float(chi2.ppf(float(chi2_q), df=df_use))
        
        # Summary statistics
        st.markdown("#### Mahalanobis Distance Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Distance", f"{mahal_distances.mean():.3f}")
        with col2:
            st.metric("Median Distance", f"{np.median(mahal_distances):.3f}")
        with col3:
            st.metric("Threshold", f"{threshold:.3f}")
        with col4:
            st.metric("Outliers", f"{m.sum()} / {len(m)}")
        
        # Main histogram + scatter plot
        st.markdown("##### Distance Distribution & PCA Scatter")
        fig_mahal = plotly_outlier_detection(scores, sample_ids, mahal_distances, threshold, m)
        st.plotly_chart(fig_mahal, use_container_width=True)
        
        # Chi-squared Q-Q Plot
        st.markdown("##### Chi-Squared Q-Q Plot")
        st.caption("If points follow the diagonal, Mahalanobis distances are approximately chi-squared distributed. Deviations suggest non-normality or true outliers.")
        fig_qq = plotly_mahal_qq_plot(mahal_distances, df=df_use)
        st.plotly_chart(fig_qq, use_container_width=True)
        
        # PCA with confidence ellipse (using first 2 PCs)
        st.markdown("##### PCA with Mahalanobis Confidence Ellipse")
        st.caption(f"Ellipse shows the {chi2_q*100:.0f}% confidence boundary based on Mahalanobis distance in PC1-PC2 space.")
        center_2d = np.mean(scores[:, :2], axis=0)
        cov_2d = np.cov(scores[:, :2].T)
        fig_ellipse = plotly_mahal_pca_ellipse(
            scores, sample_ids, mahal_distances, m,
            center_2d, cov_2d, float(chi2_q), df=2
        )
        st.plotly_chart(fig_ellipse, use_container_width=True)
        
        # PC contributions to distance
        st.markdown("##### Per-PC Contribution Analysis")
        st.caption("Shows which principal components contribute most to outlier status. High bars for outliers indicate PCs driving their extreme distance.")
        fig_contrib = plotly_mahal_pc_contributions(
            scores, sample_ids, center, inv_cov, m,
            n_pcs=min(5, df_use)
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Distance vs individual PCs
        st.markdown("##### Mahalanobis Distance vs Individual PCs")
        st.caption("Scatter plots showing relationship between Mahalanobis distance and each PC. Red points are outliers.")
        fig_vs_pcs = plotly_mahal_distance_vs_pcs(
            scores, sample_ids, mahal_distances, m,
            n_pcs=min(4, df_use)
        )
        st.plotly_chart(fig_vs_pcs, use_container_width=True)
        
        # Ranked samples by distance
        st.markdown("##### Top Samples by Mahalanobis Distance")
        n_show_ranked = st.slider("Number of samples to show", 10, 50, 25, key="mahal_ranked_slider")
        fig_ranked = plotly_mahal_ranked_samples(
            sample_ids, mahal_distances, threshold, n_show=n_show_ranked
        )
        st.plotly_chart(fig_ranked, use_container_width=True)

    if "Hierarchical clusters (Ward)" in outlier_methods:
        m, hierarchical_linkage_matrix, hierarchical_cut_height = detect_outliers_hierarchical(
            scores[:, : min(10, scores.shape[1])], 
            cut_fraction_of_max=float(cut_frac), 
            min_cluster_fraction=float(min_cluster_frac),
            method=linkage_method
        )
        masks.append(m)
        labels.append(f"Hierarchical({linkage_method}, cut={cut_frac:.2f}*max, min_cluster={min_cluster_frac:.2f})")
        
        # Show dendrogram
        if show_dendrogram and hierarchical_linkage_matrix is not None:
            st.markdown(f"#### Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)")
            fig_dend = plotly_dendrogram(hierarchical_linkage_matrix, sample_ids, 
                                         title=f"Dendrogram ({linkage_method.capitalize()} Linkage)")
            # Add horizontal line at cut height
            fig_dend.add_hline(y=hierarchical_cut_height, line_dash="dash", line_color="red",
                              annotation_text=f"Cut height: {hierarchical_cut_height:.2f}")
            st.plotly_chart(fig_dend, use_container_width=True)
            
            # Show cluster information
            clusters = fcluster(hierarchical_linkage_matrix, hierarchical_cut_height, criterion="distance")
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            st.write(f"Number of clusters at cut height: **{len(cluster_counts)}**")
            st.write("Cluster sizes:", dict(cluster_counts))

    if "IQR on PCs" in outlier_methods:
        m = detect_outliers_iqr(scores, n_pcs=int(iqr_pcs), iqr_k=float(iqr_k))
        masks.append(m)
        labels.append(f"IQR(n_pcs={iqr_pcs}, k={iqr_k:.1f})")
        
        # Show IQR analysis
        st.markdown("#### IQR Outlier Detection")
        iqr_data = []
        for k in range(min(int(iqr_pcs), scores.shape[1])):
            v = scores[:, k]
            q1, q3 = np.quantile(v, [0.25, 0.75])
            iqr = q3 - q1
            lo = q1 - float(iqr_k) * iqr
            hi = q3 + float(iqr_k) * iqr
            n_out = int(((v < lo) | (v > hi)).sum())
            iqr_data.append({"PC": f"PC{k+1}", "Q1": f"{q1:.3f}", "Q3": f"{q3:.3f}", 
                            "IQR": f"{iqr:.3f}", "Lower": f"{lo:.3f}", "Upper": f"{hi:.3f}",
                            "Outliers": n_out})
        st.dataframe(pd.DataFrame(iqr_data), use_container_width=True, hide_index=True)

    if not masks:
        st.info("Select at least one outlier method in the sidebar.")
        st.stop()

    out_mask = np.logical_or.reduce(masks)
    outliers = [s for s, flag in zip(sample_ids, out_mask) if flag]

    st.markdown("---")
    st.markdown("### Summary")
    st.write("Methods used:", labels)
    st.write(f"Outliers detected: **{len(outliers)} / {len(sample_ids)}**")

    st.dataframe(pd.DataFrame({"outlier_sample": outliers}), use_container_width=True, hide_index=True)

    # show PCA with outliers highlighted (binary color)
    st.markdown("### PCA with Outliers Highlighted")
    outlier_series = pd.Series(out_mask.astype(int), index=sample_ids, name="outlier_flag")
    chart = plotly_pca_scatter(scores, sample_ids, outlier_series, "PCA with outliers highlighted (1=outlier)")
    st.plotly_chart(chart, use_container_width=True)

    if st.button("Remove outliers and recompute PCA", type="primary"):
        # retrieve same PCA input matrix used in tab 3
        X_filt = st.session_state["X_filtered_raw"]
        X_log = log2p1(X_filt)
        X_in = st.session_state.get("X_qn", X_log)

        keep_samples = [s for s, flag in zip(sample_ids, out_mask) if not flag]
        X_clean = X_in[keep_samples].copy()

        scores2, pca2 = pca_on_samples(X_clean, n_components=int(n_pcs))
        st.session_state["X_clean"] = X_clean
        st.session_state["pca_scores_clean"] = scores2
        st.session_state["pca_input_sample_ids_clean"] = keep_samples
        st.session_state["pca_model_clean"] = pca2
        st.session_state["outliers"] = outliers

        st.success(f"Removed {len(outliers)} outliers. Clean samples: {len(keep_samples)}.")

    if "pca_scores_clean" in st.session_state:
        scores2 = st.session_state["pca_scores_clean"]
        keep_samples = st.session_state["pca_input_sample_ids_clean"]
        chart2 = plotly_pca_scatter(scores2, keep_samples, None, "PCA after outlier removal")
        st.plotly_chart(chart2, use_container_width=True)

with tabs[5]:
    st.subheader("Step 6 — Confounder regression & export")

    if "expr" not in st.session_state:
        st.info("Load expression first.")
        st.stop()

    expr: ExpressionData = st.session_state["expr"]
    genes_filt = st.session_state.get("genes_filtered", expr.genes)

    X_base = None
    # Prefer clean matrix (after outlier removal), else PCA input
    if "X_clean" in st.session_state:
        X_base = st.session_state["X_clean"]
        st.write("Using matrix after outlier removal.")
    elif "X_qn" in st.session_state:
        X_base = st.session_state["X_qn"]
        st.write("Using quantile-normalized matrix.")
    elif "X_filtered_raw" in st.session_state:
        X_base = log2p1(st.session_state["X_filtered_raw"])
        st.write("Using filtered log2(x+1) matrix.")
    else:
        st.info("Run filtering first.")
        st.stop()

    st.write("Matrix shape:", X_base.shape)

    # Add a gene id column for export convenience
    export_df = X_base.copy()
    if not genes_filt.empty:
        for c in genes_filt.columns:
            if c not in export_df.columns:
                export_df.insert(0, c, genes_filt[c].values)

    # Confounder regression
    meta_df = st.session_state.get("meta", None)

    if do_regress:
        if meta_df is None or sample_id_col not in meta_df.columns:
            st.error("Confounder regression requires metadata loaded with a valid Sample ID column.")
        else:
            covariate_options = [c for c in meta_df.columns if c != sample_id_col]
            covariates = st.multiselect(
                "Select covariates to regress out",
                options=covariate_options,
                default=[c for c in ["SMRIN", "SMTSISCH"] if c in covariate_options],
            )
            if covariates:
                if st.button("Run regression (compute residuals)", type="primary"):
                    with st.spinner("Regressing out confounders (multi-output linear regression)..."):
                        resid_df, kept_samples = regress_out_confounders(X_base, meta_df, sample_id_col, covariates)

                    st.session_state["residuals"] = resid_df
                    st.session_state["residuals_kept_samples"] = kept_samples
                    st.success(f"Residuals computed. Samples kept (no NaNs): {len(kept_samples)}.")

                if "residuals" in st.session_state:
                    resid_df = st.session_state["residuals"]
                    st.write("Residuals shape:", resid_df.shape)

                    # PCA before/after
                    scores_b, pca_b = pca_on_samples(X_base[resid_df.columns], n_components=10)
                    scores_a, pca_a = pca_on_samples(resid_df, n_components=10)
                    
                    sample_ids_resid = list(resid_df.columns)
                    
                    # Option to color by attribute
                    st.markdown("### PCA Comparison: Before vs After Regression")
                    color_attr = st.selectbox(
                        "Color scatter plots by attribute",
                        options=["(none)"] + [c for c in meta_df.columns if c != sample_id_col],
                        index=0,
                        key="regress_color_attr"
                    )
                    
                    color_before = None
                    color_after = None
                    color_name = "Value"
                    
                    if color_attr != "(none)":
                        aligned = meta_df.set_index(sample_id_col)
                        attr_data = aligned.reindex(sample_ids_resid)[color_attr]
                        if pd.api.types.is_numeric_dtype(attr_data):
                            color_before = attr_data.values.astype(float)
                            color_after = attr_data.values.astype(float)
                        else:
                            # Encode categorical
                            color_before = pd.Categorical(attr_data).codes.astype(float)
                            color_after = pd.Categorical(attr_data).codes.astype(float)
                        color_name = color_attr

                    fig_comparison = plotly_pca_comparison(
                        scores_b, scores_a,
                        sample_ids_resid, sample_ids_resid,
                        color_before=color_before, color_after=color_after,
                        color_name=color_name
                    )
                    fig_comparison.update_layout(title_text="PCA: Before vs After Confounder Regression")
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Also show 2x2 grid for covariates before/after
                    st.markdown("### PCA Grid: Multiple Covariates Before vs After")
                    grid_covariates = st.multiselect(
                        "Select covariates for comparison grid",
                        options=[c for c in meta_df.columns if c != sample_id_col],
                        default=[c for c in ["SMRIN", "SMTSISCH"] if c in meta_df.columns][:2],
                        key="regress_grid_covs"
                    )
                    
                    if grid_covariates:
                        for cov in grid_covariates:
                            aligned = meta_df.set_index(sample_id_col)
                            cov_data = aligned.reindex(sample_ids_resid)[cov]
                            if pd.api.types.is_numeric_dtype(cov_data):
                                cov_vals = cov_data.values.astype(float)
                            else:
                                cov_vals = pd.Categorical(cov_data).codes.astype(float)
                            
                            fig_cov = plotly_pca_comparison(
                                scores_b, scores_a,
                                sample_ids_resid, sample_ids_resid,
                                color_before=cov_vals, color_after=cov_vals,
                                color_name=cov
                            )
                            fig_cov.update_layout(title_text=f"PCA Before/After Regression - Colored by {cov}")
                            st.plotly_chart(fig_cov, use_container_width=True)

                    # Prepare residual export
                    resid_export = resid_df.copy()
                    if not genes_filt.empty:
                        for c in genes_filt.columns[::-1]:
                            resid_export.insert(0, c, genes_filt[c].values)

                    export_df = resid_export

            else:
                st.info("Select at least one covariate to regress out.")

    st.divider()
    st.subheader("Export cleaned matrix")

    fmt = st.selectbox("Format", options=["CSV", "Parquet"], index=1)
    out_name = st.text_input("Output filename (no folders)", value="expression_cleaned.parquet" if fmt == "Parquet" else "expression_cleaned.csv")
    save_local = st.checkbox("Also save to local path", value=False)
    out_dir = st.text_input("Local output directory", value=os.getcwd()) if save_local else ""

    if fmt == "CSV":
        data_bytes = export_df.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    else:
        # Parquet (more efficient)
        buf = io.BytesIO()
        export_df.to_parquet(buf, index=False)
        data_bytes = buf.getvalue()
        mime = "application/octet-stream"

    st.download_button(
        label="Download cleaned matrix",
        data=data_bytes,
        file_name=out_name,
        mime=mime,
        type="primary",
    )

    if save_local:
        if st.button("Write file to disk"):
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, out_name)
                if fmt == "CSV":
                    export_df.to_csv(out_path, index=False)
                else:
                    export_df.to_parquet(out_path, index=False)
                st.success(f"Wrote: {out_path}")
            except Exception as e:
                st.error(f"Failed writing file: {e}")

    # Summary
    st.divider()
    st.subheader("Summary")
    outliers = st.session_state.get("outliers", [])
    st.write(
        {
            "genes_in_export": int(export_df.shape[0]),
            "samples_in_export": int(export_df.shape[1] - (genes_filt.shape[1] if not genes_filt.empty else 0)),
            "outliers_removed": len(outliers),
            "quantile_normalized": ("X_qn" in st.session_state),
            "confounders_regressed": ("residuals" in st.session_state),
        }
    )
