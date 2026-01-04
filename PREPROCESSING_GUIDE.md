# Gene Expression Preprocessing Guide

This guide explains the preprocessing steps implemented in this Streamlit app, designed to prepare gene expression data for downstream analyses like ssGSEA (single-sample Gene Set Enrichment Analysis).

## Overview

The app implements a **6-step preprocessing pipeline** following best practices for RNA-seq and microarray data analysis. Each step addresses specific technical and biological challenges in gene expression data.

---

## Step-by-Step Preprocessing Workflow

### Step 1: Data Loading 📊

**Purpose**: Load and validate expression data and metadata

**What happens**:
- Load gene-by-sample expression matrix (genes in rows, samples in columns)
- Load optional sample metadata (e.g., sample attributes, batch information)
- Load optional subject phenotypes (e.g., age, sex, disease status)
- Merge metadata with sample attributes

**Input formats**:
- Expression: TSV, CSV, or GCT files
- Metadata: TSV or CSV with sample IDs
- Phenotypes: TSV or CSV with subject IDs

**Tips**:
- For GTEx data: Use "skip rows = 2" to skip header rows
- Gene columns typically: Name, Description, id
- Ensure sample IDs match between expression and metadata

---

### Step 2: Log Transformation & Filtering 🔬

**Purpose**: Normalize data distribution and remove uninformative genes

#### 2A. Log2(x+1) Transformation

**Why**:
- Raw expression values (TPM/RPKM) are highly skewed
- Log transformation stabilizes variance across expression ranges
- Makes data approximately normal for statistical methods
- Adding 1 handles zero expression values

**Formula**: `log2(TPM + 1)`

**When to use**:
- ✅ Always for raw TPM, RPKM, FPKM values
- ✅ For any positive-valued expression data
- ❌ Not for already log-transformed data
- ❌ Not for negative values (residuals, z-scores)

#### 2B. Low-Expression Filtering

**Why**:
- Genes with very low expression are mostly noise
- Reduces false positives in downstream analyses
- Improves statistical power (fewer multiple testing corrections)

**Logic**: Keep gene if expression ≥ `min_tpm` in ≥ `min_samples`

**Typical parameters**:
- `min_tpm`: 0.5 - 2.0 (depends on sequencing depth)
- `min_samples`: 20-50% of total samples

**Example**: Keep genes with TPM ≥ 1.0 in ≥ 20% of samples

#### 2C. Low-Variance Filtering

**Why**:
- Genes with minimal variance provide little biological information
- Reduces dimensionality for PCA
- Improves computational efficiency

**Logic**: Keep gene if variance(log2 expression) ≥ quantile threshold

**Typical parameters**:
- `var_quantile`: 0.1 - 0.3 (keep top 70-90% most variable genes)

**Example**: Remove bottom 20% least variable genes

**Interactive Tools in App**:
- Expression threshold explorer: See genes kept at different thresholds
- Variance distribution plots: Visualize variance across genes
- Combined filter preview: See which genes pass both filters

---

### Step 3: Quantile Normalization (Optional) 📈

**Purpose**: Make expression distributions identical across all samples

**Why**:
- Removes systematic technical variation between samples
- Essential for multi-batch data
- Corrects for library size differences, GC bias, batch effects

**Algorithm** (from ssGSEA tutorial notebook):
1. Transpose to samples × genes
2. Sort each gene column independently
3. Calculate row means of sorted matrix (mean at each rank)
4. Replace each value with mean of its rank
5. Transpose back to genes × samples

**Result**: All samples have identical empirical distributions

**When to use**:
✅ **Use when**:
- Samples from different batches or sequencing runs
- Sample distributions visibly different in QC plots
- Strong batch effects visible in PCA

❌ **Skip when**:
- Data already well-normalized (e.g., GTEx normalized TPM)
- Very large matrices (can be slow)
- Need to preserve exact expression magnitudes

**Visual checks**:
- Before/after histograms should show aligned distributions
- Sample boxplots should have similar ranges and medians

---

### Step 4: Quality Control with PCA 🎯

**Purpose**: Visualize sample relationships and identify technical artifacts

**What is PCA**:
- Principal Component Analysis reduces data to key dimensions
- Each sample becomes a point in PC space
- Similar samples cluster together
- Outliers appear as isolated points

**Why on samples (not genes)**:
- We want to understand sample-to-sample relationships
- Detect batch effects, outliers, mislabeled samples
- Visualize technical vs biological variation

**Key visualizations**:

1. **Variance Explained Plot**:
   - Shows how much variance each PC captures
   - First few PCs should capture most variance
   - If PC1 explains >50%, likely technical effect

2. **PC1 vs PC2 Scatter**:
   - Color by metadata to identify confounders
   - Samples should cluster by biology, not batch
   - Outliers appear far from main cluster

3. **PCA Grid** (multiple covariates):
   - View PCA colored by different attributes
   - Identify which technical factors correlate with PCs
   - RIN, batch, ischemic time often correlate with PC1

4. **Gene-Attribute Correlations**:
   - Quantify relationship between expression and covariates
   - High correlations indicate confounding
   - Guides covariate selection for regression

**Interpretation**:
- **Good**: Samples cluster by tissue/disease, not batch
- **Problem**: Samples cluster by batch/technical factor
- **Action**: Use confounder regression (Step 6)

---

### Step 5: Outlier Detection 🔍

**Purpose**: Identify and remove problematic samples

Three complementary methods for robust detection:

#### Method A: Mahalanobis Distance (PCA space)

**What it detects**: Samples unusual in multivariate expression patterns

**How it works**:
- Measures distance from center in multi-dimensional PC space
- Accounts for covariance between PCs
- Uses chi-squared distribution for statistical threshold

**Parameters**:
- `df`: Number of PCs to use (typically 5-10)
- `chi2_quantile`: Threshold (0.95 = flag top 5% most distant)

**Best for**: Detecting samples with overall unusual expression

**Diagnostics**:
- Q-Q plot: Should follow diagonal if distances are chi-squared distributed
- Confidence ellipse: Visualizes threshold boundary in PC1-PC2
- Per-PC contributions: Shows which PCs drive outlier status

#### Method B: Hierarchical Clustering (Ward linkage)

**What it detects**: Small isolated sample groups

**How it works**:
- Clusters samples based on PCA coordinates
- Cuts dendrogram at specified height
- Flags samples in small clusters as outliers

**Parameters**:
- `linkage_method`: Ward (minimizes variance) recommended
- `cut_fraction`: Fraction of max height to cut (0.5-0.8)
- `min_cluster_fraction`: Min cluster size threshold

**Best for**: Detecting batch-specific outliers

**Diagnostics**:
- Dendrogram: Visualizes sample relationships
- Cluster sizes: Should have one large cluster + potential small outliers

#### Method C: IQR Method (on PCs)

**What it detects**: Samples extreme in individual PCs

**How it works**:
- For each PC, use interquartile range to find outliers
- Union of outliers across first N PCs
- Simple and interpretable

**Parameters**:
- `n_pcs`: Number of PCs to check (3-10)
- `iqr_k`: IQR multiplier (1.5 = standard, 3.0 = strict)

**Best for**: Simple outlier detection, interpretable results

**Diagnostics**:
- Table showing IQR bounds per PC
- Number of outliers per PC

#### Combined Approach

**Recommendation**: Use all three methods
- Union of all methods = maximum sensitivity
- Samples flagged by multiple methods = high confidence outliers
- Review flagged samples before removal

**Common outlier causes**:
- Poor RNA quality (low RIN score)
- Technical failures (library prep, sequencing)
- Contamination
- Mislabeled samples
- True biological outliers (investigate before removing!)

**After outlier removal**:
- PCA should show cleaner clustering
- Variance explained by top PCs may decrease (good!)
- Downstream analyses more robust

---

### Step 6: Confounder Regression (Optional) 🧮

**Purpose**: Remove technical variation while preserving biological signal

**What is it**:
- Linear regression of covariates from expression
- Returns residuals: variation NOT explained by covariates
- Also called "batch correction" or "residualization"

**Method**: Multi-Output Linear Regression
```
For each gene:
    expression = β₀ + β₁×RIN + β₂×batch + β₃×age + ... + ε
    
Return ε (residuals) = expression - predicted
```

**Common confounders to regress**:

**Technical factors** (always regress):
- Batch/sequencing run
- RIN (RNA Integrity Number)
- Ischemic time / PMI (post-mortem interval)
- Library prep protocol
- Sequencing platform

**Demographic factors** (regress if not of interest):
- Age
- Sex
- Ancestry/population

**Biological factors** (NEVER regress):
- Disease status (if comparing disease vs control)
- Treatment group (if analyzing treatment effects)
- Any variable that IS your biological question

**Covariate types**:
- **Numeric** (RIN, age, PMI): Used directly in regression
- **Categorical** (batch, sex): One-hot encoded automatically

**When to use**:
✅ **Use when**:
- Known confounders correlate with top PCs
- Batch effects visible in PCA
- Technical factors (RIN, ischemic time) affect expression

❌ **Skip when**:
- No metadata available
- Confounders are biological variables of interest
- Only one batch (nothing to correct)

**Validation** (use app visualizations):

1. **PCA Before/After**:
   - Confounder correlation with PCs should decrease
   - Biological signal should remain
   - Overall structure preserved but "cleaner"

2. **Colored by confounders**:
   - Before: clear separation by batch/RIN
   - After: minimal separation by technical factors

3. **Gene-attribute correlations**:
   - Should decrease for technical factors
   - Should remain for biological factors

**Alternative methods**:
- **ComBat** (sva package): Specialized batch correction
- **limma removeBatchEffect**: Similar linear regression
- **SVA** (Surrogate Variable Analysis): Estimates hidden confounders

This method works when confounders are known and measured.

---

## Final Output 💾

**What you get**:
- Fully preprocessed expression matrix
- Genes × samples (after filtering)
- All transformations applied
- Outliers removed (if selected)
- Confounders regressed out (if selected)

**Export formats**:
- **CSV**: Human-readable, larger files
- **Parquet**: Compressed, faster to read/write (recommended)

**What to do next**:

**For ssGSEA**:
1. Load preprocessed matrix into ssGSEA tool
2. Ensure gene IDs match your gene sets
3. Run ssGSEA to get pathway enrichment scores per sample
4. Analyze enrichment patterns across samples

**For other analyses**:
- **GSEA**: Gene Set Enrichment Analysis (comparing groups)
- **Differential expression**: DESeq2, edgeR, limma
- **Machine learning**: Classification, clustering
- **Pathway analysis**: DAVID, Enrichr, IPA
- **Network analysis**: WGCNA, co-expression networks

---

## Best Practices Summary ✅

### Do's:
✅ Always log-transform raw TPM/RPKM data  
✅ Filter lowly expressed and low-variance genes  
✅ Use quantile normalization for multi-batch data  
✅ Always do PCA-based quality control  
✅ Remove clear outliers (low RIN, technical failures)  
✅ Regress known technical confounders  
✅ Validate each step with visualizations  
✅ Document all parameters and decisions  

### Don'ts:
❌ Don't skip quality control (PCA)  
❌ Don't remove biological outliers without investigation  
❌ Don't regress biological variables of interest  
❌ Don't over-filter (removes biological signal)  
❌ Don't normalize already-normalized data  
❌ Don't ignore batch effects visible in PCA  

---

## Troubleshooting 🔧

**Problem**: Too many outliers detected
- **Solution**: Relax thresholds (increase chi2_quantile, iqr_k)
- **Check**: Are these true technical failures or biological variation?

**Problem**: Batch effects still visible after regression
- **Solution**: Add more batch-related covariates
- **Try**: Interaction terms, non-linear models, ComBat

**Problem**: Loss of biological signal after preprocessing
- **Solution**: Less aggressive filtering, check regressed covariates
- **Validate**: Compare biological effects before/after preprocessing

**Problem**: Quantile normalization very slow
- **Solution**: Use fast method, subset genes, or skip if already normalized

**Problem**: Can't identify confounders
- **Solution**: Use unsupervised methods (SVA) to estimate hidden factors

---

## References 📚

**Key papers**:
- Bolstad et al. (2003) "A comparison of normalization methods for high density oligonucleotide array data based on variance and bias" - Quantile normalization
- Johnson et al. (2007) "Adjusting batch effects in microarray expression data using empirical Bayes methods" - ComBat
- GTEx Consortium (2020) "The GTEx Consortium atlas of genetic regulatory effects across human tissues" - Quality control guidelines

**Methods**:
- PCA: Jolliffe & Cadima (2016) "Principal component analysis: a review and recent developments"
- Mahalanobis distance: De Maesschalck et al. (2000) "The Mahalanobis distance"
- Hierarchical clustering: Murtagh & Contreras (2012) "Algorithms for hierarchical clustering"

**Preprocessing best practices**:
- Conesa et al. (2016) "A survey of best practices for RNA-seq data analysis"
- Law et al. (2016) "RNA-seq analysis is easy as 1-2-3 with limma, Glimma and edgeR"

---

## Need Help? 🆘

**Common workflows**:
- See README.md for installation and basic usage
- See app.py docstring for detailed pipeline explanation
- Use interactive tools in app to explore threshold effects

**For ssGSEA specifically**:
This preprocessing produces data ready for ssGSEA. Key points:
1. Expression should be normalized and log-transformed ✅
2. Genes should be filtered to reduce noise ✅
3. Samples should be QC'd and outliers removed ✅
4. Technical confounders should be removed ✅
5. All samples should have comparable distributions ✅

Your data is now ready for pathway enrichment analysis!
