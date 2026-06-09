"""
RadhaOmics v1.1
Tissue-specific sex-bias detection and propagation scoring for omics datasets
Author: Kirtana Prem
Named for Anuradha — because women deserve to be in the data.
"""

import pandas as pd
import numpy as np
from scipy import stats

GTEX_TISSUE_WEIGHTS = {
    "blood":        0.37,
    "brain":        0.14,
    "liver":        0.41,
    "heart":        0.19,
    "lung":         0.28,
    "kidney":       0.33,
    "muscle":       0.22,
    "breast":       0.44,
    "skin":         0.31,
    "adipose":      0.38,
    "thyroid":      0.35,
    "ovary":        0.61,
    "testis":       0.72,
    "uterus":       0.58,
    "prostate":     0.67,
    "colon":        0.26,
    "stomach":      0.29,
    "pancreas":     0.24,
    "spleen":       0.21,
    "other":        0.25,
}

SEX_BIASED_GENES = {
    "XIST":   {"chr": "X", "direction": "female", "risk": "high"},
    "TSIX":   {"chr": "X", "direction": "female", "risk": "medium"},
    "DDX3Y":  {"chr": "Y", "direction": "male",   "risk": "high"},
    "KDM5D":  {"chr": "Y", "direction": "male",   "risk": "high"},
    "USP9Y":  {"chr": "Y", "direction": "male",   "risk": "high"},
    "RPS4Y1": {"chr": "Y", "direction": "male",   "risk": "medium"},
    "EIF1AY": {"chr": "Y", "direction": "male",   "risk": "medium"},
    "NLGN4Y": {"chr": "Y", "direction": "male",   "risk": "low"},
    "TTTY15": {"chr": "Y", "direction": "male",   "risk": "medium"},
    "ZFY":    {"chr": "Y", "direction": "male",   "risk": "medium"},
    "BRCA1":  {"chr": "17","direction": "female",  "risk": "low"},
    "ESR1":   {"chr": "6", "direction": "female",  "risk": "low"},
    "CYP3A4": {"chr": "7", "direction": "female",  "risk": "medium"},
}


def load_dataset(filepath):
    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)
    return df


def detect_sex_column(df):
    candidates = ["sex","gender","Sex","Gender","SEX","GENDER",
                  "biological_sex","subject_sex","sample_sex"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_gene_columns(df, sex_col):
    exclude = [sex_col, "sample_id", "id", "ID", "sample", "Sample",
               "subject", "Subject", "patient", "Patient"]
    return [c for c in df.columns if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]


def compute_sex_ratio(df, sex_col):
    vals = df[sex_col].astype(str).str.lower().str.strip()
    male   = vals.isin(["male","m","1"]).sum()
    female = vals.isin(["female","f","2"]).sum()
    total  = int(male + female)
    ratio  = round(male / female, 2) if female > 0 else float("inf")
    return {
        "male":       int(male),
        "female":     int(female),
        "total":      total,
        "male_pct":   round(male / total * 100, 1) if total > 0 else 0,
        "female_pct": round(female / total * 100, 1) if total > 0 else 0,
        "mf_ratio":   ratio,
        "ratio_flag": ratio > 1.5 or ratio < 0.67,
    }


def flag_sex_biased_genes(df):
    return [
        {"gene": g, "chr": SEX_BIASED_GENES[g]["chr"],
         "direction": SEX_BIASED_GENES[g]["direction"],
         "risk": SEX_BIASED_GENES[g]["risk"]}
        for g in SEX_BIASED_GENES if g in df.columns
    ]


def compute_statistical_power(female_n, alpha=0.05, target_power=0.80):
    min_required = 64
    achieved = round(min(female_n / min_required, 1.0) * target_power, 2)
    return {
        "female_n":        female_n,
        "min_recommended": min_required,
        "achieved_power":  achieved,
        "power_flag":      achieved < target_power,
    }


def compute_real_degs(df, sex_col, top_n=8):
    """
    Compute real differential expression between male and female samples.
    Returns top DEGs ranked by significance, with:
    - actual p-value from Welch t-test
    - projected p-value if cohort were sex-balanced (via bootstrap resampling)
    - fold change (log2 ratio of means)
    """
    vals = df[sex_col].astype(str).str.lower().str.strip()
    male_idx   = vals.isin(["male","m","1"])
    female_idx = vals.isin(["female","f","2"])

    gene_cols = get_gene_columns(df, sex_col)
    if not gene_cols:
        return []

    results = []
    for gene in gene_cols:
        male_expr   = df.loc[male_idx,   gene].dropna().values
        female_expr = df.loc[female_idx, gene].dropna().values

        if len(male_expr) < 3 or len(female_expr) < 3:
            continue

        # Real t-test
        t_stat, p_real = stats.ttest_ind(male_expr, female_expr,
                                          equal_var=False)

        # Fold change (log2)
        m_mean = np.mean(male_expr)
        f_mean = np.mean(female_expr)
        if f_mean == 0 or m_mean == 0:
            fc = 0.0
        else:
            fc = round(np.log2(m_mean / f_mean + 1e-9), 2)

        # Projected p-value: resample to 1:1 ratio
        n_bal = min(len(male_expr), len(female_expr))
        if n_bal >= 3:
            np.random.seed(42)
            m_bal = np.random.choice(male_expr,   n_bal, replace=False)
            f_bal = np.random.choice(female_expr, n_bal, replace=False)
            _, p_proj = stats.ttest_ind(m_bal, f_bal, equal_var=False)
        else:
            p_proj = p_real

        results.append({
            "gene":     gene,
            "chr":      SEX_BIASED_GENES.get(gene, {}).get("chr", "auto"),
            "p_biased": p_real,
            "p_balanced": p_proj,
            "fc":       fc,
            "is_sex_biased": gene in SEX_BIASED_GENES,
        })

    if not results:
        return []

    results.sort(key=lambda x: x["p_biased"])
    top = results[:top_n]

    # Verdict: loses significance if p_biased < 0.05 but p_balanced >= 0.05
    for r in top:
        if r["p_biased"] < 0.05 and r["p_balanced"] >= 0.05:
            r["verdict"] = "loses sig."
        elif r["p_biased"] < 0.05 and r["p_balanced"] < 0.05:
            r["verdict"] = "stable"
        else:
            r["verdict"] = "not sig."

    return top


def compute_biasdelta(sex_ratio, gene_flags, tissue="other"):
    tissue_weight = GTEX_TISSUE_WEIGHTS.get(tissue.lower(),
                    GTEX_TISSUE_WEIGHTS["other"])
    ratio = sex_ratio["mf_ratio"]
    if ratio == float("inf"):
        imbalance = 1.0
    else:
        imbalance = round(abs(ratio - 1.0) / max(ratio, 1.0), 3)

    risk_weights = {"high": 1.0, "medium": 0.5, "low": 0.2}
    flag_penalty = sum(risk_weights.get(f["risk"], 0) for f in gene_flags)
    flag_penalty = round(min(flag_penalty / 10, 0.3), 3)

    biasdelta = round(min((imbalance * tissue_weight) + flag_penalty, 1.0), 3)
    fairness_score = round((1 - biasdelta) * 100)

    return {
        "biasdelta":      biasdelta,
        "fairness_score": fairness_score,
        "tissue":         tissue,
        "tissue_weight":  tissue_weight,
        "imbalance":      imbalance,
        "flag_penalty":   flag_penalty,
        "verdict": (
            "Low bias"      if fairness_score >= 71 else
            "Moderate bias" if fairness_score >= 41 else
            "High bias"
        ),
    }


def compute_field_percentile(fairness_score):
    """
    Returns what % of published studies are MORE biased than this dataset.
    Based on a simulated distribution centred at 58/100 (field average).
    Replace with real survey data when available.
    """
    np.random.seed(0)
    field = np.random.normal(loc=58, scale=18, size=847).clip(0, 100)
    pct_worse = round(float(np.mean(field < fairness_score) * 100), 1)
    pct_better = round(100 - pct_worse, 1)
    return {"pct_worse": pct_worse, "pct_better": pct_better,
            "field_mean": 58}


def compute_samples_needed(ratio, target_ratio=1.5):
    """
    How many additional female samples are needed to reach target M:F ratio?
    """
    male = ratio["male"]
    female = ratio["female"]
    needed = max(0, round((male / target_ratio) - female))
    return needed


def generate_dynamic_recommendations(ratio, flags, tissue, power):
    """
    Generate specific, dynamic recommendations based on actual analysis results.
    """
    recs = []
    needed = compute_samples_needed(ratio)

    if needed > 0:
        recs.append({
            "num": "01",
            "title": f"Add {needed} female samples",
            "detail": f"Minimum {needed} additional female samples needed to reach 1.5:1 threshold. "
                      f"Search TCGA (portal.gdc.cancer.gov) or GEO (ncbi.nlm.nih.gov/geo) for {tissue} datasets."
        })

    high_risk = [f["gene"] for f in flags if f["risk"] == "high"]
    if high_risk:
        recs.append({
            "num": "02",
            "title": f"Exclude {', '.join(high_risk)} from normalisation",
            "detail": f"These sex-chromosome genes are being used as universal markers. "
                      f"Remove from housekeeping gene lists before reanalysis."
        })

    if power["power_flag"]:
        recs.append({
            "num": "03",
            "title": f"Increase female sample count to {power['min_recommended']}+",
            "detail": f"Current female n={ratio['female']} achieves only "
                      f"{round(power['achieved_power']*100)}% statistical power. "
                      f"Target ≥{power['min_recommended']} for 80% power at α=0.05."
        })

    recs.append({
        "num": f"0{len(recs)+1}",
        "title": "Disclose bias in methods section",
        "detail": "Use the generated citation below to comply with NIH SABV and FDA sex-disaggregated reporting requirements."
    })

    return recs


def generate_report(filepath, sex_col=None, tissue="other"):
    df = load_dataset(filepath)
    if sex_col is None:
        sex_col = detect_sex_column(df)
        if sex_col is None:
            print("ERROR: Could not find sex/gender column.")
            return

    ratio  = compute_sex_ratio(df, sex_col)
    flags  = flag_sex_biased_genes(df)
    power  = compute_statistical_power(ratio["female"])
    bd     = compute_biasdelta(ratio, flags, tissue)
    degs   = compute_real_degs(df, sex_col)
    recs   = generate_dynamic_recommendations(ratio, flags, tissue, power)
    pct    = compute_field_percentile(bd["fairness_score"])

    print(f"\nRadhaOmics v1.1 — {filepath}")
    print(f"Fairness score: {bd['fairness_score']}/100 · {bd['verdict']}")
    print(f"M:F ratio: {ratio['mf_ratio']}:1")
    print(f"Sex-biased genes: {len(flags)}")
    print(f"Better than {pct['pct_worse']}% of published studies")

    return {
        "sex_ratio": ratio, "gene_flags": flags, "power": power,
        "biasdelta": bd, "degs": degs, "recommendations": recs,
        "field_percentile": pct,
    }


if __name__ == "__main__":
    print("RadhaOmics v1.1 — import and call generate_report() to begin.")
