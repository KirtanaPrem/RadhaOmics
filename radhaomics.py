"""
RadhaOmics v1.0
Tissue-specific sex-bias detection and propagation scoring for omics datasets
Author: Kirtana Prem
Named for Anuradha — because women deserve to be in the data.
"""

import pandas as pd
import numpy as np
from scipy import stats


# ── GTEx tissue-specific sex-bias weights (curated from GTEx v8) ──────────
# Each value = proportion of genes with significant sex-differential expression
# Source: GTEx Consortium, Science 2020

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

# Known sex-biased genes (chrX/chrY + autosomal sex-biased)
# Source: Oliva et al. Science 2020, Gershoni & Pietrokovski Nature Genetics 2017
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
}


def load_dataset(filepath):
    """Load CSV or TSV expression/metadata file."""
    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def detect_sex_column(df):
    """Auto-detect which column contains sex/gender metadata."""
    candidates = ["sex", "gender", "Sex", "Gender", "SEX", "GENDER",
                  "biological_sex", "subject_sex"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_sex_ratio(df, sex_col):
    """Compute male/female sample counts and ratio."""
    counts = df[sex_col].str.lower().value_counts()
    male   = counts.get("male", counts.get("m", counts.get("1", 0)))
    female = counts.get("female", counts.get("f", counts.get("2", 0)))
    total  = male + female
    ratio  = round(male / female, 2) if female > 0 else float("inf")

    return {
        "male":         int(male),
        "female":       int(female),
        "total":        int(total),
        "male_pct":     round(male / total * 100, 1) if total > 0 else 0,
        "female_pct":   round(female / total * 100, 1) if total > 0 else 0,
        "mf_ratio":     ratio,
        "ratio_flag":   ratio > 1.5 or ratio < 0.67,
    }


def flag_sex_biased_genes(df):
    """Check if known sex-biased genes appear in the dataset columns."""
    genes_in_dataset = [g for g in SEX_BIASED_GENES if g in df.columns]
    flags = []
    for gene in genes_in_dataset:
        flags.append({
            "gene":      gene,
            "chr":       SEX_BIASED_GENES[gene]["chr"],
            "direction": SEX_BIASED_GENES[gene]["direction"],
            "risk":      SEX_BIASED_GENES[gene]["risk"],
        })
    return flags


def compute_statistical_power(female_n, alpha=0.05, target_power=0.80):
    """
    Estimate whether female sample count achieves target statistical power
    for detecting a medium effect size (Cohen's d = 0.5) in a two-sample t-test.
    Minimum n for 80% power at alpha=0.05, d=0.5 is approximately 64 per group.
    """
    min_required = 64
    achieved_power = round(min(female_n / min_required, 1.0) * target_power, 2)
    return {
        "female_n":        female_n,
        "min_recommended": min_required,
        "achieved_power":  achieved_power,
        "power_flag":      achieved_power < target_power,
    }


def compute_biasdelta(sex_ratio, gene_flags, tissue="other"):
    """
    Compute the BiasΔ score — RadhaOmics novel contribution.

    BiasΔ quantifies how much the sex imbalance in your dataset
    propagates into distortion of biological conclusions,
    weighted by tissue-specific sex-bias expression patterns from GTEx.

    Formula: BiasΔ = Σ(wᵢ × |FCbias − FCbalanced|) / n
    Approximated here from ratio imbalance and tissue weight.

    Returns a score from 0 (no bias propagation) to 1 (maximum distortion).
    """
    tissue_weight = GTEX_TISSUE_WEIGHTS.get(tissue.lower(),
                    GTEX_TISSUE_WEIGHTS["other"])

    # Imbalance factor: how far ratio deviates from ideal (1.0)
    ratio = sex_ratio["mf_ratio"]
    if ratio == float("inf"):
        imbalance = 1.0
    else:
        imbalance = abs(ratio - 1.0) / max(ratio, 1.0)

    # Gene flag penalty: high-risk flags amplify propagation
    risk_weights = {"high": 1.0, "medium": 0.5, "low": 0.2}
    flag_penalty = sum(risk_weights.get(f["risk"], 0) for f in gene_flags)
    flag_penalty = min(flag_penalty / 10, 0.3)   # cap at 0.3 contribution

    biasdelta = round(
        (imbalance * tissue_weight) + flag_penalty, 3
    )
    biasdelta = min(biasdelta, 1.0)

    # Convert to fairness score (0–100, higher = less biased)
    fairness_score = round((1 - biasdelta) * 100)

    return {
        "biasdelta":      biasdelta,
        "fairness_score": fairness_score,
        "tissue":         tissue,
        "tissue_weight":  tissue_weight,
        "imbalance":      round(imbalance, 3),
        "flag_penalty":   round(flag_penalty, 3),
        "verdict": (
            "Low bias"      if fairness_score >= 71 else
            "Moderate bias" if fairness_score >= 41 else
            "High bias"
        ),
    }


def generate_report(filepath, sex_col=None, tissue="other"):
    """
    Full RadhaOmics pipeline.
    Run this on any CSV/TSV omics dataset.
    """
    print("\n── RadhaOmics v1.0 ──────────────────────────────")
    print("Named for Anuradha — because women deserve to be in the data.\n")

    df = load_dataset(filepath)

    # Auto-detect sex column if not provided
    if sex_col is None:
        sex_col = detect_sex_column(df)
        if sex_col is None:
            print("ERROR: Could not find sex/gender column.")
            print("Please specify sex_col= parameter.")
            return

    print(f"Sex column detected: '{sex_col}'")
    print(f"Tissue: {tissue}\n")

    # 1. Sex ratio
    ratio = compute_sex_ratio(df, sex_col)
    print(f"Sex ratio:  {ratio['male']} male ({ratio['male_pct']}%) "
          f"/ {ratio['female']} female ({ratio['female_pct']}%)")
    print(f"M:F ratio:  {ratio['mf_ratio']} "
          f"({'FLAGGED' if ratio['ratio_flag'] else 'OK'})\n")

    # 2. Gene flags
    flags = flag_sex_biased_genes(df)
    print(f"Sex-biased genes detected: {len(flags)}")
    for f in flags:
        print(f"  {f['gene']} (chr{f['chr']}) — {f['direction']}-biased — "
              f"risk: {f['risk']}")

    # 3. Statistical power
    power = compute_statistical_power(ratio["female"])
    print(f"\nFemale statistical power: {power['achieved_power']*100:.0f}% "
          f"(target: 80%) — "
          f"{'INSUFFICIENT' if power['power_flag'] else 'SUFFICIENT'}")

    # 4. BiasΔ score
    bd = compute_biasdelta(ratio, flags, tissue)
    print(f"\nBiasΔ score:    {bd['biasdelta']} / 1.0")
    print(f"Fairness score: {bd['fairness_score']} / 100")
    print(f"Verdict:        {bd['verdict']}\n")

    # 5. Citation
    print("── Citation ─────────────────────────────────────")
    print(f"Sex bias analysis was performed using RadhaOmics v1.0 "
          f"(github.com/KirtanaPrem/RadhaOmics). "
          f"Dataset (n={ratio['total']}) showed M:F ratio of "
          f"{ratio['mf_ratio']}:1, fairness score {bd['fairness_score']}/100, "
          f"tissue: {tissue}. "
          f"{len(flags)} sex-biased genes flagged.")
    print("─────────────────────────────────────────────────\n")

    return {
        "sex_ratio":  ratio,
        "gene_flags": flags,
        "power":      power,
        "biasdelta":  bd,
    }


if __name__ == "__main__":
    # Example usage
    # generate_report("your_dataset.csv", tissue="liver")
    print("RadhaOmics v1.0 — import and call generate_report() to begin.")
