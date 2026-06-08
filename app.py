import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from radhaomics import (
    compute_sex_ratio,
    flag_sex_biased_genes,
    compute_statistical_power,
    compute_biasdelta,
    detect_sex_column,
    GTEX_TISSUE_WEIGHTS,
)

st.set_page_config(
    page_title="RadhaOmics",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .main { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e9ecef;
    }
    .score-high { color: #C1440E; font-size: 2.5rem; font-weight: 600; }
    .score-med  { color: #BA7517; font-size: 2.5rem; font-weight: 600; }
    .score-low  { color: #3B6D11; font-size: 2.5rem; font-weight: 600; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RadhaOmics")
    st.markdown("*Tissue-specific sex-bias detection*")
    st.divider()
    tissue = st.selectbox(
        "Select tissue type",
        list(GTEX_TISSUE_WEIGHTS.keys()),
        index=list(GTEX_TISSUE_WEIGHTS.keys()).index("blood")
    )
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "RadhaOmics computes the **BiasΔ score** — "
        "the first tissue-aware metric for sex-bias "
        "propagation in omics datasets."
    )
    st.divider()
    st.markdown(
        "*Named for Anuradha —*  \n"
        "*because women deserve*  \n"
        "*to be in the data.*"
    )

# ── Main ──────────────────────────────────────────────────
st.title("RadhaOmics 🧬")
st.markdown(
    "**Tissue-specific sex-bias detection and propagation scoring for omics datasets**"
)
st.divider()

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or TSV)",
    type=["csv", "tsv", "txt"],
    help="Your file should include a sex/gender column and gene expression columns"
)

col_ex1, col_ex2 = st.columns([1, 4])
with col_ex1:
    load_example = st.button("Load example dataset")

if load_example or uploaded_file is not None:

    if load_example:
        np.random.seed(42)
        n = 200
        sex = (["male"] * 156) + (["female"] * 44)
        np.random.shuffle(sex)
        genes = {
            "XIST":   [0.1 if s == "male" else 8.2 + np.random.normal(0, 0.5)
                       for s in sex],
            "DDX3Y":  [7.4 + np.random.normal(0, 0.4) if s == "male" else 0.2
                       for s in sex],
            "KDM5D":  [6.1 + np.random.normal(0, 0.6) if s == "male" else 0.3
                       for s in sex],
            "TP53":   [4.2 + np.random.normal(0, 1.0) for _ in sex],
            "BRCA1":  [3.8 + np.random.normal(0, 0.8) for _ in sex],
            "RPS4Y1": [5.2 + np.random.normal(0, 0.5) if s == "male" else 0.4
                       for s in sex],
            "GAPDH":  [8.1 + np.random.normal(0, 0.3) for _ in sex],
            "ACTB":   [7.9 + np.random.normal(0, 0.3) for _ in sex],
        }
        df = pd.DataFrame({"sample_id": range(n), "sex": sex, **genes})
        st.info("Example dataset loaded — 200 samples, RNA-seq simulation")
    else:
        sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success(f"File loaded — {df.shape[0]} rows × {df.shape[1]} columns")

    sex_col = detect_sex_column(df)
    if sex_col is None:
        st.error("Could not detect a sex/gender column. "
                 "Please make sure your file has a column named 'sex' or 'gender'.")
        st.stop()

    # ── Run analysis ──────────────────────────────────────
    ratio  = compute_sex_ratio(df, sex_col)
    flags  = flag_sex_biased_genes(df)
    power  = compute_statistical_power(ratio["female"])
    bd     = compute_biasdelta(ratio, flags, tissue)

    st.divider()
    st.subheader("Results")

    # ── Score row ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        css = ("score-high" if bd["fairness_score"] < 41 else
               "score-med"  if bd["fairness_score"] < 71 else "score-low")
        st.markdown(f"**Fairness score**")
        st.markdown(
            f"<div class='{css}'>{bd['fairness_score']}<span style='font-size:1rem'>/100</span></div>",
            unsafe_allow_html=True
        )
        st.caption(bd["verdict"])
    with c2:
        st.metric("M:F ratio", f"{ratio['mf_ratio']}:1",
                  delta="Above threshold" if ratio["ratio_flag"] else "Within threshold",
                  delta_color="inverse")
    with c3:
        st.metric("Sex-biased genes", len(flags),
                  delta="Flagged" if flags else "None detected",
                  delta_color="inverse" if flags else "normal")
    with c4:
        st.metric("Female power",
                  f"{round(power['achieved_power']*100)}%",
                  delta="Insufficient" if power["power_flag"] else "Sufficient",
                  delta_color="inverse" if power["power_flag"] else "normal")

    st.divider()

    # ── Charts row ────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Sample sex distribution**")
        fig_donut = go.Figure(go.Pie(
            labels=["Male", "Female"],
            values=[ratio["male"], ratio["female"]],
            hole=0.55,
            marker_colors=["#1B6CA8", "#C1440E"],
            textinfo="label+percent"
        ))
        fig_donut.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            annotations=[dict(
                text=f"{ratio['total']}<br>samples",
                x=0.5, y=0.5, font_size=14, showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        st.markdown("**BiasΔ score breakdown**")
        components = {
            "Tissue weight": bd["tissue_weight"],
            "Imbalance":     bd["imbalance"],
            "Gene flags":    bd["flag_penalty"],
        }
        fig_bar = px.bar(
            x=list(components.keys()),
            y=list(components.values()),
            color=list(components.keys()),
            color_discrete_sequence=["#1B6CA8", "#C1440E", "#BA7517"],
            labels={"x": "", "y": "Contribution to BiasΔ"}
        )
        fig_bar.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Gene flags table ──────────────────────────────────
    if flags:
        st.divider()
        st.markdown("**Sex-biased genes detected in your dataset**")
        flag_df = pd.DataFrame(flags)
        flag_df.columns = ["Gene", "Chromosome", "Bias direction", "Risk"]
        st.dataframe(flag_df, use_container_width=True, hide_index=True)
    else:
        st.success("No known sex-biased genes detected in your dataset columns.")

    # ── Citation ──────────────────────────────────────────
    st.divider()
    st.markdown("**Copy this into your methods section**")
    citation = (
        f"Sex bias analysis was performed using RadhaOmics v1.0 "
        f"(github.com/KirtanaPrem/RadhaOmics). "
        f"Dataset (n={ratio['total']}) showed M:F ratio of "
        f"{ratio['mf_ratio']}:1, fairness score {bd['fairness_score']}/100, "
        f"tissue: {tissue}. "
        f"{len(flags)} sex-biased genes flagged. "
        f"Female statistical power: {round(power['achieved_power']*100)}%."
    )
    st.code(citation, language=None)

else:
    st.info(
        "Upload a CSV/TSV dataset or click **Load example dataset** to see "
        "RadhaOmics in action."
    )
    st.markdown("""
    **What RadhaOmics checks:**
    - Sex ratio audit — M:F threshold vs NIH/FDA guidelines
    - Tissue-specific BiasΔ score — novel contribution
    - Sex-biased gene flags — chrX/chrY and autosomal
    - Female statistical power estimate
    - Auto-generated methods citation
    """)
