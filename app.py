import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from radhaomics import (
    compute_sex_ratio, flag_sex_biased_genes,
    compute_statistical_power, compute_biasdelta,
    detect_sex_column, GTEX_TISSUE_WEIGHTS,
)

st.set_page_config(page_title="RadhaOmics", page_icon="👑", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 760px; padding: 2rem 1.5rem; }
div[data-testid="stSidebar"] { display: none; }
.ro-nav {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 0.5px solid #e5e7eb;
    margin-bottom: 2rem;
}
.ro-logo { font-size: 16px; font-weight: 500; letter-spacing: -0.02em; color: #111; }
.ro-logo span { color: #1B6CA8; }
.ro-nav-right { font-size: 11px; color: #aaa; letter-spacing: 0.04em; }
.ro-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; color: #1B6CA8;
    background: #EBF3FA; padding: 4px 12px;
    border-radius: 20px; margin-bottom: 1rem;
}
.ro-h1 {
    font-size: 30px; font-weight: 300;
    letter-spacing: -0.03em; line-height: 1.2;
    margin-bottom: 0.75rem; color: #111;
}
.ro-h1 em {
    font-family: 'Instrument Serif', serif;
    font-style: italic; color: #1B6CA8;
}
.ro-sub {
    font-size: 13px; color: #666;
    font-weight: 300; line-height: 1.7;
    margin-bottom: 1.75rem;
}
.ro-card {
    background: #fff;
    border: 0.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ro-section-label {
    font-size: 10px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #aaa; margin-bottom: 10px;
}
.ro-score-num {
    font-size: 48px; font-weight: 500;
    letter-spacing: -0.05em; line-height: 1;
}
.ro-score-denom { font-size: 18px; color: #aaa; }
.ro-verdict { font-size: 12px; margin-top: 4px; }
.ro-red   { color: #C1440E; }
.ro-amber { color: #BA7517; }
.ro-green { color: #3B6D11; }
.ro-badge {
    display: inline-block; font-size: 10px;
    padding: 3px 9px; border-radius: 20px; margin: 2px;
}
.ro-badge-red   { background:#FCEBEB; color:#A32D2D; }
.ro-badge-amber { background:#FAEEDA; color:#854F0B; }
.ro-badge-green { background:#EAF3DE; color:#3B6D11; }
.ro-badge-blue  { background:#EBF3FA; color:#185FA5; }
.ro-tl {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 7px 10px; border-radius: 7px;
    background: #f9fafb; margin-bottom: 5px;
    font-size: 12px;
}
.ro-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; margin-right: 8px; flex-shrink: 0;
}
.ro-dot-r { background:#E24B4A; }
.ro-dot-a { background:#EF9F27; }
.ro-dot-g { background:#639922; }
.ro-cite {
    font-family: monospace; font-size: 11px;
    line-height: 1.7; color: #555;
    background: #f9fafb;
    border-left: 2px solid #1B6CA8;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin-top: 8px; word-break: break-word;
}
.ro-footer {
    text-align: center; margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 0.5px solid #e5e7eb;
    font-size: 11px; color: #bbb;
    font-family: 'Instrument Serif', serif;
    font-style: italic;
}
.stFileUploader > div {
    border: 1.5px dashed #c5d9ec !important;
    border-radius: 12px !important;
    background: #f7fbff !important;
}
.stButton > button {
    background: #1B6CA8 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 400 !important;
    padding: 0.5rem 1.25rem !important; width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ro-nav">
  <div class="ro-logo">RadhaOmics<span>.</span></div>
  <div class="ro-nav-right">FDA SABV &nbsp;·&nbsp; NIH SABV &nbsp;·&nbsp; Open source</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ro-pill">● Tissue-specific sex-bias detection</div>', unsafe_allow_html=True)
st.markdown('<h1 class="ro-h1">Your omics data might be<br><em>missing</em> half the picture.</h1>', unsafe_allow_html=True)
st.markdown('<p class="ro-sub">RadhaOmics scans your dataset for sex bias and computes the BiasΔ score — the first tissue-aware metric that quantifies exactly how much your biological conclusions are distorted.</p>', unsafe_allow_html=True)

st.markdown('<div class="ro-section-label">Step 1 — Configure your analysis</div>', unsafe_allow_html=True)
cfg1, cfg2 = st.columns(2)
with cfg1:
    tissue = st.selectbox(
        "Tissue type",
        list(GTEX_TISSUE_WEIGHTS.keys()),
        index=list(GTEX_TISSUE_WEIGHTS.keys()).index("blood"),
        help="RadhaOmics uses GTEx tissue-specific sex-bias profiles to weight the BiasΔ score."
    )
with cfg2:
    organism = st.selectbox(
        "Organism",
        ["Human (Homo sapiens)", "Mouse (Mus musculus)", "Rat (Rattus norvegicus)", "Other"],
        help="Currently optimised for human datasets. Mouse and rat support coming in v1.1."
    )

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown('<div class="ro-section-label">Step 2 — Upload your dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv","tsv","txt"],
    label_visibility="collapsed",
    help="CSV or TSV with a sex/gender column and gene expression columns"
)
st.markdown("<div style='font-size:11px;color:#aaa;margin-top:4px;margin-bottom:8px;'>Accepts CSV, TSV · RNA-seq · GWAS · Proteomics · Single-cell · Microarray · max 200MB</div>", unsafe_allow_html=True)
load_example = st.button("Load example RNA-seq dataset →")

if load_example or uploaded_file is not None:
    if load_example:
        np.random.seed(42)
        n = 200
        sex = (["male"]*156) + (["female"]*44)
        np.random.shuffle(sex)
        genes = {
            "XIST":   [0.1 if s=="male" else 8.2+np.random.normal(0,.5) for s in sex],
            "DDX3Y":  [7.4+np.random.normal(0,.4) if s=="male" else 0.2 for s in sex],
            "KDM5D":  [6.1+np.random.normal(0,.6) if s=="male" else 0.3 for s in sex],
            "TP53":   [4.2+np.random.normal(0,1.0) for _ in sex],
            "BRCA1":  [3.8+np.random.normal(0,.8) for _ in sex],
            "RPS4Y1": [5.2+np.random.normal(0,.5) if s=="male" else 0.4 for s in sex],
            "GAPDH":  [8.1+np.random.normal(0,.3) for _ in sex],
        }
        df = pd.DataFrame({"sample_id": range(n), "sex": sex, **genes})
        st.info(f"Example dataset loaded — {n} samples, RNA-seq simulation, human blood")
    else:
        sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success(f"Dataset loaded — {df.shape[0]} samples · {df.shape[1]} columns")

    sex_col = detect_sex_column(df)
    if not sex_col:
        st.error("Could not find a sex/gender column. Please include a column named 'sex' or 'gender'.")
        st.stop()

    ratio = compute_sex_ratio(df, sex_col)
    flags = flag_sex_biased_genes(df)
    power = compute_statistical_power(ratio["female"])
    bd    = compute_biasdelta(ratio, flags, tissue)

    st.divider()
    st.markdown('<div class="ro-section-label">Step 3 — Results</div>', unsafe_allow_html=True)

    score_cls = ("ro-red" if bd["fairness_score"] < 41 else
                 "ro-amber" if bd["fairness_score"] < 71 else "ro-green")

    st.markdown(f"""
    <div class="ro-card">
      <div class="ro-section-label">Fairness score</div>
      <div class="ro-score-num {score_cls}">{bd['fairness_score']}<span class="ro-score-denom">/100</span></div>
      <div class="ro-verdict {score_cls}">{bd['verdict']} · BiasΔ = {bd['biasdelta']}</div>
      <div style="margin-top:12px;">
        <span class="ro-badge ro-badge-{'red' if ratio['ratio_flag'] else 'green'}">M:F {ratio['mf_ratio']}:1</span>
        <span class="ro-badge ro-badge-{'red' if flags else 'green'}">{len(flags)} gene{'s' if len(flags)!=1 else ''} flagged</span>
        <span class="ro-badge ro-badge-{'amber' if power['power_flag'] else 'green'}">Female power {round(power['achieved_power']*100)}%</span>
        <span class="ro-badge ro-badge-blue">{tissue} · {organism.split()[0]}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns([1,1])
    with r1:
        st.markdown('<div class="ro-card"><div class="ro-section-label">Sample distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Male","Female"],
            values=[ratio["male"], ratio["female"]],
            hole=0.58,
            marker_colors=["#1B6CA8","#C1440E"],
            textinfo="label+percent",
            textfont=dict(family="Inter",size=11),
        ))
        fig.update_layout(
            height=200, margin=dict(t=0,b=0,l=0,r=0),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"<b>{ratio['total']}</b><br>samples",
                x=0.5, y=0.5, font_size=13,
                font_family="Inter", showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="font-size:12px;color:#555;margin-top:-8px;">
          <b>{ratio['male']}</b> male ({ratio['male_pct']}%) &nbsp;·&nbsp;
          <b>{ratio['female']}</b> female ({ratio['female_pct']}%)<br>
          <span style="color:#C1440E;">M:F ratio {ratio['mf_ratio']}:1
          {'— exceeds 1.5:1 threshold' if ratio['ratio_flag'] else '— within threshold'}</span>
        </div></div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="ro-card"><div class="ro-section-label">Compliance checklist</div>', unsafe_allow_html=True)
        checks = [
            ("Sex ratio ≤ 1.5:1",      not ratio["ratio_flag"], f"{ratio['mf_ratio']}:1"),
            ("FDA SABV compliance",     not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass"),
            ("NIH SABV policy",         not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass"),
            ("Female power ≥ 80%",      not power["power_flag"], f"{round(power['achieved_power']*100)}%"),
            ("Sex metadata present",    True,                    "Pass"),
            ("Sample size ≥ 100",       ratio["total"] >= 100,   str(ratio["total"])),
        ]
        for label, ok, val in checks:
            dot = "ro-dot-g" if ok else "ro-dot-r"
            val_color = "#3B6D11" if ok else "#C1440E"
            st.markdown(f"""
            <div class="ro-tl">
              <span><span class="ro-dot {dot}"></span>{label}</span>
              <span style="font-weight:500;color:{val_color};font-size:11px;">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ro-card"><div class="ro-section-label">Sex-biased genes detected</div>', unsafe_allow_html=True)
    if flags:
        flag_df = pd.DataFrame(flags)
        flag_df.columns = ["Gene","Chromosome","Bias direction","Risk"]
        st.dataframe(flag_df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="font-size:12px;color:#3B6D11;">No known sex-biased genes detected.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ro-card"><div class="ro-section-label">Methods section citation</div>', unsafe_allow_html=True)
    citation = (
        f"Sex bias analysis was performed using RadhaOmics v1.0 "
        f"(github.com/KirtanaPrem/RadhaOmics). "
        f"Dataset (n={ratio['total']}) showed M:F ratio of {ratio['mf_ratio']}:1, "
        f"fairness score {bd['fairness_score']}/100, tissue: {tissue}, "
        f"organism: {organism}. "
        f"{len(flags)} sex-biased genes flagged. "
        f"Female statistical power: {round(power['achieved_power']*100)}% (target 80%, α=0.05)."
    )
    st.markdown(f'<div class="ro-cite">{citation}</div>', unsafe_allow_html=True)
    st.code(citation, language=None)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.divider()
    st.markdown('<div class="ro-section-label">What RadhaOmics checks</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        ("Sex ratio audit",     "M:F threshold checking against NIH and FDA guidelines"),
        ("BiasΔ score",         "First tissue-aware metric — novel contribution"),
        ("Statistical power",   "Female sample adequacy for 80% power at α=0.05"),
        ("Gene-level flags",    "chrX/Y and autosomal sex-biased gene detection"),
        ("Compliance check",    "FDA SABV and NIH SABV traffic-light report"),
        ("Citation generator",  "Ready-to-paste methods section text"),
    ]
    for i, (title, desc) in enumerate(features):
        col = [f1, f2, f3][i % 3]
        with col:
            st.markdown(f'<div class="ro-card"><div class="ro-section-label">{title}</div><div style="font-size:12px;color:#666;line-height:1.6;">{desc}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="ro-footer">Named for Anuradha — because women deserve to be in the data.</div>', unsafe_allow_html=True)
