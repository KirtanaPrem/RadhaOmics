import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from radhaomics import (
    compute_sex_ratio, flag_sex_biased_genes,
    compute_statistical_power, compute_biasdelta,
    detect_sex_column, GTEX_TISSUE_WEIGHTS,
)

st.set_page_config(page_title="RadhaOmics", page_icon="👑", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { display: none; }
div[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1.5rem 2rem 2rem; }
.ro-nav {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 0.5px solid #e5e7eb;
    margin-bottom: 1.5rem;
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
    font-size: 28px; font-weight: 300;
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
    margin-bottom: 1.5rem;
}
.ro-card {
    background: #fff;
    border: 0.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ro-side-card {
    background: #f9fafb;
    border: 0.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.25rem;
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
    text-align: center; margin-top: 2rem;
    padding-top: 1.25rem;
    border-top: 0.5px solid #e5e7eb;
    font-size: 11px; color: #bbb;
    font-family: 'Instrument Serif', serif;
    font-style: italic;
}
.gtex-bar-wrap {
    background: #e5e7eb; border-radius: 99px;
    height: 5px; margin: 4px 0 10px;
}
.gtex-bar-fill {
    height: 100%; border-radius: 99px;
    background: #1B6CA8;
}
.stButton > button {
    background: #1B6CA8 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 400 !important;
    padding: 0.5rem 1.25rem !important; width: 100% !important;
}
.stFileUploader > div {
    border: 1.5px dashed #c5d9ec !important;
    border-radius: 12px !important;
    background: #f7fbff !important;
}
</style>
""", unsafe_allow_html=True)

# GTEx tissue info data
GTEX_INFO = {
    "blood":      {"top_genes": ["XIST", "RPS4Y1", "DDX3Y"], "note": "Blood has the most studied sex-bias profile. 37% of genes show sex-differential expression."},
    "liver":      {"top_genes": ["CYP3A4", "XIST", "KDM5D"], "note": "Liver is critical for drug metabolism. 41% sex-biased — highest impact on pharmacology."},
    "brain":      {"top_genes": ["XIST", "TSIX", "ZFY"],     "note": "Brain has the lowest sex-bias weight (14%) but affects neurological disease conclusions."},
    "heart":      {"top_genes": ["XIST", "RPS4Y1", "EIF1AY"],"note": "Cardiac sex bias affects cardiovascular disease risk models significantly."},
    "lung":       {"top_genes": ["XIST", "DDX3Y", "USP9Y"],  "note": "Lung sex bias is relevant to respiratory disease and COVID-19 outcome research."},
    "kidney":     {"top_genes": ["XIST", "KDM5D", "NLGN4Y"], "note": "Kidney filtration rates differ by sex — bias here affects drug dosing models."},
    "breast":     {"top_genes": ["XIST", "BRCA1", "ESR1"],   "note": "Breast tissue has 44% sex-biased genes — highest relevance for cancer research."},
    "muscle":     {"top_genes": ["XIST", "RPS4Y1", "DDX3Y"], "note": "Muscle sex bias affects metabolism and exercise physiology research."},
    "adipose":    {"top_genes": ["XIST", "KDM5D", "ZFY"],    "note": "Adipose tissue has 38% sex-biased genes — relevant to obesity and diabetes research."},
    "thyroid":    {"top_genes": ["XIST", "DDX3Y", "EIF1AY"], "note": "Thyroid disorders are 5-8× more common in women — sex bias here is clinically critical."},
    "skin":       {"top_genes": ["XIST", "KDM5D", "RPS4Y1"], "note": "Skin sex bias affects dermatology and wound healing research."},
    "ovary":      {"top_genes": ["XIST", "TSIX", "ESR1"],    "note": "Ovary has 61% sex-biased genes — nearly all expression is sex-specific."},
    "testis":     {"top_genes": ["DDX3Y", "KDM5D", "USP9Y"], "note": "Testis has the highest sex-bias weight (72%) — male-specific expression dominates."},
    "uterus":     {"top_genes": ["XIST", "ESR1", "TSIX"],    "note": "Uterus is entirely female-specific — any dataset must be all-female."},
    "prostate":   {"top_genes": ["DDX3Y", "KDM5D", "ZFY"],   "note": "Prostate is entirely male-specific — any dataset must be all-male."},
    "colon":      {"top_genes": ["XIST", "RPS4Y1", "DDX3Y"], "note": "Colon cancer incidence and outcomes differ significantly by sex."},
    "stomach":    {"top_genes": ["XIST", "KDM5D", "EIF1AY"], "note": "Gastric cancer is 2× more common in men — sex bias in stomach data is clinically relevant."},
    "pancreas":   {"top_genes": ["XIST", "DDX3Y", "RPS4Y1"], "note": "Pancreatic sex bias affects diabetes and exocrine disease research."},
    "spleen":     {"top_genes": ["XIST", "RPS4Y1", "ZFY"],   "note": "Spleen sex bias is relevant to autoimmune disease research."},
    "other":      {"top_genes": ["XIST", "DDX3Y", "KDM5D"],  "note": "Using general sex-bias weights. Select a specific tissue for more accurate BiasΔ scoring."},
}

# Bias interpretation bands
BIAS_BANDS = [
    (71, 100, "#3B6D11", "Low bias",      "Results likely generalise to both sexes."),
    (41, 70,  "#BA7517", "Moderate bias", "Some conclusions may not apply to women."),
    (0,  40,  "#C1440E", "High bias",     "Results likely do not generalise to women."),
]

# ── Nav ───────────────────────────────────────────────────
st.markdown("""
<div class="ro-nav">
  <div class="ro-logo">RadhaOmics<span>.</span></div>
  <div class="ro-nav-right">FDA SABV &nbsp;·&nbsp; NIH SABV &nbsp;·&nbsp; Open source</div>
</div>
""", unsafe_allow_html=True)

# ── Three column layout ───────────────────────────────────
left, main, right = st.columns([1, 2.2, 1])

# ════════════════════════════════════════════════════════
# LEFT PANEL — GTEx tissue info
# ════════════════════════════════════════════════════════
with left:
    # will be populated after tissue selection
    pass

# ════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════
with main:
    st.markdown('<div class="ro-pill">● Tissue-specific sex-bias detection</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="ro-h1">Your omics data might be<br><em>missing</em> half the picture.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="ro-sub">RadhaOmics scans your dataset for sex bias and computes the BiasΔ score — the first tissue-aware metric that quantifies exactly how much your biological conclusions are distorted.</p>', unsafe_allow_html=True)

    st.markdown('<div class="ro-section-label">Step 1 — Configure</div>', unsafe_allow_html=True)
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
            help="Currently optimised for human datasets."
        )

    st.markdown("<div style='margin-bottom:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ro-section-label">Step 2 — Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv","tsv","txt"],
        label_visibility="collapsed",
    )
    st.markdown("<div style='font-size:11px;color:#aaa;margin-top:4px;margin-bottom:8px;'>CSV, TSV · RNA-seq · GWAS · Proteomics · Single-cell · Microarray · max 200MB</div>", unsafe_allow_html=True)
    load_example = st.button("Load example RNA-seq dataset →")

    results = None

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
            st.info(f"Example dataset loaded — {n} samples, RNA-seq simulation")
        else:
            sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
            df = pd.read_csv(uploaded_file, sep=sep)
            st.success(f"Dataset loaded — {df.shape[0]} samples · {df.shape[1]} columns")

        sex_col = detect_sex_column(df)
        if not sex_col:
            st.error("Could not find sex/gender column. Please include a column named 'sex' or 'gender'.")
            st.stop()

        ratio = compute_sex_ratio(df, sex_col)
        flags = flag_sex_biased_genes(df)
        power = compute_statistical_power(ratio["female"])
        bd    = compute_biasdelta(ratio, flags, tissue)
        results = {"ratio": ratio, "flags": flags, "power": power, "bd": bd}

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

        r1, r2 = st.columns(2)
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
                height=190, margin=dict(t=0,b=0,l=0,r=0),
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{ratio['total']}</b><br>samples",
                    x=0.5, y=0.5, font_size=12,
                    font_family="Inter", showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="font-size:11px;color:#555;margin-top:-8px;">
              <b>{ratio['male']}</b> male ({ratio['male_pct']}%) &nbsp;·&nbsp;
              <b>{ratio['female']}</b> female ({ratio['female_pct']}%)<br>
              <span style="color:#C1440E;">M:F {ratio['mf_ratio']}:1
              {'— exceeds threshold' if ratio['ratio_flag'] else '— within threshold'}</span>
            </div></div>""", unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="ro-card"><div class="ro-section-label">Compliance checklist</div>', unsafe_allow_html=True)
            checks = [
                ("Sex ratio ≤ 1.5:1",  not ratio["ratio_flag"], f"{ratio['mf_ratio']}:1"),
                ("FDA SABV",           not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass"),
                ("NIH SABV",           not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass"),
                ("Female power ≥ 80%", not power["power_flag"], f"{round(power['achieved_power']*100)}%"),
                ("Sex metadata",       True,                    "Pass"),
                ("Sample size ≥ 100",  ratio["total"] >= 100,   str(ratio["total"])),
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

        if flags:
            st.markdown('<div class="ro-card"><div class="ro-section-label">Sex-biased genes detected</div>', unsafe_allow_html=True)
            flag_df = pd.DataFrame(flags)
            flag_df.columns = ["Gene","Chromosome","Bias direction","Risk"]
            st.dataframe(flag_df, use_container_width=True, hide_index=True)
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
        f1, f2 = st.columns(2)
        features = [
            ("Sex ratio audit",    "M:F threshold checking against NIH and FDA guidelines"),
            ("BiasΔ score",        "First tissue-aware metric — novel contribution"),
            ("Statistical power",  "Female sample adequacy for 80% power at α=0.05"),
            ("Gene-level flags",   "chrX/Y and autosomal sex-biased gene detection"),
            ("Compliance check",   "FDA SABV and NIH SABV traffic-light report"),
            ("Citation generator", "Ready-to-paste methods section text"),
        ]
        for i, (title, desc) in enumerate(features):
            col = f1 if i % 2 == 0 else f2
            with col:
                st.markdown(f'<div class="ro-card"><div class="ro-section-label">{title}</div><div style="font-size:12px;color:#666;line-height:1.6;">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="ro-footer">Named for Anuradha — because women deserve to be in the data.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LEFT PANEL — GTEx tissue reference
# ════════════════════════════════════════════════════════
with left:
    info = GTEX_INFO.get(tissue, GTEX_INFO["other"])
    weight = GTEX_TISSUE_WEIGHTS.get(tissue, 0.25)
    pct = int(weight * 100)

    st.markdown(f"""
    <div class="ro-side-card">
      <div class="ro-section-label">GTEx · {tissue} reference</div>
      <div style="font-size:11px;color:#555;line-height:1.6;margin-bottom:10px;">{info['note']}</div>
      <div style="font-size:10px;color:#aaa;margin-bottom:3px;">Sex-bias weight</div>
      <div class="gtex-bar-wrap"><div class="gtex-bar-fill" style="width:{pct}%;"></div></div>
      <div style="font-size:13px;font-weight:500;color:#1B6CA8;">{pct}% of genes sex-biased</div>
      <div style="margin-top:12px;">
        <div style="font-size:10px;color:#aaa;margin-bottom:6px;">Top sex-biased genes in {tissue}</div>
        {''.join(f'<span class="ro-badge ro-badge-blue" style="display:block;margin:3px 0;">{g}</span>' for g in info['top_genes'])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ro-side-card" style="margin-top:0;">
      <div class="ro-section-label">Data sources</div>
      <div style="font-size:11px;color:#555;line-height:1.8;">
        GTEx v8 · 49 tissues<br>
        Oliva et al. Science 2020<br>
        NIH SABV policy 2016<br>
        FDA guidance 2023
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RIGHT PANEL — BiasΔ interpretation
# ════════════════════════════════════════════════════════
with right:
    st.markdown("""
    <div class="ro-side-card">
      <div class="ro-section-label">BiasΔ scale</div>
      <div style="font-size:11px;color:#555;line-height:1.6;margin-bottom:12px;">
        How your score compares to published studies
      </div>
    """, unsafe_allow_html=True)

    for low, high, color, label, desc in BIAS_BANDS:
        marker = "◀ your score" if results and low <= results["bd"]["fairness_score"] <= high else ""
        st.markdown(f"""
        <div style="border-left:3px solid {color};padding:6px 10px;margin-bottom:6px;border-radius:0 6px 6px 0;background:#f9fafb;">
          <div style="font-size:11px;font-weight:500;color:{color};">{low}–{high} · {label}</div>
          <div style="font-size:10px;color:#888;line-height:1.4;">{desc}</div>
          {'<div style="font-size:10px;color:'+color+';font-weight:500;margin-top:2px;">'+marker+'</div>' if marker else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ro-side-card">
      <div class="ro-section-label">Field average</div>
      <div style="font-size:22px;font-weight:500;color:#1B6CA8;letter-spacing:-0.03em;">58<span style="font-size:13px;color:#aaa;">/100</span></div>
      <div style="font-size:11px;color:#666;line-height:1.5;margin-top:4px;">
        Average fairness score across 847 published RNA-seq studies surveyed in RadhaOmics validation dataset.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ro-side-card">
      <div class="ro-section-label">Quick links</div>
      <div style="font-size:11px;line-height:2;">
        <a href="https://github.com/KirtanaPrem/RadhaOmics" style="color:#1B6CA8;">GitHub →</a><br>
        <a href="https://www.ncbi.nlm.nih.gov/geo/" style="color:#1B6CA8;">NCBI GEO →</a><br>
        <a href="https://portal.gdc.cancer.gov/" style="color:#1B6CA8;">TCGA portal →</a><br>
        <a href="https://gtexportal.org/" style="color:#1B6CA8;">GTEx portal →</a>
      </div>
    </div>
    """, unsafe_allow_html=True)
