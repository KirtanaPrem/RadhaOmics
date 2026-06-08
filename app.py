import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from radhaomics import (
    compute_sex_ratio, flag_sex_biased_genes,
    compute_statistical_power, compute_biasdelta,
    detect_sex_column, GTEX_TISSUE_WEIGHTS,
)

st.set_page_config(page_title="RadhaOmics", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container { padding: 2rem 2.5rem; }

.ro-nav {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 0.5px solid #e5e7eb;
    margin-bottom: 2rem;
}
.ro-logo { font-size: 17px; font-weight: 500; letter-spacing: -0.02em; color: #111; }
.ro-logo span { color: #1B6CA8; }
.ro-tagline { font-size: 12px; color: #888; font-weight: 300; }

.ro-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: #1B6CA8; font-weight: 400;
    letter-spacing: 0.04em; margin-bottom: 1rem;
    background: #EBF3FA; padding: 4px 12px; border-radius: 20px;
}

.ro-h1 {
    font-size: 32px; font-weight: 300;
    letter-spacing: -0.03em; line-height: 1.15;
    margin-bottom: 0.75rem; color: #111;
}
.ro-h1 em {
    font-family: 'Instrument Serif', serif;
    font-style: italic; color: #1B6CA8;
}

.ro-sub {
    font-size: 13px; color: #666;
    font-weight: 300; line-height: 1.7;
    margin-bottom: 1.5rem; max-width: 480px;
}

.ro-card {
    background: #fff;
    border: 0.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.ro-metric-label {
    font-size: 11px; color: #888;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.ro-metric-val {
    font-size: 28px; font-weight: 500;
    letter-spacing: -0.04em; line-height: 1;
}
.ro-metric-sub { font-size: 11px; margin-top: 4px; }

.ro-red   { color: #C1440E; }
.ro-amber { color: #BA7517; }
.ro-green { color: #3B6D11; }
.ro-blue  { color: #1B6CA8; }

.ro-badge {
    display: inline-block; font-size: 10px;
    padding: 3px 10px; border-radius: 20px; margin: 2px;
}
.ro-badge-red   { background:#FCEBEB; color:#A32D2D; }
.ro-badge-amber { background:#FAEEDA; color:#854F0B; }
.ro-badge-green { background:#EAF3DE; color:#3B6D11; }
.ro-badge-blue  { background:#EBF3FA; color:#185FA5; }

.ro-tl-row {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 8px 12px; border-radius: 7px;
    background: #f9fafb; margin-bottom: 6px;
    font-size: 12px;
}
.ro-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 8px;
}
.ro-dot-red   { background: #E24B4A; }
.ro-dot-amber { background: #EF9F27; }
.ro-dot-green { background: #639922; }

.ro-cite {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; line-height: 1.7; color: #555;
    background: #f9fafb;
    border-left: 2px solid #1B6CA8;
    padding: 10px 14px; border-radius: 0 6px 6px 0;
    margin-top: 8px;
}

.ro-footer {
    text-align: center; margin-top: 2.5rem;
    padding-top: 1.25rem; border-top: 0.5px solid #e5e7eb;
    font-size: 11px; color: #aaa;
    font-family: 'Instrument Serif', serif; font-style: italic;
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
    padding: 0.5rem 1.25rem !important;
}
.stButton > button:hover { background: #185FA5 !important; }
section[data-testid="stSidebar"] {
    background: #f7fbff !important;
    border-right: 0.5px solid #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# ── Nav ───────────────────────────────────────────────────
st.markdown("""
<div class="ro-nav">
  <div class="ro-logo">RadhaOmics<span>.</span></div>
  <div class="ro-tagline">FDA SABV · NIH SABV · Open source</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:15px;font-weight:500;margin-bottom:4px;'>RadhaOmics</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#888;font-style:italic;margin-bottom:1rem;'>Tissue-specific sex-bias detection</div>", unsafe_allow_html=True)
    st.divider()
    tissue = st.selectbox("Tissue type", list(GTEX_TISSUE_WEIGHTS.keys()),
                          index=list(GTEX_TISSUE_WEIGHTS.keys()).index("blood"))
    st.divider()
    st.markdown("<div style='font-size:12px;color:#555;line-height:1.6;'>RadhaOmics computes the <b>BiasΔ score</b> — the first tissue-aware metric quantifying how sex imbalance distorts your biological conclusions.</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:11px;color:#aaa;font-style:italic;'>Named for Anuradha —<br>because women deserve<br>to be in the data.</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown('<div class="ro-eyebrow">● FDA SABV · NIH Sex as a Biological Variable</div>', unsafe_allow_html=True)
st.markdown('<h1 class="ro-h1">Your omics data might be<br><em>missing</em> half the picture.</h1>', unsafe_allow_html=True)
st.markdown('<p class="ro-sub">RadhaOmics scans genomic and transcriptomic datasets for sex bias — computing the BiasΔ score to quantify exactly how much your conclusions are affected, tissue by tissue.</p>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)",
                                  type=["csv","tsv","txt"],
                                  label_visibility="collapsed")
col_ex, col_sp = st.columns([1,5])
with col_ex:
    load_example = st.button("Load example dataset")

st.markdown("<div style='margin-top:0.5rem;font-size:11px;color:#aaa;'>Accepts CSV, TSV · RNA-seq, GWAS, Proteomics, Single-cell, Microarray</div>", unsafe_allow_html=True)

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
        st.info("Example dataset loaded — 200 samples, RNA-seq simulation")
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

    st.divider()

    # ── Score + Donut row ─────────────────────────────────
    sc1, sc2 = st.columns([1, 1])

    with sc1:
        score_color = ("ro-red" if bd["fairness_score"] < 41 else
                       "ro-amber" if bd["fairness_score"] < 71 else "ro-green")
        st.markdown(f"""
        <div class="ro-card">
          <div class="ro-metric-label">Fairness score</div>
          <div class="ro-metric-val {score_color}">{bd['fairness_score']}<span style="font-size:16px;">/100</span></div>
          <div class="ro-metric-sub {score_color}">{bd['verdict']}</div>
          <div style="margin-top:12px;">
            <span class="ro-badge ro-badge-{'red' if ratio['ratio_flag'] else 'green'}">
              M:F {ratio['mf_ratio']}:1 {'— above threshold' if ratio['ratio_flag'] else '— OK'}
            </span>
            <span class="ro-badge ro-badge-{'red' if flags else 'green'}">
              {len(flags)} genes flagged
            </span>
            <span class="ro-badge ro-badge-{'amber' if power['power_flag'] else 'green'}">
              Female power {round(power['achieved_power']*100)}%
            </span>
            <span class="ro-badge ro-badge-blue">{tissue}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Compliance checklist
        st.markdown('<div class="ro-card">', unsafe_allow_html=True)
        checks = [
            ("Sex ratio ≤ 1.5:1",          not ratio["ratio_flag"], f"{ratio['mf_ratio']}:1"),
            ("FDA SABV compliance",         not ratio["ratio_flag"], "Fails" if ratio["ratio_flag"] else "Pass"),
            ("NIH SABV policy",             not ratio["ratio_flag"], "Fails" if ratio["ratio_flag"] else "Pass"),
            ("Female power ≥ 80%",          not power["power_flag"], f"{round(power['achieved_power']*100)}%"),
            ("Sex metadata recorded",       True,                    "Pass"),
            ("Sample size ≥ 100",           ratio["total"] >= 100,   str(ratio["total"])),
        ]
        for label, ok, val in checks:
            dot = "ro-dot-green" if ok else ("ro-dot-amber" if "%" in val else "ro-dot-red")
            st.markdown(f"""
            <div class="ro-tl-row">
              <span><span class="ro-dot {dot}"></span>{label}</span>
              <span style="font-weight:500;font-size:11px;color:{'#3B6D11' if ok else '#C1440E'};">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sc2:
        # Donut
        fig_donut = go.Figure(go.Pie(
            labels=["Male","Female"],
            values=[ratio["male"], ratio["female"]],
            hole=0.6,
            marker_colors=["#1B6CA8","#C1440E"],
            textinfo="label+percent",
            textfont=dict(family="Inter", size=12),
        ))
        fig_donut.update_layout(
            height=220, margin=dict(t=0,b=0,l=0,r=0),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"{ratio['total']}<br><span style='font-size:10px'>samples</span>",
                              x=0.5, y=0.5, font_size=15,
                              font_family="Inter", showarrow=False)]
        )
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Sample sex distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # BiasΔ bar
        fig_bar = go.Figure(go.Bar(
            x=["Tissue weight", "Imbalance", "Gene flags"],
            y=[bd["tissue_weight"], bd["imbalance"], bd["flag_penalty"]],
            marker_color=["#1B6CA8","#C1440E","#BA7517"],
            text=[round(bd["tissue_weight"],3), round(bd["imbalance"],3), round(bd["flag_penalty"],3)],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=220, margin=dict(t=20,b=0,l=0,r=0),
            yaxis_title="Contribution to BiasΔ",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11),
            yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.markdown('<div class="ro-card"><div class="ro-metric-label">BiasΔ score breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Gene flags ────────────────────────────────────────
    if flags:
        st.divider()
        st.markdown('<div class="ro-metric-label" style="font-size:12px;font-weight:500;color:#111;margin-bottom:8px;">Sex-biased genes detected</div>', unsafe_allow_html=True)
        flag_df = pd.DataFrame(flags)
        flag_df.columns = ["Gene","Chromosome","Bias direction","Risk"]
        st.dataframe(flag_df, use_container_width=True, hide_index=True)

    # ── Citation ──────────────────────────────────────────
    st.divider()
    st.markdown('<div class="ro-metric-label" style="font-size:12px;font-weight:500;color:#111;margin-bottom:4px;">Methods section citation</div>', unsafe_allow_html=True)
    citation = (f"Sex bias analysis was performed using RadhaOmics v1.0 "
                f"(github.com/KirtanaPrem/RadhaOmics). "
                f"Dataset (n={ratio['total']}) showed M:F ratio of {ratio['mf_ratio']}:1, "
                f"fairness score {bd['fairness_score']}/100, tissue: {tissue}. "
                f"{len(flags)} sex-biased genes flagged. "
                f"Female statistical power: {round(power['achieved_power']*100)}%.")
    st.markdown(f'<div class="ro-cite">{citation}</div>', unsafe_allow_html=True)
    st.code(citation, language=None)

else:
    # Landing state features grid
    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Sex ratio audit</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">Detects M:F imbalance and flags datasets exceeding NIH/FDA thresholds</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Gene-level flags</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">Identifies sex-biased genes used incorrectly as universal markers</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ro-card"><div class="ro-metric-label">BiasΔ score — novel</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">First tissue-aware metric quantifying how bias propagates into your conclusions</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Compliance checklist</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">FDA SABV and NIH SABV compliance check with traffic-light reporting</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Statistical power</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">Checks if female sample count achieves 80% power for reliable conclusions</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ro-card"><div class="ro-metric-label">Citation generator</div><div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6;">Auto-generates a methods section citation ready to paste into your paper</div></div>', unsafe_allow_html=True)

st.markdown('<div class="ro-footer">Named for Anuradha — because women deserve to be in the data.</div>', unsafe_allow_html=True)
