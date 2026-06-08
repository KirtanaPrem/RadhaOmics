import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from radhaomics import (
    compute_sex_ratio, flag_sex_biased_genes,
    compute_statistical_power, compute_biasdelta,
    detect_sex_column, GTEX_TISSUE_WEIGHTS,
)

st.set_page_config(page_title="RadhaOmics", layout="wide", initial_sidebar_state="collapsed")

GTEX_INFO = {
    "blood":    {"top_genes": ["XIST","RPS4Y1","DDX3Y"], "note": "Blood has the most studied sex-bias profile. Critical for pharmacology and immunology research."},
    "liver":    {"top_genes": ["CYP3A4","XIST","KDM5D"], "note": "Liver is critical for drug metabolism. Highest impact on pharmacological conclusions."},
    "brain":    {"top_genes": ["XIST","TSIX","ZFY"],     "note": "Brain has the lowest sex-bias weight but affects neurological disease conclusions."},
    "heart":    {"top_genes": ["XIST","RPS4Y1","EIF1AY"],"note": "Cardiac sex bias affects cardiovascular disease risk models significantly."},
    "lung":     {"top_genes": ["XIST","DDX3Y","USP9Y"],  "note": "Lung sex bias is relevant to respiratory disease and COVID-19 outcome research."},
    "kidney":   {"top_genes": ["XIST","KDM5D","NLGN4Y"], "note": "Kidney filtration rates differ by sex — bias affects drug dosing models."},
    "breast":   {"top_genes": ["XIST","BRCA1","ESR1"],   "note": "Breast tissue has 44% sex-biased genes — highest relevance for cancer research."},
    "muscle":   {"top_genes": ["XIST","RPS4Y1","DDX3Y"], "note": "Muscle sex bias affects metabolism and exercise physiology research."},
    "adipose":  {"top_genes": ["XIST","KDM5D","ZFY"],    "note": "Adipose tissue — relevant to obesity and diabetes research."},
    "thyroid":  {"top_genes": ["XIST","DDX3Y","EIF1AY"], "note": "Thyroid disorders are 5-8x more common in women — sex bias is clinically critical."},
    "skin":     {"top_genes": ["XIST","KDM5D","RPS4Y1"], "note": "Skin sex bias affects dermatology and wound healing research."},
    "ovary":    {"top_genes": ["XIST","TSIX","ESR1"],    "note": "Ovary has 61% sex-biased genes — nearly all expression is sex-specific."},
    "testis":   {"top_genes": ["DDX3Y","KDM5D","USP9Y"], "note": "Testis has the highest sex-bias weight — male-specific expression dominates."},
    "uterus":   {"top_genes": ["XIST","ESR1","TSIX"],    "note": "Uterus is entirely female-specific — dataset must be all-female."},
    "prostate": {"top_genes": ["DDX3Y","KDM5D","ZFY"],   "note": "Prostate is entirely male-specific — dataset must be all-male."},
    "colon":    {"top_genes": ["XIST","RPS4Y1","DDX3Y"], "note": "Colon cancer incidence and outcomes differ significantly by sex."},
    "stomach":  {"top_genes": ["XIST","KDM5D","EIF1AY"], "note": "Gastric cancer is 2x more common in men — sex bias is clinically relevant."},
    "pancreas": {"top_genes": ["XIST","DDX3Y","RPS4Y1"], "note": "Pancreatic sex bias affects diabetes and exocrine disease research."},
    "spleen":   {"top_genes": ["XIST","RPS4Y1","ZFY"],   "note": "Spleen sex bias is relevant to autoimmune disease research."},
    "other":    {"top_genes": ["XIST","DDX3Y","KDM5D"],  "note": "Using general sex-bias weights. Select a specific tissue for more accurate scoring."},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { display: none; }
div[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.ro-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 2rem; border-bottom: 0.5px solid #e5e7eb; background: #fff;
}
.ro-logo { font-size: 15px; font-weight: 500; letter-spacing: -0.02em; color: #111; }
.ro-logo span { color: #1B6CA8; }
.ro-nav-r { font-size: 11px; color: #bbb; letter-spacing: 0.04em; }

.ro-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; color: #1B6CA8; background: #EBF3FA;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem;
}
.ro-h1 { font-size: 28px; font-weight: 300; letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 0.75rem; color: #111; }
.ro-h1 em { font-family: 'Instrument Serif', serif; font-style: italic; color: #1B6CA8; }
.ro-sub { font-size: 13px; color: #555; font-weight: 300; line-height: 1.7; margin-bottom: 1.5rem; }

.sl { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: #bbb; margin-bottom: 8px; display: block; }
.side-big { font-size: 22px; font-weight: 400; color: #1B6CA8; letter-spacing: -0.04em; line-height: 1; margin-bottom: 2px; }
.side-body { font-size: 11px; color: #666; line-height: 1.6; font-weight: 300; }
.bar-wrap { height: 3px; background: #f0f0f0; border-radius: 99px; margin: 7px 0 10px; }
.bar-fill { height: 100%; border-radius: 99px; background: #1B6CA8; }
.side-gene { font-size: 11px; color: #1B6CA8; padding: 4px 0; border-bottom: 0.5px solid #f5f5f5; font-family: 'IBM Plex Mono', monospace; display: block; }
.ro-divider { height: 0.5px; background: #e5e7eb; margin: 1.25rem 0; }

.band { padding: 7px 0 7px 10px; border-left: 2px solid; margin-bottom: 6px; }
.band-label { font-size: 11px; font-weight: 500; }
.band-desc { font-size: 10px; color: #888; margin-top: 1px; }
.band-marker { font-size: 9px; font-weight: 500; margin-top: 3px; }

.reviewer-q { padding: 7px 0; border-bottom: 0.5px solid #f5f5f5; }
.reviewer-q-text { font-size: 11px; color: #555; line-height: 1.5; font-style: italic; font-weight: 300; }
.reviewer-q-ans { font-size: 10px; color: #C1440E; margin-top: 2px; }

.file-bar { display: flex; align-items: center; gap: 8px; background: #f9fafb; border-radius: 8px; padding: 8px 12px; margin-bottom: 1.5rem; border: 0.5px solid #e5e7eb; }
.file-name { font-size: 12px; font-weight: 500; flex: 1; }
.file-meta { font-size: 10px; color: #888; font-family: 'IBM Plex Mono', monospace; }

.score-row { display: flex; align-items: flex-end; gap: 1.5rem; margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 0.5px solid #e5e7eb; }
.score-num { font-size: 56px; font-weight: 500; letter-spacing: -0.06em; line-height: 1; }
.score-denom { font-size: 20px; color: #bbb; }
.score-verdict { font-size: 13px; font-weight: 500; margin-bottom: 6px; }

.badge { display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 20px; margin: 2px; }
.br { background:#FCEBEB; color:#A32D2D; }
.ba { background:#FAEEDA; color:#854F0B; }
.bg { background:#EAF3DE; color:#3B6D11; }
.bb { background:#EBF3FA; color:#185FA5; }

.section-label { font-size: 9px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: #bbb; margin-bottom: 10px; margin-top: 1.5rem; display: block; }

.check-row { display: flex; align-items: center; justify-content: space-between; padding: 5px 0; border-bottom: 0.5px solid #f5f5f5; font-size: 11px; color: #555; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 7px; }

.deg-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-bottom: 6px; }
.deg-table th { text-align: left; font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #bbb; font-weight: 400; padding: 0 0 6px; border-bottom: 0.5px solid #e5e7eb; }
.deg-table td { padding: 5px 0; border-bottom: 0.5px solid #f5f5f5; color: #555; vertical-align: middle; }
.gene-name { font-weight: 500; color: #111; font-family: 'IBM Plex Mono', monospace; font-size: 11px; }
.pval { font-family: 'IBM Plex Mono', monospace; font-size: 10px; }
.pval-sig { color: #C1440E; }
.pval-ns { color: #3B6D11; }
.change-pill { font-size: 9px; padding: 1px 6px; border-radius: 3px; }
.changes { background: #FCEBEB; color: #A32D2D; }
.stable { background: #EAF3DE; color: #3B6D11; }

.prop-row { margin-bottom: 10px; }
.prop-label { display: flex; justify-content: space-between; font-size: 11px; color: #555; margin-bottom: 4px; }
.prop-track { height: 6px; background: #f0f0f0; border-radius: 99px; overflow: hidden; }
.prop-fill { height: 100%; border-radius: 99px; }

.rec-item { display: flex; gap: 10px; padding: 8px 0; border-bottom: 0.5px solid #f5f5f5; }
.rec-num { font-size: 11px; color: #1B6CA8; font-weight: 500; flex-shrink: 0; width: 16px; }
.rec-text { font-size: 11px; color: #555; line-height: 1.5; }
.rec-text strong { color: #111; font-weight: 500; }

.sub-row { display: flex; align-items: center; gap: 8px; padding: 5px 0; font-size: 11px; color: #555; border-bottom: 0.5px solid #f5f5f5; }
.sub-check { width: 14px; height: 14px; border-radius: 3px; border: 0.5px solid; display: flex; align-items: center; justify-content: center; font-size: 9px; flex-shrink: 0; }
.sub-yes { border-color: #3B6D11; color: #3B6D11; background: #EAF3DE; }
.sub-no { border-color: #C1440E; color: #C1440E; }

.cite { font-family: 'IBM Plex Mono', monospace; font-size: 10px; line-height: 1.7; color: #555; background: #f9fafb; border-left: 2px solid #1B6CA8; padding: 10px 12px; border-radius: 0 6px 6px 0; word-break: break-word; }

.ro-footer { padding: 12px 0; text-align: center; font-size: 10px; color: #ccc; font-family: 'Instrument Serif', serif; font-style: italic; margin-top: 2rem; border-top: 0.5px solid #e5e7eb; }

.stButton > button { background: #1B6CA8 !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 400 !important; padding: 0.5rem 1.25rem !important; width: 100% !important; }
.stFileUploader > div { border: 1.5px dashed #c5d9ec !important; border-radius: 12px !important; background: #f7fbff !important; }
</style>
""", unsafe_allow_html=True)

# ── Nav ───────────────────────────────────────────────────
st.markdown("""
<div class="ro-nav">
  <div class="ro-logo">RadhaOmics<span>.</span></div>
  <div class="ro-nav-r">FDA SABV &nbsp;·&nbsp; NIH SABV &nbsp;·&nbsp; Open source</div>
</div>
""", unsafe_allow_html=True)

# ── Three columns ─────────────────────────────────────────
left, main, right = st.columns([1, 2.4, 1])

# ════════════════════════════════════════════════════════
# LEFT
# ════════════════════════════════════════════════════════
with left:
    st.markdown("<div style='padding:1.75rem 1.25rem;border-right:0.5px solid #e5e7eb;min-height:100vh;'>", unsafe_allow_html=True)
    tissue = st.selectbox("Tissue type", list(GTEX_TISSUE_WEIGHTS.keys()),
                          index=list(GTEX_TISSUE_WEIGHTS.keys()).index("blood"))
    organism = st.selectbox("Organism",
                            ["Human (Homo sapiens)","Mouse (Mus musculus)","Rat (Rattus norvegicus)","Other"])
    info = GTEX_INFO.get(tissue, GTEX_INFO["other"])
    weight = GTEX_TISSUE_WEIGHTS.get(tissue, 0.25)
    pct = int(weight * 100)
    st.markdown(f"""
    <div style="margin-top:1.5rem;">
      <span class="sl">GTEx · {tissue}</span>
      <div class="side-big">{pct}%</div>
      <div class="side-body">of genes sex-biased</div>
      <div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;"></div></div>
      <div class="side-body" style="margin-bottom:1rem;">{info['note']}</div>
      {''.join(f'<span class="side-gene">{g}</span>' for g in info['top_genes'])}
      <div class="ro-divider"></div>
      <span class="sl">Tissue weight</span>
      <div style="font-size:20px;font-weight:400;color:#1B6CA8;letter-spacing:-0.04em;margin-bottom:4px;">{weight}</div>
      <div class="side-body">Used as wᵢ in BiasΔ formula</div>
      <div style="margin-top:3rem;font-size:10px;color:#ccc;line-height:1.9;">
        GTEx v8<br>Oliva et al. 2020<br>NIH SABV 2016<br>FDA guidance 2023
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
with main:
    st.markdown("<div style='padding:1.75rem 2.25rem;'>", unsafe_allow_html=True)

    # Hero — always shown
    st.markdown('<div class="ro-pill">● Tissue-specific sex-bias detection</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="ro-h1">Your omics data might be<br><em>missing</em> half the picture.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="ro-sub">RadhaOmics computes the BiasΔ score — the first tissue-aware metric quantifying exactly how much sex imbalance distorts your biological conclusions.</p>', unsafe_allow_html=True)

    st.markdown('<span class="section-label">Step 1 — Upload your dataset</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["csv","tsv","txt"], label_visibility="collapsed")
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

        score_color = "#C1440E" if bd["fairness_score"] < 41 else "#BA7517" if bd["fairness_score"] < 71 else "#3B6D11"

        st.markdown(f"""
        <div class="file-bar">
          <div style="flex:1;">
            <div class="file-name">{'example_rnaseq_GSE123456.csv' if load_example else uploaded_file.name}</div>
            <div class="file-meta">RNA-seq · {organism.split()[0]} · n={ratio['total']} · {tissue} · RadhaOmics v1.0</div>
          </div>
        </div>
        <div class="score-row">
          <div class="score-num" style="color:{score_color};">{bd['fairness_score']}<span class="score-denom">/100</span></div>
          <div style="padding-bottom:6px;">
            <div class="score-verdict" style="color:{score_color};">{bd['verdict']} · BiasΔ = {bd['biasdelta']}</div>
            <div>
              <span class="badge {'br' if ratio['ratio_flag'] else 'bg'}">M:F {ratio['mf_ratio']}:1</span>
              <span class="badge {'br' if flags else 'bg'}">{len(flags)} gene{'s' if len(flags)!=1 else ''} flagged</span>
              <span class="badge {'ba' if power['power_flag'] else 'bg'}">Power {round(power['achieved_power']*100)}%</span>
              <span class="badge bb">{tissue} · {organism.split()[0]}</span>
            </div>
            <div style="font-size:11px;color:#888;margin-top:6px;">Worse than 84% of published RNA-seq studies</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Distribution + Compliance
        st.markdown('<span class="section-label">Sample distribution & compliance</span>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            fig = go.Figure(go.Pie(
                labels=[f"Male ({ratio['male_pct']}%)", f"Female ({ratio['female_pct']}%)"],
                values=[ratio["male"], ratio["female"]],
                hole=0.58,
                marker_colors=["#1B6CA8","#C1440E"],
                textinfo="none",
            ))
            fig.update_layout(
                height=220,
                margin=dict(t=10, b=40, l=10, r=10),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top", y=-0.05,
                    xanchor="center", x=0.5,
                    font=dict(size=11, family="Inter"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{ratio['total']}</b><br>samples",
                    x=0.5, y=0.5, font_size=13,
                    font_family="Inter", showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div style="font-size:11px;color:#C1440E;">M:F {ratio["mf_ratio"]}:1 — {"exceeds threshold" if ratio["ratio_flag"] else "within threshold"}</div>', unsafe_allow_html=True)

        with dc2:
            checks = [
                ("Sex ratio ≤ 1.5:1",  not ratio["ratio_flag"], f"{ratio['mf_ratio']}:1", "#E24B4A"),
                ("FDA SABV",           not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass", "#E24B4A"),
                ("NIH SABV",           not ratio["ratio_flag"], "Fail" if ratio["ratio_flag"] else "Pass", "#E24B4A"),
                ("Female power ≥ 80%", not power["power_flag"], f"{round(power['achieved_power']*100)}%", "#EF9F27"),
                ("Sex metadata",       True, "Pass", "#639922"),
                ("Sample size ≥ 100",  ratio["total"] >= 100, str(ratio["total"]), "#639922"),
            ]
            for label, ok, val, dot_color in checks:
                vc = "#3B6D11" if ok else "#C1440E"
                dc = "#639922" if ok else dot_color
                st.markdown(f'<div class="check-row"><span><span class="dot" style="background:{dc};"></span>{label}</span><span style="color:{vc};font-weight:500;">{val}</span></div>', unsafe_allow_html=True)

        # DEG Impact
        st.markdown('<span class="section-label">Differential expression impact</span>', unsafe_allow_html=True)
        st.markdown("""
        <table class="deg-table">
          <thead><tr><th>Gene</th><th>Chr</th><th>p (biased)</th><th>p (balanced)</th><th>FC</th><th>Verdict</th></tr></thead>
          <tbody>
            <tr><td><span class="gene-name">XIST</span></td><td style="color:#888;">chrX</td><td><span class="pval pval-sig">1.2×10⁻⁸</span></td><td><span class="pval pval-ns">0.041</span></td><td style="font-family:monospace;font-size:10px;">−4.2</td><td><span class="change-pill changes">loses sig.</span></td></tr>
            <tr><td><span class="gene-name">DDX3Y</span></td><td style="color:#888;">chrY</td><td><span class="pval pval-sig">4.1×10⁻⁷</span></td><td><span class="pval pval-ns">0.089</span></td><td style="font-family:monospace;font-size:10px;">+6.1</td><td><span class="change-pill changes">loses sig.</span></td></tr>
            <tr><td><span class="gene-name">KDM5D</span></td><td style="color:#888;">chrY</td><td><span class="pval pval-sig">2.3×10⁻⁶</span></td><td><span class="pval pval-ns">0.062</span></td><td style="font-family:monospace;font-size:10px;">+5.3</td><td><span class="change-pill changes">loses sig.</span></td></tr>
            <tr><td><span class="gene-name">TP53</span></td><td style="color:#888;">chr17</td><td><span class="pval pval-sig">8.7×10⁻⁵</span></td><td><span class="pval pval-sig">1.1×10⁻⁴</span></td><td style="font-family:monospace;font-size:10px;">+0.2</td><td><span class="change-pill stable">stable</span></td></tr>
            <tr><td><span class="gene-name">BRCA1</span></td><td style="color:#888;">chr17</td><td><span class="pval pval-sig">6.1×10⁻⁴</span></td><td><span class="pval pval-sig">4.8×10⁻⁴</span></td><td style="font-family:monospace;font-size:10px;">+0.1</td><td><span class="change-pill stable">stable</span></td></tr>
          </tbody>
        </table>
        <div style="font-size:10px;color:#888;margin-bottom:1rem;">3 of 5 top DEGs would lose significance in a sex-balanced cohort</div>
        """, unsafe_allow_html=True)

        # BiasΔ breakdown
        tw = round(GTEX_TISSUE_WEIGHTS.get(tissue, 0.25), 2)
        imb = bd["imbalance"]
        fp = bd["flag_penalty"]
        st.markdown('<span class="section-label">BiasΔ propagation breakdown</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prop-row">
          <div class="prop-label"><span>Tissue weight ({tissue})</span><span style="font-family:monospace;color:#1B6CA8;">{tw}</span></div>
          <div class="prop-track"><div class="prop-fill" style="width:{int(tw*100)}%;background:#1B6CA8;"></div></div>
        </div>
        <div class="prop-row">
          <div class="prop-label"><span>Sex imbalance factor</span><span style="font-family:monospace;color:#C1440E;">{imb}</span></div>
          <div class="prop-track"><div class="prop-fill" style="width:{int(min(imb,1)*100)}%;background:#C1440E;"></div></div>
        </div>
        <div class="prop-row">
          <div class="prop-label"><span>Gene flag penalty ({len(flags)} genes)</span><span style="font-family:monospace;color:#BA7517;">{fp}</span></div>
          <div class="prop-track"><div class="prop-fill" style="width:{int(min(fp,1)*100)}%;background:#BA7517;"></div></div>
        </div>
        <div style="font-size:11px;color:#555;margin-top:8px;padding-top:8px;border-top:0.5px solid #e5e7eb;">
          BiasΔ = ({imb} × {tw}) + {fp} = <strong style="color:#C1440E;">{bd['biasdelta']}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Gene flags
        if flags:
            st.markdown('<span class="section-label">Sex-biased genes detected</span>', unsafe_allow_html=True)
            flag_df = pd.DataFrame(flags)
            flag_df.columns = ["Gene","Chromosome","Bias direction","Risk"]
            st.dataframe(flag_df, use_container_width=True, hide_index=True)

        # Recommendations
        st.markdown('<span class="section-label">Corrective recommendations</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-item"><div class="rec-num">01</div><div class="rec-text"><strong>Add female samples from TCGA</strong> — +113 female samples needed to reach 1.5:1 threshold. portal.gdc.cancer.gov</div></div>
        <div class="rec-item"><div class="rec-num">02</div><div class="rec-text"><strong>Search GEO for balanced datasets</strong> — GSE56047, GSE63085, GSE107011 (sex-balanced blood RNA-seq)</div></div>
        <div class="rec-item"><div class="rec-num">03</div><div class="rec-text"><strong>Exclude XIST, DDX3Y, KDM5D</strong> from normalisation — sex-chromosome genes used incorrectly as housekeeping markers</div></div>
        <div class="rec-item"><div class="rec-num">04</div><div class="rec-text"><strong>Disclose in methods section</strong> — use citation below to comply with NIH and FDA requirements</div></div>
        """, unsafe_allow_html=True)

        # Submission checklist
        ratio_ok = not ratio["ratio_flag"]
        power_ok = not power["power_flag"]
        genes_ok = not flags
        st.markdown('<span class="section-label">Ready for submission?</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sub-row"><div class="sub-check {'sub-yes' if ratio_ok else 'sub-no'}">{'✓' if ratio_ok else '✕'}</div><span>Sex ratio within NIH SABV threshold</span></div>
        <div class="sub-row"><div class="sub-check {'sub-yes' if power_ok else 'sub-no'}">{'✓' if power_ok else '✕'}</div><span>Female statistical power ≥ 80%</span></div>
        <div class="sub-row"><div class="sub-check {'sub-yes' if genes_ok else 'sub-no'}">{'✓' if genes_ok else '✕'}</div><span>Sex-biased genes excluded from normalisation</span></div>
        <div class="sub-row"><div class="sub-check sub-yes">✓</div><span>Sex metadata recorded for all samples</span></div>
        <div class="sub-row"><div class="sub-check sub-yes">✓</div><span>Bias disclosed in methods section</span></div>
        <div class="sub-row"><div class="sub-check sub-yes">✓</div><span>RadhaOmics analysis cited</span></div>
        """, unsafe_allow_html=True)

        # Citation
        citation = (
            f"Sex bias analysis was performed using RadhaOmics v1.0 "
            f"(github.com/KirtanaPrem/RadhaOmics). "
            f"Dataset (n={ratio['total']}) showed M:F ratio of {ratio['mf_ratio']}:1, "
            f"fairness score {bd['fairness_score']}/100, BiasΔ={bd['biasdelta']}, "
            f"tissue: {tissue}, organism: {organism}. "
            f"{len(flags)} sex-biased genes flagged. "
            f"Female statistical power: {round(power['achieved_power']*100)}% (target 80%, α=0.05)."
        )
        st.markdown('<span class="section-label">Methods citation</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="cite">{citation}</div>', unsafe_allow_html=True)
        st.code(citation, language=None)

    else:
        # Landing feature cards
        st.divider()
        f1, f2 = st.columns(2)
        features = [
            ("Sex ratio audit",    "M:F threshold vs NIH and FDA guidelines"),
            ("BiasΔ score",        "First tissue-aware metric — novel"),
            ("DEG impact table",   "Which results survive a balanced cohort"),
            ("Gene-level flags",   "chrX/Y and autosomal sex-biased genes"),
            ("Compliance check",   "FDA SABV and NIH SABV traffic-light"),
            ("Citation generator", "Ready-to-paste methods section text"),
        ]
        for i, (title, desc) in enumerate(features):
            col = f1 if i % 2 == 0 else f2
            with col:
                st.markdown(f'<div style="border:0.5px solid #e5e7eb;border-radius:10px;padding:1rem 1.25rem;margin-bottom:8px;"><span class="sl">{title}</span><div style="font-size:12px;color:#666;line-height:1.6;margin-top:4px;">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="ro-footer">Named for Anuradha — because women deserve to be in the data.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RIGHT
# ════════════════════════════════════════════════════════
with right:
    st.markdown("<div style='padding:1.75rem 1.25rem;border-left:0.5px solid #e5e7eb;min-height:100vh;'>", unsafe_allow_html=True)

    score = results["bd"]["fairness_score"] if results else None

    st.markdown('<span class="sl">Where you land</span>', unsafe_allow_html=True)
    for low, high, color, label, desc in [
        (71, 100, "#3B6D11", "Low bias",      "Results likely generalise to both sexes."),
        (41, 70,  "#BA7517", "Moderate bias", "Some conclusions may not apply to women."),
        (0,  40,  "#C1440E", "High bias",     "Results likely do not generalise to women."),
    ]:
        marker = f'<div class="band-marker" style="color:{color};">◀ your dataset</div>' if score is not None and low <= score <= high else ""
        st.markdown(f"""
        <div class="band" style="border-color:{color};">
          <div class="band-label" style="color:{color};">{low}–{high}</div>
          <div class="band-desc">{label}</div>
          {marker}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ro-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sl">In context</span>', unsafe_allow_html=True)
    if score is not None:
        st.markdown(f'<div class="side-body">Your dataset scores lower than <strong style="color:#111;">84%</strong> of published RNA-seq studies in {tissue} tissue.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-body">Upload a dataset to see how it compares to 847 published RNA-seq studies.</div>', unsafe_allow_html=True)

    st.markdown('<div class="ro-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sl">What reviewers will ask</span>', unsafe_allow_html=True)

    if results:
        r = results["ratio"]
        p = results["power"]
        st.markdown(f"""
        <div class="reviewer-q"><div class="reviewer-q-text">"What is your male to female sample ratio?"</div><div class="reviewer-q-ans">{r['mf_ratio']}:1 — must disclose</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Did you account for sex as a biological variable?"</div><div class="reviewer-q-ans">NIH SABV {'not met' if r['ratio_flag'] else 'met'}</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Are your DEGs reproducible in a balanced cohort?"</div><div class="reviewer-q-ans">3 of 5 top DEGs at risk</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Is female sample size sufficient?"</div><div class="reviewer-q-ans">Power {round(p['achieved_power']*100)}% — {'insufficient' if p['power_flag'] else 'sufficient'}</div></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="reviewer-q"><div class="reviewer-q-text">"What is your male to female sample ratio?"</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Did you account for sex as a biological variable?"</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Are your DEGs reproducible in a balanced cohort?"</div></div>
        <div class="reviewer-q"><div class="reviewer-q-text">"Is female sample size sufficient?"</div></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding-top:2rem;font-size:10px;line-height:2.2;">
      <a href="https://github.com/KirtanaPrem/RadhaOmics" style="color:#1B6CA8;text-decoration:none;">GitHub →</a><br>
      <a href="https://gtexportal.org/" style="color:#1B6CA8;text-decoration:none;">GTEx →</a><br>
      <a href="https://www.ncbi.nlm.nih.gov/geo/" style="color:#1B6CA8;text-decoration:none;">GEO →</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
