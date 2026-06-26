import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Descriptive Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f9fc; }
[data-testid="stSidebar"] { background: #1a2340; }
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
.page-title { font-size:1.8rem; font-weight:700; color:#1a2340; margin-bottom:4px; }
.page-sub   { font-size:.95rem; color:#7a8399; margin-bottom:24px; }
.section-hdr {
    font-size:1.05rem; font-weight:600; color:#1a2340;
    border-bottom:2px solid #e8edf5; padding-bottom:8px; margin:28px 0 16px;
}
.metric-card {
    background:white; border-radius:12px; padding:18px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,.07); text-align:center;
}
.metric-card .mv { font-size:1.7rem; font-weight:700; color:#1a2340; }
.metric-card .ml { font-size:.8rem; color:#7a8399; margin-top:3px; }
.warn-box {
    background:#fdf3c8; border-left:4px solid #f6c90e;
    border-radius:8px; padding:12px 16px; font-size:.88rem; color:#92550a;
}
.ok-box {
    background:#def7ec; border-left:4px solid #31c48d;
    border-radius:8px; padding:12px 16px; font-size:.88rem; color:#057a55;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Descriptive Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Dataset overview, distributions, cross-tabulations and data quality assessment.</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Insurance CSV", type=["csv"])
if uploaded:
    st.session_state["df_raw"] = pd.read_csv(uploaded)

if "df_raw" not in st.session_state:
    st.info("👆 Please upload the Insurance CSV file to begin.")
    st.stop()

df = st.session_state["df_raw"].copy()
for c in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

TARGET = "POLICY_STATUS"
APPROVED = "Approved Death Claim"
REPUDIATED = "Repudiate Death"

# ── Dataset KPIs ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🗂️ Dataset Snapshot</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
miss_pct = (df.isnull().sum().sum() / df.size * 100)
approve_rate = (df[TARGET]==APPROVED).mean()*100
for col, (val, lbl) in zip(
    [c1,c2,c3,c4,c5,c6],
    [(len(df),"Total Records"),(df.shape[1],"Features"),
     (f"{approve_rate:.1f}%","Approval Rate"),
     (f"{100-approve_rate:.1f}%","Repudiation Rate"),
     (df.isnull().sum().sum(),"Missing Cells"),
     (f"{miss_pct:.1f}%","Missing %")]
):
    with col:
        st.markdown(f'<div class="metric-card"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

# ── Class Distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🎯 Target Variable Distribution</div>', unsafe_allow_html=True)
col_pie, col_bar = st.columns(2)

vc = df[TARGET].value_counts().reset_index()
vc.columns = ["Status","Count"]

with col_pie:
    fig = px.pie(vc, names="Status", values="Count",
                 color_discrete_sequence=["#1a56db","#e34948"],
                 title="Claim Status Split")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with col_bar:
    fig2 = px.bar(vc, x="Status", y="Count",
                  color="Status",
                  color_discrete_sequence=["#1a56db","#e34948"],
                  text="Count", title="Claim Counts")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
                       yaxis_title="Count", xaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

imbalance = approve_rate / (100-approve_rate)
if imbalance > 1.5:
    st.markdown(f'<div class="warn-box">⚠️ <strong>Class imbalance detected:</strong> Approved ({approve_rate:.1f}%) vs Repudiated ({100-approve_rate:.1f}%) — ratio {imbalance:.2f}:1. SMOTE will be applied in Feature Engineering.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="ok-box">✅ Class balance is acceptable.</div>', unsafe_allow_html=True)

# ── Numerical distributions ───────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📈 Numerical Feature Distributions</div>', unsafe_allow_html=True)
num_cols = ["PI_AGE","SUM_ASSURED","PI_ANNUAL_INCOME"]
num_cols = [c for c in num_cols if c in df.columns]

for col_name in num_cols:
    c_l, c_r = st.columns(2)
    with c_l:
        fig = px.histogram(df, x=col_name, color=TARGET,
                           barmode="overlay", opacity=0.7,
                           color_discrete_sequence=["#1a56db","#e34948"],
                           title=f"{col_name} Distribution by Claim Status",
                           nbins=40)
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", legend_title="Status")
        st.plotly_chart(fig, use_container_width=True)
    with c_r:
        fig2 = px.box(df, x=TARGET, y=col_name, color=TARGET,
                      color_discrete_sequence=["#1a56db","#e34948"],
                      title=f"{col_name} Boxplot")
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Categorical distributions ─────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📊 Categorical Feature Distributions</div>', unsafe_allow_html=True)
cat_cols = ["PI_GENDER","EARLY_NON","MEDICAL_NONMED","PAYMENT_MODE","ZONE"]
cat_cols = [c for c in cat_cols if c in df.columns]

for cat in cat_cols:
    grp = df.groupby([cat, TARGET]).size().reset_index(name="Count")
    total = grp.groupby(cat)["Count"].transform("sum")
    grp["Pct"] = (grp["Count"] / total * 100).round(1)

    fig = px.bar(grp, x=cat, y="Pct", color=TARGET,
                 barmode="group",
                 color_discrete_sequence=["#1a56db","#e34948"],
                 text="Pct",
                 title=f"{cat} vs POLICY_STATUS (%)",
                 labels={"Pct":"Percentage (%)"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      legend_title="Status", xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ── Cross-tabulation ──────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📋 Cross-Tabulation Explorer</div>', unsafe_allow_html=True)
sel = st.selectbox("Select a variable to cross-tabulate against POLICY_STATUS:", cat_cols)
ct = pd.crosstab(df[sel], df[TARGET])
ct_pct = pd.crosstab(df[sel], df[TARGET], normalize="index").mul(100).round(1)
ct_pct.columns = [f"{c} %" for c in ct_pct.columns]

tab1, tab2 = st.tabs(["Counts", "Row Percentages"])
with tab1:
    st.dataframe(ct, use_container_width=True)
with tab2:
    st.dataframe(ct_pct, use_container_width=True)

# ── Missing values ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🔍 Missing Value Analysis</div>', unsafe_allow_html=True)
miss = df.isnull().sum().reset_index()
miss.columns = ["Column","Missing Count"]
miss["Missing %"] = (miss["Missing Count"]/len(df)*100).round(2)
miss = miss[miss["Missing Count"]>0].sort_values("Missing %",ascending=False)

if miss.empty:
    st.markdown('<div class="ok-box">✅ No missing values found in this dataset.</div>', unsafe_allow_html=True)
else:
    fig = px.bar(miss, x="Column", y="Missing %", text="Missing %",
                 color="Missing %", color_continuous_scale="Reds",
                 title="Missing Values per Column")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(miss, use_container_width=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📐 Summary Statistics</div>', unsafe_allow_html=True)
st.dataframe(df[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)
