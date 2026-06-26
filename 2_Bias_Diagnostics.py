import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Bias Diagnostics", page_icon="🔍", layout="wide")

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
.stat-card {
    background:white; border-radius:12px; padding:18px 20px;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.stat-card h4 { font-size:.95rem; font-weight:600; color:#1a2340; margin-bottom:10px; }
.sig-badge {
    display:inline-block; font-size:.78rem; padding:4px 12px;
    border-radius:20px; font-weight:600;
}
.sig-yes   { background:#fde8e8; color:#c81e1e; }
.sig-no    { background:#def7ec; color:#057a55; }
.sig-maybe { background:#fdf3c8; color:#92550a; }
.di-card {
    background:white; border-radius:12px; padding:20px 22px;
    box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:16px;
}
.di-card h3 { font-size:1rem; font-weight:600; color:#1a2340; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔍 Bias Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Statistical tests for bias across gender, age, income, zone, early claim, and medical type dimensions.</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
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
df["REPUDIATED"] = (df[TARGET] != APPROVED).astype(int)

# ── Helper ────────────────────────────────────────────────────────────────────
def chi_square_test(col):
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))
    return chi2, p, dof, cramers_v

def sig_badge(p):
    if p < 0.01:
        return '<span class="sig-badge sig-yes">🔴 Highly Significant (p&lt;0.01)</span>'
    elif p < 0.05:
        return '<span class="sig-badge sig-yes">🟠 Significant (p&lt;0.05)</span>'
    elif p < 0.10:
        return '<span class="sig-badge sig-maybe">🟡 Marginal (p&lt;0.10)</span>'
    else:
        return '<span class="sig-badge sig-no">🟢 Not Significant</span>'

def repud_rate_chart(col, title):
    grp = df.groupby(col)["REPUDIATED"].agg(["mean","count"]).reset_index()
    grp.columns = [col, "Repudiation Rate", "Count"]
    grp["Repudiation Rate %"] = (grp["Repudiation Rate"]*100).round(1)
    overall = df["REPUDIATED"].mean()*100
    fig = px.bar(grp, x=col, y="Repudiation Rate %", text="Repudiation Rate %",
                 color="Repudiation Rate %",
                 color_continuous_scale=["#def7ec","#fde8e8"],
                 title=title)
    fig.add_hline(y=overall, line_dash="dash", line_color="#1a2340",
                  annotation_text=f"Overall {overall:.1f}%", annotation_position="top right")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      coloraxis_showscale=False, xaxis_tickangle=-30)
    return fig, grp

# ── Bias Summary Table ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📋 Bias Summary — All Dimensions</div>', unsafe_allow_html=True)

bias_dims = {
    "PI_GENDER":       "Gender Bias",
    "EARLY_NON":       "Early Claim Bias",
    "MEDICAL_NONMED":  "Medical vs Non-Medical",
    "PAYMENT_MODE":    "Payment Mode Bias",
}
summary_rows = []
for col, label in bias_dims.items():
    if col in df.columns:
        chi2, p, dof, cv = chi_square_test(col)
        summary_rows.append({
            "Dimension": label,
            "Variable": col,
            "Chi² Stat": round(chi2, 3),
            "p-value": round(p, 4),
            "Degrees of Freedom": dof,
            "Cramér's V": round(cv, 3),
            "Significance": "Significant ✅" if p < 0.05 else "Not Significant ❌"
        })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# ── 1. Gender Bias ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">👥 1. Gender Bias</div>', unsafe_allow_html=True)
col_chart, col_stat = st.columns([2, 1])

fig, grp = repud_rate_chart("PI_GENDER", "Repudiation Rate by Gender")
with col_chart:
    st.plotly_chart(fig, use_container_width=True)

chi2, p, dof, cv = chi_square_test("PI_GENDER")
with col_stat:
    st.markdown(f"""
    <div class="stat-card">
      <h4>Chi-Square Test Result</h4>
      <b>Chi² Statistic:</b> {chi2:.3f}<br>
      <b>p-value:</b> {p:.4f}<br>
      <b>Degrees of Freedom:</b> {dof}<br>
      <b>Cramér's V:</b> {cv:.3f}<br><br>
      {sig_badge(p)}<br><br>
      <small>Cramér's V interpretation: &lt;0.1 weak · 0.1–0.3 moderate · &gt;0.3 strong</small>
    </div>""", unsafe_allow_html=True)

# Disparate Impact
if "PI_GENDER" in df.columns:
    m_rate = df[df["PI_GENDER"]=="M"]["REPUDIATED"].mean()
    f_rate = df[df["PI_GENDER"]=="F"]["REPUDIATED"].mean()
    di = min(m_rate,f_rate)/max(m_rate,f_rate) if max(m_rate,f_rate)>0 else 1
    st.markdown(f"""
    **Disparate Impact Ratio (Gender):** `{di:.3f}`
    {"⚠️ Below 0.80 — potential discriminatory impact (80% rule)" if di < 0.80 else "✅ Above 0.80 — within acceptable fairness threshold"}
    """)

# ── 2. Age Bias ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🎂 2. Age-Wise Bias</div>', unsafe_allow_html=True)
df["AGE_GROUP"] = pd.cut(df["PI_AGE"],
                          bins=[0,30,45,60,100],
                          labels=["≤30","31–45","46–60","60+"])
col_chart2, col_stat2 = st.columns([2,1])

fig2, _ = repud_rate_chart("AGE_GROUP", "Repudiation Rate by Age Group")
with col_chart2:
    st.plotly_chart(fig2, use_container_width=True)

# ANOVA
groups = [grp["REPUDIATED"].values for _, grp in df.groupby("AGE_GROUP")]
f_stat, p_anova = stats.f_oneway(*groups)
with col_stat2:
    st.markdown(f"""
    <div class="stat-card">
      <h4>ANOVA Test Result</h4>
      <b>F-Statistic:</b> {f_stat:.3f}<br>
      <b>p-value:</b> {p_anova:.4f}<br><br>
      {sig_badge(p_anova)}<br><br>
      <small>ANOVA tests whether repudiation rate differs significantly across age groups.</small>
    </div>""", unsafe_allow_html=True)

# Age distribution
fig_age = px.histogram(df, x="PI_AGE", color=TARGET,
                        barmode="overlay", opacity=0.7,
                        color_discrete_sequence=["#1a56db","#e34948"],
                        title="Age Distribution: Approved vs Repudiated", nbins=30)
fig_age.update_layout(paper_bgcolor="white", plot_bgcolor="white")
st.plotly_chart(fig_age, use_container_width=True)

# ── 3. Income Bias ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">💰 3. Income-Wise Bias</div>', unsafe_allow_html=True)
df["INCOME_QUARTILE"] = pd.qcut(df["PI_ANNUAL_INCOME"].dropna(),
                                  q=4,
                                  labels=["Q1 Low","Q2 Mid-Low","Q3 Mid-High","Q4 High"],
                                  duplicates="drop")
df["INCOME_QUARTILE"] = df["INCOME_QUARTILE"].astype(str)

col_c3, col_s3 = st.columns([2,1])
fig3, _ = repud_rate_chart("INCOME_QUARTILE", "Repudiation Rate by Income Quartile")
with col_c3:
    st.plotly_chart(fig3, use_container_width=True)

# ANOVA on income quartiles
inc_groups = [g["REPUDIATED"].values for _, g in df.groupby("INCOME_QUARTILE") if len(g)>1]
f_inc, p_inc = stats.f_oneway(*inc_groups)
with col_s3:
    st.markdown(f"""
    <div class="stat-card">
      <h4>ANOVA — Income Quartiles</h4>
      <b>F-Statistic:</b> {f_inc:.3f}<br>
      <b>p-value:</b> {p_inc:.4f}<br><br>
      {sig_badge(p_inc)}<br><br>
      <small>Tests whether lower-income policyholders face higher repudiation rates.</small>
    </div>""", unsafe_allow_html=True)

# ── 4. Zone Bias ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🗺️ 4. Zone-Wise Bias</div>', unsafe_allow_html=True)
zone_grp = df.groupby("ZONE")["REPUDIATED"].agg(["mean","count"]).reset_index()
zone_grp.columns = ["ZONE","Repudiation Rate","Count"]
zone_grp["Repudiation Rate %"] = (zone_grp["Repudiation Rate"]*100).round(1)
zone_grp = zone_grp.sort_values("Repudiation Rate %", ascending=False)
overall_rate = df["REPUDIATED"].mean()*100

fig_zone = px.bar(zone_grp, x="ZONE", y="Repudiation Rate %",
                   text="Repudiation Rate %", color="Repudiation Rate %",
                   color_continuous_scale=["#def7ec","#fde8e8"],
                   title="Repudiation Rate by Zone (sorted highest to lowest)")
fig_zone.add_hline(y=overall_rate, line_dash="dash", line_color="#1a2340",
                    annotation_text=f"Overall {overall_rate:.1f}%")
fig_zone.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_zone.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                        coloraxis_showscale=False,
                        xaxis_tickangle=-45, height=450)
st.plotly_chart(fig_zone, use_container_width=True)

chi2_z, p_z, dof_z, cv_z = chi_square_test("ZONE")
col_z1, col_z2, col_z3 = st.columns(3)
col_z1.metric("Chi² Statistic", f"{chi2_z:.2f}")
col_z2.metric("p-value", f"{p_z:.4f}")
col_z3.metric("Cramér's V", f"{cv_z:.3f}")
st.markdown(sig_badge(p_z), unsafe_allow_html=True)

# ── 5. Early Claim Bias ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">⏱️ 5. Early vs Non-Early Claim Bias</div>', unsafe_allow_html=True)
col_c5, col_s5 = st.columns([2,1])
fig5, _ = repud_rate_chart("EARLY_NON", "Repudiation Rate: Early vs Non-Early Claims")
with col_c5:
    st.plotly_chart(fig5, use_container_width=True)

early = df[df["EARLY_NON"]=="EARLY"]["REPUDIATED"]
non_early = df[df["EARLY_NON"]=="NON EARLY"]["REPUDIATED"]
z_stat, p_z_test = stats.ttest_ind(early, non_early)
risk_ratio = early.mean() / non_early.mean() if non_early.mean() > 0 else np.nan
with col_s5:
    st.markdown(f"""
    <div class="stat-card">
      <h4>Two-Sample t-Test</h4>
      <b>Early Repudiation Rate:</b> {early.mean()*100:.1f}%<br>
      <b>Non-Early Repudiation Rate:</b> {non_early.mean()*100:.1f}%<br>
      <b>Risk Ratio:</b> {risk_ratio:.2f}x<br>
      <b>p-value:</b> {p_z_test:.4f}<br><br>
      {sig_badge(p_z_test)}<br><br>
      <small>Risk ratio: early claims are repudiated {risk_ratio:.1f}x more than non-early.</small>
    </div>""", unsafe_allow_html=True)

# ── 6. Medical vs Non-Medical ─────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🏥 6. Medical vs Non-Medical Bias</div>', unsafe_allow_html=True)
col_c6, col_s6 = st.columns([2,1])
fig6, _ = repud_rate_chart("MEDICAL_NONMED", "Repudiation Rate: Medical vs Non-Medical")
with col_c6:
    st.plotly_chart(fig6, use_container_width=True)

med = df[df["MEDICAL_NONMED"]=="MEDICAL"]["REPUDIATED"]
nmed = df[df["MEDICAL_NONMED"]=="NON MEDICAL"]["REPUDIATED"]
t_stat, p_med = stats.ttest_ind(med, nmed)
chi2_m, p_chi_m, _, cv_m = chi_square_test("MEDICAL_NONMED")
with col_s6:
    st.markdown(f"""
    <div class="stat-card">
      <h4>Chi-Square + t-Test</h4>
      <b>Medical Repudiation Rate:</b> {med.mean()*100:.1f}%<br>
      <b>Non-Medical Repudiation Rate:</b> {nmed.mean()*100:.1f}%<br>
      <b>Chi² p-value:</b> {p_chi_m:.4f}<br>
      <b>t-test p-value:</b> {p_med:.4f}<br>
      <b>Cramér's V:</b> {cv_m:.3f}<br><br>
      {sig_badge(p_chi_m)}
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Statistical significance at α=0.05. Chi-square for categorical variables, ANOVA/t-test for continuous/binary comparisons.")
