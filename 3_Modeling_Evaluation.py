import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              precision_recall_curve, average_precision_score)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Modeling & Evaluation", page_icon="🤖", layout="wide")

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
.metric-card .mv { font-size:1.6rem; font-weight:700; color:#1a2340; }
.metric-card .ml { font-size:.8rem; color:#7a8399; margin-top:3px; }
.cm-box {
    background:white; border-radius:12px; padding:20px;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}
.best-badge {
    background:#def7ec; color:#057a55; font-size:.82rem;
    padding:4px 12px; border-radius:20px; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🤖 Modeling & Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">KNN · Decision Tree · Random Forest · Gradient Boosting — with cross-validation, ROC/PR curves, confusion matrices and business metrics.</div>', unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
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

# ── Prepare data ───────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET, "PI_NAME", "POLICY_NO"], errors="ignore")
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
pos_label_name = le.classes_[1]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh",  OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

MODELS = {
    "KNN":               KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":     DecisionTreeClassifier(random_state=42, max_depth=8),
    "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}
COLORS = {"KNN":"#1a56db","Decision Tree":"#e34948","Random Forest":"#057a55","Gradient Boosting":"#d97706"}

# ── Train ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_all(X_tr, X_te, y_tr, y_te, _pre, _models):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in _models.items():
        pipe = Pipeline([("pre", _pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        fpr_arr, tpr_arr, _ = roc_curve(y_te, y_prob)
        prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_prob)
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        results[name] = {
            "pipe": pipe,
            "train_acc": accuracy_score(y_tr, pipe.predict(X_tr)),
            "test_acc":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred),
            "recall":    recall_score(y_te, y_pred),
            "f1":        f1_score(y_te, y_pred),
            "roc_auc":   auc(fpr_arr, tpr_arr),
            "pr_auc":    average_precision_score(y_te, y_prob),
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "cm":        cm,
            "tn":tn,"fp":fp,"fn":fn,"tp":tp,"total":total,
            "fp_pct":    fp/total*100,
            "fn_pct":    fn/total*100,
            "fpr_val":   fp/(fp+tn)*100,
            "fnr_val":   fn/(fn+tp)*100,
            "fpr_arr":   fpr_arr,
            "tpr_arr":   tpr_arr,
            "prec_arr":  prec_arr,
            "rec_arr":   rec_arr,
        }
    return results

with st.spinner("Training models with cross-validation… this takes ~30 seconds"):
    results = train_all(X_tr, X_te, y_tr, y_te, pre, MODELS)

# ── Model Comparison Table ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📊 Model Comparison</div>', unsafe_allow_html=True)

best_model = max(results, key=lambda k: results[k]["roc_auc"])
rows = []
for name, r in results.items():
    rows.append({
        "Model": f"⭐ {name}" if name == best_model else name,
        "Train Acc": f"{r['train_acc']:.4f}",
        "Test Acc":  f"{r['test_acc']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall":    f"{r['recall']:.4f}",
        "F1 Score":  f"{r['f1']:.4f}",
        "ROC-AUC":   f"{r['roc_auc']:.4f}",
        "PR-AUC":    f"{r['pr_auc']:.4f}",
        "CV AUC (mean±std)": f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.markdown(f'<span class="best-badge">⭐ Best Model: {best_model} (ROC-AUC {results[best_model]["roc_auc"]:.4f})</span>', unsafe_allow_html=True)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📈 ROC Curves</div>', unsafe_allow_html=True)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="grey"),name="Random (AUC=0.50)"))
for name, r in results.items():
    fig_roc.add_trace(go.Scatter(
        x=r["fpr_arr"], y=r["tpr_arr"], mode="lines",
        name=f"{name} (AUC={r['roc_auc']:.3f})",
        line=dict(color=COLORS[name], width=2)
    ))
fig_roc.update_layout(
    title="ROC Curves — All Models",
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(x=0.6, y=0.1), height=450
)
st.plotly_chart(fig_roc, use_container_width=True)

# ── PR Curve ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📉 Precision-Recall Curves</div>', unsafe_allow_html=True)
fig_pr = go.Figure()
for name, r in results.items():
    fig_pr.add_trace(go.Scatter(
        x=r["rec_arr"], y=r["prec_arr"], mode="lines",
        name=f"{name} (PR-AUC={r['pr_auc']:.3f})",
        line=dict(color=COLORS[name], width=2)
    ))
fig_pr.update_layout(
    title="Precision-Recall Curves — All Models",
    xaxis_title="Recall", yaxis_title="Precision",
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(x=0.6, y=0.5), height=450
)
st.plotly_chart(fig_pr, use_container_width=True)
st.caption("PR-AUC is more informative than ROC-AUC when class imbalance exists (Approved 68% vs Repudiated 32%).")

# ── Confusion Matrices ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
cols_cm = st.columns(2)
for i, (name, r) in enumerate(results.items()):
    with cols_cm[i % 2]:
        cm = r["cm"]
        labels = ["Repudiate Death", "Approved Death Claim"]
        fig_cm = px.imshow(cm, text_auto=True,
                           x=labels, y=labels,
                           color_continuous_scale=["#f0f7ff","#1a2340"],
                           title=f"{name} — Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(paper_bgcolor="white", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

# ── Business Metrics ──────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">💼 Business Metrics — FP%, FN%, FPR, FNR</div>', unsafe_allow_html=True)
st.markdown("""
> **Insurance context:** A False Negative (wrongly repudiating a legitimate claim) carries higher regulatory & ethical risk
> than a False Positive (wrongly approving an invalid claim). Lower FN% is critical.
""")

biz_rows = []
for name, r in results.items():
    biz_rows.append({
        "Model": name,
        "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "TN": r["tn"],
        "FP %":  f"{r['fp_pct']:.2f}%",
        "FN %":  f"{r['fn_pct']:.2f}%",
        "False Positive Rate": f"{r['fpr_val']:.2f}%",
        "False Negative Rate": f"{r['fnr_val']:.2f}%",
    })
st.dataframe(pd.DataFrame(biz_rows), use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🔑 Feature Importance — Random Forest & Gradient Boosting</div>', unsafe_allow_html=True)

for model_name in ["Random Forest", "Gradient Boosting"]:
    pipe = results[model_name]["pipe"]
    model_obj = pipe.named_steps["model"]
    feat_names = (
        num_cols +
        list(pipe.named_steps["pre"]
             .named_transformers_["cat"]
             .named_steps["oh"]
             .get_feature_names_out(cat_cols))
    )
    importances = model_obj.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#e8f0fe","#1a2340"],
                    title=f"Top 15 Feature Importances — {model_name}")
    fig_fi.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=450)
    st.plotly_chart(fig_fi, use_container_width=True)

# ── CV Score chart ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">📐 5-Fold Cross-Validation AUC</div>', unsafe_allow_html=True)
cv_df = pd.DataFrame({
    "Model": list(results.keys()),
    "CV AUC Mean": [r["cv_mean"] for r in results.values()],
    "CV AUC Std":  [r["cv_std"]  for r in results.values()],
})
fig_cv = px.bar(cv_df, x="Model", y="CV AUC Mean",
                error_y="CV AUC Std", text="CV AUC Mean",
                color="Model",
                color_discrete_map=COLORS,
                title="Cross-Validated ROC-AUC (mean ± std)")
fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_cv.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      showlegend=False, yaxis_range=[0.5, 1.0])
st.plotly_chart(fig_cv, use_container_width=True)

st.session_state["model_results"] = results
st.session_state["best_model"] = best_model
