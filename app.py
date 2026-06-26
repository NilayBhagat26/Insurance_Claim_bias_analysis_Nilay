import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import *

st.set_page_config(
    page_title="Insurance Claim Settlement Bias Dashboard",
    page_icon="🏢",
    layout="wide"
)

# --------------------------------------------------------
# Corporate Theme
# --------------------------------------------------------

st.markdown("""
<style>

.main{
background-color:#F5F7FA;
}

.block-container{
padding-top:1rem;
padding-bottom:1rem;
padding-left:2rem;
padding-right:2rem;
}

.metric-card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:0px 2px 8px rgba(0,0,0,0.15);

text-align:center;

}

h1{
color:#003366;
}

h2{
color:#003366;
}

.sidebar .sidebar-content{
background:#002B5B;
}

</style>
""",unsafe_allow_html=True)

# --------------------------------------------------------

st.title("🏢 Insurance Claim Settlement Bias Analysis Dashboard")

st.markdown("""
### Business Problem

You have recently joined the Claims Settlement Department of an Insurance Company.

Management suspects that claim settlement decisions may contain hidden bias.

The objective of this dashboard is to investigate claim settlement behaviour using:

- Descriptive Analytics
- Diagnostic Analytics
- Machine Learning
- Bias Detection
- Model Evaluation
- Business Recommendations

""")

# --------------------------------------------------------

st.sidebar.title("Navigation")

page=st.sidebar.radio(

"Select Module",

[
"🏠 Dashboard",

"📊 Dataset Overview",

"📈 Descriptive Analysis",

"🔍 Bias Diagnostics",

"🤖 Machine Learning",

"📉 Model Evaluation",

"💼 Findings"

]

)

# --------------------------------------------------------

uploaded_file=st.sidebar.file_uploader(

"Upload Insurance Dataset",

type="csv"

)

if uploaded_file is None:

    st.info("Please upload Insurance.csv to begin analysis.")

    st.stop()

df=pd.read_csv(uploaded_file)

# --------------------------------------------------------
# Cleaning
# --------------------------------------------------------

for c in ["SUM_ASSURED","PI_ANNUAL_INCOME"]:

    df[c]=pd.to_numeric(

        df[c].astype(str).str.replace(",",""),

        errors="coerce"

    )

df["PI_AGE"]=pd.to_numeric(df["PI_AGE"],errors="coerce")

# --------------------------------------------------------

approved=(df["POLICY_STATUS"]=="Approved Death Claim").sum()

rejected=(df["POLICY_STATUS"]=="Repudiate Death").sum()

total=len(df)

approval_rate=approved/total*100

avg_income=df["PI_ANNUAL_INCOME"].mean()

avg_sum=df["SUM_ASSURED"].mean()

# --------------------------------------------------------

if page=="🏠 Dashboard":

    st.header("Executive Dashboard")

    c1,c2,c3=st.columns(3)

    c4,c5,c6=st.columns(3)

    with c1:

        st.metric("Total Policies",f"{total:,}")

    with c2:

        st.metric("Approved Claims",approved)

    with c3:

        st.metric("Repudiated Claims",rejected)

    with c4:

        st.metric("Approval Rate",f"{approval_rate:.2f}%")

    with c5:

        st.metric("Average Income",f"₹ {avg_income:,.0f}")

    with c6:

        st.metric("Average Sum Assured",f"₹ {avg_sum:,.0f}")

    st.divider()

    left,right=st.columns([2,1])

    with left:

        fig=px.pie(

            df,

            names="POLICY_STATUS",

            title="Policy Status Distribution",

            hole=.55,

            color="POLICY_STATUS",

            color_discrete_sequence=["#2E86DE","#E74C3C"]

        )

        st.plotly_chart(fig,use_container_width=True)

    with right:

        fig2=px.histogram(

            df,

            x="PI_AGE",

            nbins=20,

            title="Age Distribution",

            color_discrete_sequence=["#003366"]

        )

        st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20),use_container_width=True)
