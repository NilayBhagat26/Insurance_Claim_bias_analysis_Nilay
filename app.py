
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

st.title("Insurance Claim Settlement Bias Analysis")

file = st.file_uploader("Upload Insurance CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    for c in ["SUM_ASSURED","PI_ANNUAL_INCOME"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")

    st.subheader("Dataset")
    st.dataframe(df.head())

    target="POLICY_STATUS"
    st.subheader("Cross Tabulation")
    for col in [c for c in df.columns if c!=target][:5]:
        if df[col].dtype=="object":
            st.write(pd.crosstab(df[col], df[target], normalize="index")*100)

    X=df.drop(columns=[target,"PI_NAME","POLICY_NO"], errors="ignore")
    y=LabelEncoder().fit_transform(df[target])

    cat=X.select_dtypes(include="object").columns
    num=X.select_dtypes(exclude="object").columns

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    models={
        "KNN":KNeighborsClassifier(),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "Random Forest":RandomForestClassifier(random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(random_state=42)
    }

    metrics=[]
    figroc, axroc = plt.subplots()

    for name,model in models.items():
        pipe=Pipeline([("pre",pre),("model",model)])
        pipe.fit(Xtr,ytr)

        tr=pipe.predict(Xtr)
        pr=pipe.predict(Xte)

        cm=confusion_matrix(yte,pr)
        tn,fp,fn,tp=cm.ravel()

        metrics.append({
            "Model":name,
            "Train Accuracy":accuracy_score(ytr,tr),
            "Test Accuracy":accuracy_score(yte,pr),
            "Precision":precision_score(yte,pr),
            "Recall":recall_score(yte,pr),
            "F1":f1_score(yte,pr),
            "FP %":fp/cm.sum()*100,
            "FN %":fn/cm.sum()*100
        })

        if hasattr(pipe, "predict_proba"):
            prob=pipe.predict_proba(Xte)[:,1]
            fpr,tpr,_=roc_curve(yte,prob)
            axroc.plot(fpr,tpr,label=f"{name} AUC={auc(fpr,tpr):.3f}")

        st.subheader(name+" Confusion Matrix")
        st.write(cm)

    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(metrics))

    axroc.legend()
    st.pyplot(figroc)
