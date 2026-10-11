import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Studio", page_icon="👥", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.hero { padding: 26px 30px; border-radius: 18px; background: linear-gradient(120deg,#211d4e,#654fe0); color: white; margin-bottom: 20px; }
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { color: #dfdcff; margin: 8px 0 0; }
div[data-testid="stMetric"] { background: white; padding: 17px; border: 1px solid #e8eaf2; border-radius: 14px; }
.panel { background: white; border: 1px solid #e8eaf2; border-radius: 14px; padding: 18px; }
</style>
<div class="hero">
<h1>Customer Segmentation Studio</h1>
<p>Understand customer behavior, discover groups, and plan smarter marketing campaigns.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def sample_data():
    rng=np.random.default_rng(42)
    groups=[
        # income, spending, age, visits, cluster size
        (28,22,43,2,90),(78,82,35,8,85),(65,28,51,3,80),(42,72,29,7,75)
    ]
    rows=[]
    n=0
    for income,spend,age,visits,count in groups:
        for _ in range(count):
            n+=1
            rows.append({
                "CustomerID":f"C{n:04d}",
                "Age":int(np.clip(rng.normal(age,7),18,75)),
                "AnnualIncome_k":round(float(np.clip(rng.normal(income,10),12,150)),1),
                "SpendingScore":int(np.clip(rng.normal(spend,12),1,100)),
                "AnnualVisits":int(np.clip(rng.normal(visits,1.5),1,15)),
                "Gender":rng.choice(["Female","Male"]),
                "PreferredChannel":rng.choice(["Online","In-store","Mobile"],p=[.45,.3,.25])
            })
    return pd.DataFrame(rows)

def clean_data(df):
    df=df.copy()
    df.columns=[str(c).strip() for c in df.columns]
    required=["Age","AnnualIncome_k","SpendingScore"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing required columns: "+", ".join(missing))
    for c in required:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=required)
    if "CustomerID" not in df: df["CustomerID"]=[f"C{i+1:04d}" for i in range(len(df))]
    if "Gender" not in df: df["Gender"]="Unknown"
    if "AnnualVisits" not in df: df["AnnualVisits"]=np.nan
    if "PreferredChannel" not in df: df["PreferredChannel"]="Unknown"
    return df

with st.sidebar:
    st.header("⚙️ Dashboard controls")
    uploaded=st.file_uploader("Upload customer CSV", type=["csv"])
    st.caption("Required columns: Age, AnnualIncome_k, SpendingScore")
    st.divider()
    clusters=st.slider("Number of customer segments",2,8,4)
    features=st.multiselect("Clustering features",["Age","AnnualIncome_k","SpendingScore","AnnualVisits"],default=["AnnualIncome_k","SpendingScore"])
    st.caption("Income should be annual income in thousands (e.g., 55.5).")

if uploaded:
    try:
        raw=pd.read_csv(uploaded)
        df=clean_data(raw)
        data_source=f"Uploaded: {uploaded.name}"
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        st.stop()
else:
    df=sample_data()
    data_source="Built-in demo data (synthetic)"

if len(df)<clusters:
    st.error("There must be at least as many valid customers as selected clusters.")
    st.stop()
if not features:
    st.warning("Choose at least one clustering feature in the sidebar.")
    st.stop()
for f in features:
    if f not in df: df[f]=np.nan
model_df=df[features].apply(pd.to_numeric,errors="coerce")
valid=model_df.notna().all(axis=1)
df=df.loc[valid].copy()
model_df=model_df.loc[valid]
if len(df)<clusters:
    st.error("Not enough complete rows for the selected features.")
    st.stop()
scaler=StandardScaler()
scaled=scaler.fit_transform(model_df)
km=KMeans(n_clusters=clusters,random_state=42,n_init=10)
df["Segment"]=km.fit_predict(scaled)
df["SegmentLabel"]=df["Segment"].map(lambda x:f"Segment {x+1}")
st.caption(f"Data source: {data_source}  •  {len(df):,} valid customers • K-Means clustering")

# Filters
c1,c2,c3=st.columns(3)
with c1:
    genders=sorted(df["Gender"].astype(str).unique())
    selected_gender=st.multiselect("Gender filter",genders,default=genders)
with c2:
    channels=sorted(df["PreferredChannel"].astype(str).unique())
    selected_channel=st.multiselect("Preferred channel",channels,default=channels)
with c3:
    segments=sorted(df["SegmentLabel"].unique())
    selected_segments=st.multiselect("Show segments",segments,default=segments)
view=df[df["Gender"].astype(str).isin(selected_gender)&df["PreferredChannel"].astype(str).isin(selected_channel)&df["SegmentLabel"].isin(selected_segments)].copy()

revenue_proxy=(view["SpendingScore"]*10).sum()
avg_income=view["AnnualIncome_k"].mean() if len(view) else 0
avg_spend=view["SpendingScore"].mean() if len(view) else 0
m1,m2,m3,m4=st.columns(4)
m1.metric("Customers in view",f"{len(view):,}")
m2.metric("Average annual income",f"${avg_income:,.1f}k")
m3.metric("Average spending score",f"{avg_spend:.1f}/100")
m4.metric("High-spending customers",f"{int((view['SpendingScore']>=70).sum()):,}")

st.subheader("Customer insights")
left,right=st.columns([1.25,1])
with left:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("**Income vs. spending behavior**")
    fig,ax=plt.subplots(figsize=(7,4))
    for i,label in enumerate(sorted(df["SegmentLabel"].unique())):
        part=view[view["SegmentLabel"]==label]
        if len(part):
            ax.scatter(part["AnnualIncome_k"],part["SpendingScore"],s=35,alpha=.72,label=label)
    ax.set_xlabel("Annual income ($ thousands)")
    ax.set_ylabel("Spending score (1–100)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)
    st.markdown('</div>',unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("**Customers per segment**")
    counts=view["SegmentLabel"].value_counts().sort_index()
    fig2,ax2=plt.subplots(figsize=(5,4))
    ax2.bar(counts.index,counts.values)
    ax2.set_ylabel("Number of customers")
    ax2.tick_params(axis="x",rotation=25)
    ax2.grid(axis="y",alpha=.2)
    fig2.tight_layout()
    st.pyplot(fig2,use_container_width=True)
    plt.close(fig2)
    st.markdown('</div>',unsafe_allow_html=True)

st.subheader("Segment profile")
profile_cols={"CustomerID":"Customer count","Age":"Avg age","AnnualIncome_k":"Avg income ($k)","SpendingScore":"Avg spending score"}
if "AnnualVisits" in view.columns and view["AnnualVisits"].notna().any(): profile_cols["AnnualVisits"]="Avg annual visits"
profile=view.groupby("SegmentLabel").agg(
    Customers=("CustomerID","count"),
    Avg_Age=("Age","mean"),
    Avg_Income_k=("AnnualIncome_k","mean"),
    Avg_Spending_Score=("SpendingScore","mean"),
    Avg_Annual_Visits=("AnnualVisits","mean")
).round(1)
st.dataframe(profile,use_container_width=True)

st.subheader("Recommended marketing actions")
for label,row in profile.iterrows():
    if row["Avg_Spending_Score"]>=70 and row["Avg_Income_k"]>=50:
        advice="High-value potential: loyalty rewards, early access, and premium bundles."
    elif row["Avg_Spending_Score"]>=65:
        advice="Engaged shoppers: personalized offers, cross-sell suggestions, and referral rewards."
    elif row["Avg_Income_k"]>=60 and row["Avg_Spending_Score"]<45:
        advice="High-income, lower-spending: learn preferences and test relevant product recommendations."
    else:
        advice="Value-focused or occasional shoppers: useful bundles, gentle reminders, and clear discounts."
    st.write(f"**{label} ({int(row['Customers'])} customers):** {advice}")

st.subheader("Customer records")
st.dataframe(view.drop(columns=["Segment"],errors="ignore").sort_values("SpendingScore",ascending=False),use_container_width=True,hide_index=True)
csv=view.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download segmented customer CSV",data=csv,file_name="customer_segments.csv",mime="text/csv")
st.info("Learning note: demo data is synthetic. Spending score is an example index, not actual revenue. K-Means segment numbers are arbitrary; compare the segment profiles rather than interpreting the number itself.")
