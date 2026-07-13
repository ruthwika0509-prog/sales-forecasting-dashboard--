import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Product Demand Segments",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Demand Segmentation")
st.markdown("---")

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

    return df

df = load_data()

# ==========================================================
# CREATE FEATURES
# ==========================================================

segment = df.groupby("Sub-Category").agg(
    Total_Sales=("Sales","sum"),
    Average_Sales=("Sales","mean"),
    Sales_STD=("Sales","std"),
    Orders=("Sales","count")
).fillna(0)

# ==========================================================
# SCALE FEATURES
# ==========================================================

scaler = StandardScaler()

X = scaler.fit_transform(segment)

# ==========================================================
# KMEANS
# ==========================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

segment["Cluster"] = kmeans.fit_predict(X)

# ==========================================================
# CLUSTER NAMES
# ==========================================================

cluster_names = {
    0:"High Volume Stable",
    1:"Growing Demand",
    2:"Low Volume",
    3:"High Volatility"
}

segment["Demand Segment"] = segment["Cluster"].map(cluster_names)

# ==========================================================
# PCA
# ==========================================================

pca = PCA(n_components=2)

components = pca.fit_transform(X)

plot_df = pd.DataFrame()

plot_df["PC1"] = components[:,0]
plot_df["PC2"] = components[:,1]
plot_df["Cluster"] = segment["Demand Segment"].values
plot_df["Sub-Category"] = segment.index

# ==========================================================
# PLOT
# ==========================================================

st.subheader("Demand Clusters")

fig, ax = plt.subplots(figsize=(10,6))

for label in plot_df["Cluster"].unique():

    temp = plot_df[
        plot_df["Cluster"]==label
    ]

    ax.scatter(
        temp["PC1"],
        temp["PC2"],
        label=label,
        s=80
    )

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Product Demand Segmentation")

ax.legend()

st.pyplot(fig)

# ==========================================================
# TABLE
# ==========================================================

st.subheader("Sub-Category Demand Segments")

st.dataframe(
    segment.reset_index()[
        [
            "Sub-Category",
            "Total_Sales",
            "Average_Sales",
            "Sales_STD",
            "Orders",
            "Demand Segment"
        ]
    ]
)

# ==========================================================
# STOCKING STRATEGY
# ==========================================================

st.subheader("Recommended Stocking Strategy")

strategy = pd.DataFrame({

"Demand Segment":[
"High Volume Stable",
"Growing Demand",
"High Volatility",
"Low Volume"
],

"Recommended Strategy":[
"Maintain high inventory with regular replenishment.",
"Increase stock gradually to meet rising demand.",
"Maintain safety stock and monitor weekly.",
"Keep minimum inventory to reduce holding cost."
]

})

st.table(strategy)

# ==========================================================
# SUMMARY
# ==========================================================

best = (
segment.groupby("Demand Segment")["Total_Sales"]
.sum()
.idxmax()
)

st.success(
f"Highest demand segment: {best}"
)