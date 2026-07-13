import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")

st.title("🚨 Anomaly Report")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================

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

# =====================================================
# WEEKLY SALES
# =====================================================

weekly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
    .sum()
    .reset_index()
)

# =====================================================
# ISOLATION FOREST
# =====================================================

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Isolation"] = iso.fit_predict(
    weekly_sales[["Sales"]]
)

weekly_sales["IF_Anomaly"] = (
    weekly_sales["Isolation"] == -1
)

# =====================================================
# Z SCORE
# =====================================================

weekly_sales["Rolling Mean"] = (
    weekly_sales["Sales"]
    .rolling(8)
    .mean()
)

weekly_sales["Rolling Std"] = (
    weekly_sales["Sales"]
    .rolling(8)
    .std()
)

weekly_sales["Z Score"] = (
    weekly_sales["Sales"]
    - weekly_sales["Rolling Mean"]
) / weekly_sales["Rolling Std"]

weekly_sales["Z_Anomaly"] = (
    weekly_sales["Z Score"].abs() > 2
)

# =====================================================
# PLOT
# =====================================================

st.subheader("Weekly Sales with Anomalies")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    weekly_sales["Order Date"],
    weekly_sales["Sales"],
    label="Weekly Sales",
    linewidth=2
)

# Isolation Forest

ax.scatter(
    weekly_sales.loc[
        weekly_sales["IF_Anomaly"],
        "Order Date"
    ],
    weekly_sales.loc[
        weekly_sales["IF_Anomaly"],
        "Sales"
    ],
    color="red",
    s=80,
    label="Isolation Forest"
)

# Z Score

ax.scatter(
    weekly_sales.loc[
        weekly_sales["Z_Anomaly"],
        "Order Date"
    ],
    weekly_sales.loc[
        weekly_sales["Z_Anomaly"],
        "Sales"
    ],
    color="green",
    s=60,
    marker="x",
    label="Z-Score"
)

ax.set_title("Weekly Sales Anomaly Detection")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# =====================================================
# TABLE
# =====================================================

st.subheader("Detected Anomalies")

table = weekly_sales[
    weekly_sales["IF_Anomaly"] |
    weekly_sales["Z_Anomaly"]
][[
    "Order Date",
    "Sales",
    "IF_Anomaly",
    "Z_Anomaly"
]]

st.dataframe(table)

# =====================================================
# SUMMARY
# =====================================================

st.subheader("Summary")

st.write("Isolation Forest detected anomalies based on machine learning.")

st.write("Z-Score detected statistical outliers more than 2 standard deviations from the rolling mean.")

common = table[
    table["IF_Anomaly"] &
    table["Z_Anomaly"]
].shape[0]

st.success(f"Common anomalies detected by both methods: {common}")

st.info("""
Possible reasons for anomalies:

• Festival sales

• Holiday discounts

• Black Friday promotions

• Year-end clearance

• Sudden demand spikes

• Supply shortages
""")