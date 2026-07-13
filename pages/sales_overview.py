import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Sales Overview",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Overview Dashboard")
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

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    return df

df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Filters")

region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(df["Region"].unique())
)

category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["Category"].unique())
)

filtered_df = df.copy()

if region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == region
    ]

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == category
    ]

# ==========================================================
# KPI CARDS
# ==========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"${filtered_df['Sales'].sum():,.2f}"
)

col2.metric(
    "Total Orders",
    len(filtered_df)
)

col3.metric(
    "Average Sales",
    f"${filtered_df['Sales'].mean():,.2f}"
)

st.markdown("---")

# ==========================================================
# TOTAL SALES BY YEAR
# ==========================================================

st.subheader("📊 Total Sales by Year")

year_sales = (
    filtered_df
    .groupby("Year")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    year_sales.index.astype(str),
    year_sales.values
)

ax.set_xlabel("Year")
ax.set_ylabel("Sales")

st.pyplot(fig)

# ==========================================================
# MONTHLY SALES TREND
# ==========================================================

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby("Month")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.xticks(rotation=45)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")

st.pyplot(fig)

# ==========================================================
# SALES BY REGION
# ==========================================================

st.subheader("🌍 Sales by Region")

region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    region_sales.index,
    region_sales.values
)

plt.xticks(rotation=30)

st.pyplot(fig)

# ==========================================================
# SALES BY CATEGORY
# ==========================================================

st.subheader("📦 Sales by Category")

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    category_sales.index,
    category_sales.values
)

st.pyplot(fig)

# ==========================================================
# DATA PREVIEW
# ==========================================================

st.subheader("📄 Dataset Preview")

st.dataframe(filtered_df.head(20))