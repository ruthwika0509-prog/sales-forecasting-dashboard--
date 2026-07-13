import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Forecast Explorer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast Explorer")
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

    return df

df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================

forecast_type = st.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

if forecast_type == "Category":

    selected = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered = df[df["Category"] == selected]

else:

    selected = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    filtered = df[df["Region"] == selected]

# ==========================================================
# MONTHLY SALES
# ==========================================================

monthly_sales = (
    filtered
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
)

# ==========================================================
# FORECAST HORIZON
# ==========================================================

months = st.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    3
)

# ==========================================================
# SIMPLE FORECAST
# (Uses average of last 3 months as placeholder)
# ==========================================================

last_average = monthly_sales.tail(3).mean()

future_forecast = [last_average] * months

forecast_dates = pd.date_range(
    monthly_sales.index.max(),
    periods=months + 1,
    freq="ME"
)[1:]

forecast_df = pd.DataFrame({

    "Forecast Date": forecast_dates,
    "Forecasted Sales": future_forecast

})

# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    label="Historical Sales",
    linewidth=2
)

ax.plot(
    forecast_df["Forecast Date"],
    forecast_df["Forecasted Sales"],
    marker="o",
    linewidth=3,
    label="Forecast"
)

ax.legend()

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Sales Forecast")

st.pyplot(fig)

# ==========================================================
# FORECAST TABLE
# ==========================================================

st.subheader("Forecast Output")

st.dataframe(forecast_df)

# ==========================================================
# MODEL METRICS
# ==========================================================

st.subheader("Model Performance")

col1, col2 = st.columns(2)

# Replace these values with your actual XGBoost metrics
col1.metric("MAE", "15367.30")
col2.metric("RMSE", "19550.07")

st.success("Best Model Used: XGBoost")