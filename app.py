import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("📊 Sales Forecasting Dashboard")

st.markdown("---")

st.header("Welcome!")

st.write("""
This dashboard was developed as part of the Sales Forecasting and Demand Analysis Project.

Use the navigation menu on the left to explore the different pages of the dashboard.
""")

st.markdown("---")

st.subheader("Dashboard Pages")

st.write("""
📈 **Page 1 – Sales Overview Dashboard**
- Total sales by year
- Monthly sales trend
- Sales by region
- Sales by category

🔮 **Page 2 – Forecast Explorer**
- Select Category or Region
- Forecast next 1, 2 or 3 months
- View MAE and RMSE

🚨 **Page 3 – Anomaly Report**
- Isolation Forest anomalies
- Z-Score anomalies
- Weekly anomaly table

📦 **Page 4 – Product Demand Segments**
- Product clustering
- PCA visualization
- Demand cluster table
""")

st.success("Select a page from the left sidebar to begin.")