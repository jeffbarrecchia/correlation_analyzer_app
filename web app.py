# -*- coding: utf-8 -*-

from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide", page_title="Correlation & Time-Series Analyzer")

st.title("📊 Correlation & Time-Series Analyzer")
st.markdown("Upload a file and explore correlations, modeling, and time-series patterns.")

file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, parse_dates=True)
        else:
            df = pd.read_excel(file, parse_dates=True)

        st.success("✅ File loaded successfully.")
        df_processed = df.copy()

        # Convert potential object columns to datetime
        datetime_cols = []
        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                try:
                    converted = pd.to_datetime(df_processed[col])
                    if not converted.isna().all():
                        df_processed[col] = converted
                        datetime_cols.append(col)
                except:
                    pass
            elif np.issubdtype(df_processed[col].dtype, np.datetime64):
                datetime_cols.append(col)

        # Encode data for correlation and modeling
        df_encoded = pd.get_dummies(df_processed, drop_first=True)
        df_encoded = df_encoded.select_dtypes(include=[np.number])
        df_encoded = df_encoded.loc[:, df_encoded.nunique() > 1]
        df_encoded = df_encoded.loc[:, df_encoded.nunique() < df_encoded.shape[0]]

        numeric_columns = df_encoded.columns.tolist()
        all_columns = df_processed.columns.tolist()

        # Sidebar
        st.sidebar.header("🔧 Settings")
        y_var = st.sidebar.selectbox("Select Y variable", numeric_columns)
        show_p_warnings = st.sidebar.checkbox("⚠️ Show tiny p-value warnings", True)
        show_r_warnings = st.sidebar.checkbox("⚠️ Show high correlation warnings", True)
        show_heatmap = st.sidebar.checkbox("🖼️ Show heatmap", True)
        annotate_heatmap = st.sidebar.checkbox("🔢 Annotate heatmap", False)
        show_timeseries = st.sidebar.checkbox("📆 Enable time-series & seasonality analysis")

        # Correlation
        if y_var:
            st.subheader(f"Correlations with '{y_var}'")
            x_vars = [col for col in numeric_columns if col != y_var]
            results = []
            for col in x_vars:
                try:
                    r, p = pearsonr(df_encoded[col], df_encoded[y_var])
                    results.append((col, r, p))
                except:
                    continue
            results.sort(key=lambda x: abs(x[1]), reverse=True)
            top_results = results[:5]
            for col, r, p in top_results:
                p_warn = " ‼️" if show_p_warnings and p < 1e-100 else ""
                r_warn = " ⚠️" if show_r_warnings and abs(r) > 0.999 else ""
                st.write(f"**{col}**: r = `{r:.4f}`, p = `{p:.2e}`{p_warn}{r_warn}")

            if show_heatmap:
                with st.expander("📊 Heatmap"):
                    top_vars = [col for col, _, _ in top_results]
                    corr_data = df_encoded[[y_var] + top_vars].corr().fillna(0)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_data, annot=annotate_heatmap, cmap="coolwarm", center=0, ax=ax)
                    ax.set_title("Top Variable Heatmap")
                    st.pyplot(fig)

        # Time-Series Analysis
        if show_timeseries and datetime_cols:
            with st.expander("📆 Time-Series & Seasonality Decomposition"):
                st.subheader("📈 Time-Series Decomposition")
                ts_col = st.selectbox("Select datetime column", datetime_cols)
                ts_metric = st.selectbox("Select numeric column for decomposition", numeric_columns)

                freq_map = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M",
                    "Quarterly": "Q",
                    "Annually": "A"
                }
                freq_label = st.selectbox("Select decomposition frequency", list(freq_map.keys()))
                freq = freq_map[freq_label]

                try:
                    ts_df = df_processed[[ts_col, ts_metric]].dropna()
                    ts_df = ts_df.sort_values(ts_col).set_index(ts_col)
                    ts_df = ts_df.resample(freq).mean().interpolate()

                    decomposition = seasonal_decompose(ts_df[ts_metric], model='additive', period=1)
                    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                    decomposition.observed.plot(ax=axes[0], title='Observed')
                    decomposition.trend.plot(ax=axes[1], title='Trend')
                    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                    decomposition.resid.plot(ax=axes[3], title='Residual')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"⚠️ Error in decomposition: {e}")
        elif show_timeseries and not datetime_cols:
            st.warning("⚠️ No datetime columns detected for time-series analysis.")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
