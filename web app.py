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

st.set_page_config(layout="wide", page_title="Correlation Analyzer")
st.title("📊 CSV/Excel Correlation Analyzer")
st.markdown("Upload a file and select a target variable to see correlations and p-values.")

# --- File uploader ---
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

if file:
    try:
        # Load file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, parse_dates=True)
        else:
            df = pd.read_excel(file, parse_dates=True)

        st.success("✅ File loaded successfully.")

        df_processed = df.copy()

        # Convert object columns to datetime if possible
        for col in df_processed.columns:
            if df_processed[col].dtype == "object":
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col])
                except:
                    pass

        df_encoded = pd.get_dummies(df_processed, drop_first=True)
        df_encoded = df_encoded.select_dtypes(include=[np.number])
        df_encoded = df_encoded.loc[:, df_encoded.nunique() > 1]

        exclude_cols = ['student_id', 'StudentID', 'ID']
        for col in exclude_cols:
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=col)

        num_rows = df_encoded.shape[0]
        df_encoded = df_encoded.loc[:, df_encoded.nunique() < num_rows]

        numeric_columns = df_encoded.columns.tolist()
        all_columns = df_processed.columns.tolist()

        if not numeric_columns:
            st.warning("No valid numeric columns found after preprocessing.")
            st.stop()

        # --- Sidebar settings ---
        st.sidebar.header("🔧 Settings")
        y_var = st.sidebar.selectbox("Select Y variable", numeric_columns)
        show_p_warnings = st.sidebar.checkbox("⚠️ Show tiny p-value warnings", True)
        show_r_warnings = st.sidebar.checkbox("⚠️ Show high correlation warnings", True)
        show_heatmap = st.sidebar.checkbox("🖼️ Show heatmap", True)
        annotate_heatmap = st.sidebar.checkbox("🔢 Annotate heatmap", False)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📌 Select Predictors")
        predictor_checkboxes = {}
        for col in numeric_columns:
            if col != y_var:
                predictor_checkboxes[col] = st.sidebar.checkbox(col, value=False)

        predictors = [col for col, selected in predictor_checkboxes.items() if selected]

        # --- Correlation Analysis ---
        if y_var:
            st.subheader(f"Correlations with '{y_var}'")
            x_vars = [col for col in numeric_columns if col != y_var]

            results = []
            progress = st.progress(0)
            for i, col in enumerate(x_vars):
                try:
                    r, p = pearsonr(df_encoded[col], df_encoded[y_var])
                    results.append((col, r, p))
                except:
                    continue
                progress.progress((i + 1) / len(x_vars))

            if not results:
                st.error("No valid correlations found.")
                st.stop()

            results.sort(key=lambda x: abs(x[1]), reverse=True)
            top_results = results[:5]

            for col, r, p in top_results:
                p_warn = " ‼️" if show_p_warnings and p < 1e-100 else ""
                r_warn = " ⚠️" if show_r_warnings and abs(r) > 0.999 else ""
                st.write(f"**{col}**: r = `{r:.4f}`, p = `{p:.2e}`{p_warn}{r_warn}")

            if show_heatmap:
                with st.expander("📊 Heatmap of Top Correlated Variables", expanded=False):
                    st.subheader("Heatmap of Top Correlated Variables")
                    top_vars = [col for col, _, _ in top_results]
                    corr_data = df_encoded[[y_var] + top_vars].corr().fillna(0)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_data, annot=annotate_heatmap, cmap="coolwarm", center=0, ax=ax)
                    ax.set_title(f"Correlation Heatmap: '{y_var}' vs Top Variables")
                    st.pyplot(fig)

            # --- Scatterplot Generator ---
            with st.expander("📈 Interactive Scatterplot Generator", expanded=False):
                st.subheader("Interactive Scatterplot Generator")

                scatter_x = st.selectbox("Select X variable", all_columns, key="scatter_x")
                scatter_y = st.selectbox("Select Y variable", numeric_columns, index=numeric_columns.index(y_var), key="scatter_y")
                show_regression = st.checkbox("Show regression line", value=False)

                if scatter_x and scatter_y and scatter_x != scatter_y:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    try:
                        if show_regression and np.issubdtype(df_processed[scatter_x].dtype, np.number):
                            sns.regplot(x=df_processed[scatter_x], y=df_encoded[scatter_y], ax=ax, scatter_kws={"s": 50})
                        else:
                            sns.scatterplot(x=df_processed[scatter_x], y=df_encoded[scatter_y], ax=ax)
                        ax.set_xlabel(scatter_x)
                        ax.set_ylabel(scatter_y)
                        ax.set_title(f"Scatterplot: {scatter_x} vs {scatter_y}")
                        st.pyplot(fig)

                        if np.issubdtype(df_processed[scatter_x].dtype, np.number):
                            r_val, p_val = pearsonr(df_processed[scatter_x], df_encoded[scatter_y])
                            st.markdown(f"**Correlation (r)**: `{r_val:.4f}`  \n**p-value**: `{p_val:.2e}`")
                        else:
                            st.info("ℹ️ Correlation not shown for non-numeric X-axis.")
                    except Exception as e:
                        st.error(f"Error rendering scatterplot: {e}")

            # --- Predictive Modeling ---
            st.subheader("📈 Predictive Modeling")

            with st.expander("Train a simple regression model"):
                if predictors:
                    X = df_encoded[predictors]
                    y = df_encoded[y_var]
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.success("✅ Model trained successfully!")

                        st.write("### Performance Metrics")
                        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
                        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")

                        fig2, ax2 = plt.subplots()
                        ax2.scatter(y_test, y_pred, alpha=0.6)
                        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        ax2.set_xlabel("Actual Values")
                        ax2.set_ylabel("Predicted Values")
                        ax2.set_title("Actual vs Predicted")
                        st.pyplot(fig2)
                    except Exception as e:
                        st.error(f"Error training model: {e}")
                else:
                    st.info("☝️ Select predictors in the sidebar to enable model training.")

            # --- Predict Future Values ---
            if 'model' in locals() and predictors:
                with st.expander("Predict Future Values"):
                    st.markdown("Upload new predictor data (CSV) with the same predictor columns to generate predictions.")
                    pred_file = st.file_uploader("Upload predictor CSV for prediction", type=["csv"], key="pred_file")
                    if pred_file:
                        pred_df = pd.read_csv(pred_file)
                        missing_cols = [col for col in predictors if col not in pred_df.columns]
                        if missing_cols:
                            st.error(f"Uploaded data is missing columns: {missing_cols}")
                        else:
                            try:
                                X_new = pred_df[predictors]
                                future_preds = model.predict(X_new)
                                pred_df["Predicted_" + y_var] = future_preds
                                st.write("Predictions:")
                                st.dataframe(pred_df)
                                csv = pred_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Download predictions CSV", csv, "predictions.csv")
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")

            # --- LazyPredict Auto Model Selection ---
            if predictors:
                with st.expander("🤖 Auto Model Selection & Benchmarking"):
                    try:
                        X = df_encoded[predictors]
                        y = df_encoded[y_var]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        reg = LazyRegressor(verbose=0, ignore_warnings=True)
                        models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)
                        st.success("✅ Model benchmarking complete!")
                        st.write("### 📋 Model Comparison")
                        st.dataframe(models)
                        st.write(f"🔍 Best model: **{models.index[0]}** (based on R²)")
                    except Exception as e:
                        st.error(f"Error during auto model selection: {e}")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
