# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:07:41 2025

@author: jeffb
"""

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
    # --- Load data ---
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.success("✅ File loaded successfully.")

        # --- Preprocessing ---
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_encoded = df_encoded.select_dtypes(include=[np.number])
        df_encoded = df_encoded.loc[:, df_encoded.nunique() > 1]
        
        # Remove ID-like columns
        exclude_cols = ['student_id', 'StudentID', 'ID']
        for col in exclude_cols:
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=col)
        num_rows = df_encoded.shape[0]
        df_encoded = df_encoded.loc[:, df_encoded.nunique() < num_rows]

        numeric_columns = df_encoded.columns.tolist()

        if not numeric_columns:
            st.warning("No valid numeric columns found after preprocessing.")
            st.stop()

        # --- Sidebar options ---
        st.sidebar.header("🔧 Settings")
        y_var = st.sidebar.selectbox("Select Y variable", numeric_columns)
        show_p_warnings = st.sidebar.checkbox("⚠️ Show tiny p-value warnings", True)
        show_r_warnings = st.sidebar.checkbox("⚠️ Show high correlation warnings", True)
        show_heatmap = st.sidebar.checkbox("🖼️ Show heatmap", True)
        annotate_heatmap = st.sidebar.checkbox("🔢 Annotate heatmap", False)

        # --- Perform correlation analysis ---
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

            # --- Display results ---
            for col, r, p in top_results:
                p_warn = " ‼️" if show_p_warnings and p < 1e-100 else ""
                r_warn = " ⚠️" if show_r_warnings and abs(r) > 0.999 else ""
                st.write(f"**{col}**: r = `{r:.4f}`, p = `{p:.2e}`{p_warn}{r_warn}")

            # --- Heatmap ---
            if show_heatmap:
                st.subheader("Heatmap of Top Correlated Variables")
                top_vars = [col for col, _, _ in top_results]
                corr_data = df_encoded[[y_var] + top_vars].corr().fillna(0)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_data, annot=annotate_heatmap, cmap="coolwarm", center=0, ax=ax)
                ax.set_title(f"Correlation Heatmap: '{y_var}' vs Top Variables")
                st.pyplot(fig)

            # --- Predictive Modeling ---
            st.subheader("📈 Predictive Modeling")

            with st.expander("Train a simple regression model"):
            st.markdown("Select one or more predictor variables to train a linear regression model to predict the Y variable.")

            predictors = st.multiselect("Select predictor(s)", [col for col in numeric_columns if col != y_var])

            if predictors:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score

                X = df_encoded[predictors]
                y = df_encoded[y_var]

                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    st.success("✅ Model trained successfully!")

                    # Display metrics
                    st.write("### Performance Metrics")
                    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
                    st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.4f}")

                    # Optional: Scatterplot of predictions vs actuals
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(y_test, y_pred, alpha=0.6)
                    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax2.set_xlabel("Actual Values")
                    ax2.set_ylabel("Predicted Values")
                    ax2.set_title("Actual vs Predicted")
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Error training model: {e}")


        # --- Scatterplot Generator ---
        st.subheader("📈 Interactive Scatterplot Generator")

        scatter_x = st.selectbox("Select X variable", numeric_columns, key="scatter_x")
        scatter_y = st.selectbox("Select Y variable", numeric_columns, index=numeric_columns.index(y_var) if y_var in numeric_columns else 0, key="scatter_y")
        show_regression = st.checkbox("Show regression line", value=False)

        if scatter_x and scatter_y and scatter_x != scatter_y:
            fig, ax = plt.subplots(figsize=(8, 6))
            if show_regression:
                sns.regplot(x=df_encoded[scatter_x], y=df_encoded[scatter_y], ax=ax, scatter_kws={"s": 50})
            else:
                sns.scatterplot(x=df_encoded[scatter_x], y=df_encoded[scatter_y], ax=ax)
            ax.set_xlabel(scatter_x)
            ax.set_ylabel(scatter_y)
            ax.set_title(f"Scatterplot: {scatter_x} vs {scatter_y}")
            st.pyplot(fig)

            # Correlation stats
            try:
                r_val, p_val = pearsonr(df_encoded[scatter_x], df_encoded[scatter_y])
                st.markdown(f"**Correlation (r)**: `{r_val:.4f}`  \n**p-value**: `{p_val:.2e}`")
            except:
                st.warning("⚠️ Could not compute correlation for the selected pair.")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

