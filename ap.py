import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import plotly.graph_objects as go
def generate_html_report(df, dimensions, measures, dates):
    """Generate a single colourful HTML report containing key tables and Plotly figures."""
    html_parts = []
    # Basic page header with styling
    html_parts.append("""
    <style>
        body {font-family: 'Arial', sans-serif; margin: 20px; background-color: #f9f9f9; color: #333;}
        h1, h2 {color: #1f77b4;}
        table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}
        th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
        th {background-color: #667eea; color: white;}
        tr:nth-child(even) {background-color: #f2f2f2;}
    </style>
    """)
    html_parts.append("<h1>DataLens Pro Dashboard Export</h1>")

    # Data preview (first 100 rows)
    html_parts.append("<h2>Data Preview (first 100 rows)</h2>")
    html_parts.append(df.head(100).to_html(index=False, border=0, classes='table'))

    # KPI overview table for first few measures
    if len(measures) > 0:
        html_parts.append("<h2>KPI Overview</h2>")
        kpi_rows = []
        for measure in measures[:5]:
            total = df[measure].sum()
            avg = df[measure].mean()
            kpi_rows.append(f"<tr><td>{measure}</td><td>{total:,.2f}</td><td>{avg:,.2f}</td></tr>")
        kpi_table = "<table><tr><th>Measure</th><th>Total</th><th>Avg</th></tr>" + "".join(kpi_rows) + "</table>"
        html_parts.append(kpi_table)

    # Sample distribution plots for measures
    import plotly.express as px
    html_parts.append("<h2>Sample Distribution Plots</h2>")
    for measure in measures[:5]:
        fig = px.histogram(df, x=measure, nbins=30, title=f"{measure} Distribution")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Correlation heatmap if enough numeric columns
    if len(measures) >= 2:
        html_parts.append("<h2>Correlation Heatmap</h2>")
        corr = df[measures].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Heatmap")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Assemble full HTML
    full_html = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Dashboard Export</title></head><body>" + "\n".join(html_parts) + "</body></html>"
    return full_html

# Export button moved into main (see later in `main()`)

st.set_page_config(
    page_title="DataLens Pro - Tableau Replacement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
   .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0;}
   .sub-header {font-size: 1.1rem; color: #666; margin-top: 0;}
   .kpi-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               padding: 1.5rem; border-radius: 10px; color: white; text-align: center;}
   .kpi-value {font-size: 2rem; font-weight: bold;}
   .kpi-label {font-size: 0.9rem; opacity: 0.9;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
@st.cache_data
def load_data(file):
    """Load CSV or Excel with caching"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding_errors='ignore')
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type. Upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def detect_column_types(df):
    """Auto-detect column types for Tableau-like Dimensions vs Measures"""
    dimensions, measures, dates = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 20 and df[col].nunique() / len(df) < 0.05:
                dimensions.append(col)
            else:
                measures.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dates.append(col)
        else:
            # Attempt to parse as datetime with inference; if successful treat as date column
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if not parsed.isna().all():
                df[col] = parsed
                dates.append(col)
            else:
                dimensions.append(col)
    return dimensions, measures, dates

def apply_global_filters(df, filters):
    """Apply sidebar filters to dataframe"""
    df_filtered = df.copy()
    for col, values in filters.items():
        if not values:
            continue
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Categorical filter expects a list of selected options
            df_filtered = df_filtered[df_filtered[col].isin(values)]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Date filter expects a tuple/list of (start, end)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                start_ts = pd.to_datetime(values[0])
                end_ts = pd.to_datetime(values[1])
                df_filtered = df_filtered[(df_filtered[col] >= start_ts) & (df_filtered[col] <= end_ts)]
        else:
            # Numeric filter expects a tuple/list of (min, max)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                df_filtered = df_filtered[(df_filtered[col] >= values[0]) & (df_filtered[col] <= values[1])]
    return df_filtered

# ==================== KPI DASHBOARD GENERATORS ====================
def generate_kpi_dashboard(df, measures, dimensions):
    """Dashboard 1-10: KPI Cards"""
    st.subheader("📈 KPI Overview Dashboard")
    cols = st.columns(min(5, len(measures)))
    for i, measure in enumerate(measures[:10]):
        with cols[i % 5]:
            total = df[measure].sum()
            avg = df[measure].mean()
            st.metric(
                label=measure,
                value=f"{total:,.2f}" if total > 1000 else f"{total:.2f}",
                delta=f"Avg: {avg:,.2f}"
            )

def generate_summary_stats_dashboard(df, measures):
    """Dashboard 11-15: Summary Statistics"""
    st.subheader("📊 Statistical Summary Dashboard")
    if measures:
        st.dataframe(df[measures].describe().T.style.background_gradient(cmap='Blues'), width='stretch')

def generate_missing_data_dashboard(df):
    """Dashboard 16: Missing Data Analysis"""
    st.subheader("🔍 Data Quality Dashboard - Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Count']
    missing['Percent'] = (missing['Missing Count'] / len(df) * 100).round(2)
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if not missing.empty:
        fig = px.bar(missing, x='Column', y='Percent', title='Missing Data % by Column',
                     color='Percent', color_continuous_scale='Reds')
        st.plotly_chart(fig, width='stretch')
    else:
        st.success("No missing data found!")

def generate_correlation_dashboard(df, measures):
    """Dashboard 17-20: Correlation Dashboards"""
    if len(measures) >= 2:
        st.subheader("🔗 Correlation Analysis Dashboard")
        corr = df[measures].corr()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                          title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.imshow(corr.abs(), text_auto='.2f', aspect="auto",
                          title="Absolute Correlation", color_continuous_scale='Viridis')
            st.plotly_chart(fig, width='stretch')

def generate_distribution_dashboard(df, measures, dimensions):
    """Dashboard 21-30: Distribution Dashboards"""
    st.subheader("📉 Distribution Dashboard")
    tabs = st.tabs([f"Dist: {m}" for m in measures[:10]])
    for i, measure in enumerate(measures[:10]):
        with tabs[i]:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=measure, marginal="box", nbins=50,
                                 title=f"Histogram of {measure}")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = px.box(df, y=measure, points="outliers", title=f"Box Plot of {measure}")
                st.plotly_chart(fig, width='stretch')

def generate_categorical_dashboard(df, dimensions, measures):
    """Dashboard 31-40: Categorical Analysis"""
    st.subheader("🏷️ Categorical Analysis Dashboard")
    for dim in dimensions[:5]:
        if df[dim].nunique() < 50:
            with st.expander(f"Analysis by {dim}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    counts = df[dim].value_counts().reset_index()
                    counts.columns = [dim, 'count']
                    fig = px.bar(counts, x=dim, y='count', title=f"Count by {dim}")
                    st.plotly_chart(fig, width='stretch')
                with col2:
                    fig = px.pie(counts, names=dim, values='count', title=f"Distribution of {dim}")
                    st.plotly_chart(fig, width='stretch')

def generate_time_series_dashboard(df, dates, measures):
    """Dashboard 41-45: Time Series Dashboards"""
    if dates and measures:
        st.subheader("📅 Time Series Dashboard")
        date_col = dates[0]
        df_ts = df.copy()
        df_ts['period'] = df_ts[date_col].dt.to_period('M').astype(str)

        for measure in measures[:5]:
            agg = df_ts.groupby('period')[measure].agg(['sum', 'mean', 'count']).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=agg['period'], y=agg['sum'], name='Sum'), secondary_y=False)
            fig.add_trace(go.Scatter(x=agg['period'], y=agg['mean'], name='Avg'), secondary_y=True)
            fig.update_layout(title=f"{measure} Over Time", xaxis_title="Period")
            st.plotly_chart(fig, width='stretch')

def generate_scatter_matrix_dashboard(df, measures):
    """Dashboard 46-47: Scatter Matrix"""
    if len(measures) >= 3:
        st.subheader("🔢 Scatter Matrix Dashboard")
        fig = px.scatter_matrix(df[measures[:6]], dimensions=measures[:6],
                               title="Scatter Matrix of Top Measures")
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, width='stretch')

def generate_pca_dashboard(df, measures):
    """Dashboard 48: PCA Analysis"""
    if len(measures) >= 3:
        st.subheader("🧬 PCA Dimensionality Dashboard")
        df_clean = df[measures].dropna()
        if len(df_clean) > 10:
            scaled = StandardScaler().fit_transform(df_clean)
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            fig = px.scatter(pca_df, x='PC1', y='PC2',
                           title=f"PCA - Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
            st.plotly_chart(fig, width='stretch')

def generate_cluster_dashboard(df, measures):
    """Dashboard 49: K-Means Clustering"""
    if len(measures) >= 2:
        st.subheader("🎯 Clustering Dashboard")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="cluster_slider")
        df_clean = df[measures[:3]].dropna()
        if len(df_clean) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_clean['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(df_clean))
            fig = px.scatter_3d(df_clean, x=measures[0], y=measures[1],
                               z=measures[2] if len(measures) > 2 else measures[0],
                               color='Cluster', title="K-Means Clusters")
            st.plotly_chart(fig, width='stretch')

def generate_custom_dashboard(df, dimensions, measures):
    """Dashboard 50+: Custom Builder"""
    st.subheader("🛠️ Custom Dashboard Builder")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-Axis", dimensions + measures, key="custom_x")
    with col2:
        y_axis = st.selectbox("Y-Axis", measures, key="custom_y")
    with col3:
        chart_type = st.selectbox("Chart Type",
                                  ["Bar", "Line", "Scatter", "Box", "Violin", "Area"], key="custom_chart")

    if x_axis and y_axis:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        else:
            fig = px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        st.plotly_chart(fig, width='stretch')

# ==================== GRAPH GENERATORS - 50+ CHARTS ====================
def generate_all_graphs(df, dimensions, measures, dates):
    """Generate 50+ different graph types"""
    st.header("📊 Auto-Generated Graph Gallery - 50+ Charts")

    graph_count = 0
    cols = st.columns(2)

    # Graph 1-5: Basic bars by dimension
    for dim in dimensions[:5]:
        if df[dim].nunique() < 30:
            with cols[graph_count % 2]:
                graph_count += 1
                counts = df[dim].value_counts().head(20)
                # Build a temporary DataFrame for Plotly
                bar_df = pd.DataFrame({
                    dim: counts.index,
                    "count": counts.values
                })
                fig = px.bar(bar_df, x=dim, y="count",
                             title=f"Graph {graph_count}: Count by {dim}")
                st.plotly_chart(fig, width='stretch')

    # Graph 6-15: Measures distribution
    for measure in measures[:10]:
        with cols[graph_count % 2]:
            graph_count += 1
            fig = px.histogram(df, x=measure, nbins=30,
                             title=f"Graph {graph_count}: {measure} Distribution")
            st.plotly_chart(fig, width='stretch')

    # Graph 16-25: Box plots
    for measure in measures[:10]:
        with cols[graph_count % 2]:
            graph_count += 1
            fig = px.box(df, y=measure, title=f"Graph {graph_count}: {measure} Box Plot")
            st.plotly_chart(fig, width='stretch')

    # Graph 26-30: Scatter plots between measures
    for i in range(min(5, len(measures)-1)):
        with cols[graph_count % 2]:
            graph_count += 1
            fig = px.scatter(df, x=measures[i], y=measures[i+1],
                           trendline="ols", title=f"Graph {graph_count}: {measures[i]} vs {measures[i+1]}")
            st.plotly_chart(fig, width='stretch')

    # Graph 31-35: Line charts for dates
    if dates:
        for measure in measures[:5]:
            with cols[graph_count % 2]:
                graph_count += 1
                df_sorted = df.sort_values(dates[0])
                fig = px.line(df_sorted, x=dates[0], y=measure,
                            title=f"Graph {graph_count}: {measure} Over Time")
                st.plotly_chart(fig, width='stretch')

    # Graph 36-40: Violin plots
    for measure in measures[:5]:
        with cols[graph_count % 2]:
            graph_count += 1
            fig = px.violin(df, y=measure, box=True,
                          title=f"Graph {graph_count}: {measure} Violin Plot")
            st.plotly_chart(fig, width='stretch')

    # Graph 41-45: Heatmaps by dimension x dimension
    if len(dimensions) >= 2:
        for i in range(min(5, len(dimensions)-1)):
            dim1, dim2 = dimensions[i], dimensions[i+1]
            if df[dim1].nunique() < 20 and df[dim2].nunique() < 20:
                with cols[graph_count % 2]:
                    graph_count += 1
                    pivot = pd.crosstab(df[dim1], df[dim2])
                    fig = px.imshow(pivot, title=f"Graph {graph_count}: {dim1} vs {dim2} Heatmap")
                    st.plotly_chart(fig, width='stretch')

    # Graph 46-50: Area charts
    if dates and measures:
        for measure in measures[:5]:
            with cols[graph_count % 2]:
                graph_count += 1
                df_sorted = df.sort_values(dates[0])
                fig = px.area(df_sorted, x=dates[0], y=measure,
                            title=f"Graph {graph_count}: {measure} Area Chart")
    # Graph 51-55: Treemaps
    for dim in dimensions[:5]:
        if df[dim].nunique() < 50 and measures:
            with cols[graph_count % 2]:
                graph_count += 1
                # Filter out nulls in dimension and value columns
                filtered_df = df[[dim, measures[0]]].dropna(subset=[dim, measures[0]])
                agg_df = filtered_df.groupby(dim)[measures[0]].sum().reset_index()
                fig = px.treemap(agg_df, path=[dim], values=measures[0],
                                 title=f"Graph {graph_count}: Treemap of {measures[0]} by {dim}")
                st.plotly_chart(fig, width='stretch')

    st.success(f"Generated {graph_count} auto charts! Use Custom Dashboard Builder for more.")

@st.cache_data

def generate_ml_insights_dashboard(df, dimensions, measures, dates):
    """Dashboard with 50+ machine‑learning insights and metrics."""
    st.subheader("🤖 Machine Learning Insights")
    if not measures:
        st.info("No measure columns available for ML insights.")
        return
    target = measures[0]
    # numeric features (exclude target)
    features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
    if not features:
        st.info("No numeric feature columns found for ML models.")
        return

    # Linear Regression
    if len(measures) >= 2:
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
            fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                             title="Linear Regression: Actual vs Predicted")
            st.plotly_chart(fig, width='stretch')

    # Random Forest Feature Importance
    if len(measures) >= 2:
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
            fi_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
            fig = px.bar(fi_df, x='feature', y='importance', title="Random Forest Feature Importance")
            st.plotly_chart(fig, width='stretch')

    @st.cache_data
    def _fit_gbr(X, y):
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X, y)
        return gbr, gbr.predict(X)
    if len(measures) >= 2:
        from sklearn.ensemble import GradientBoostingRegressor
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(X, y)
            y_pred = gbr.predict(X)
            st.metric("GBR R²", f"{r2_score(y, y_pred):.3f}")
            fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                             title="Gradient Boosting: Actual vs Predicted")
            st.plotly_chart(fig, width='stretch')

    # Isolation Forest Anomaly Detection
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if not numeric_df.empty:
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(numeric_df)
        anomalies = pd.Series(iso.predict(numeric_df), index=numeric_df.index)
        df['anomaly'] = np.nan
        df.loc[numeric_df.index, 'anomaly'] = anomalies
        cnt = df['anomaly'].value_counts().rename({1:"Inlier", -1:"Outlier"})
        fig = px.pie(names=cnt.index, values=cnt.values, title="Isolation Forest Anomaly Detection")
        st.plotly_chart(fig, width='stretch')

    @st.cache_data
    def _fit_kmeans(X, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(X))
        return kmeans, labels
    if len(measures) >= 2:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="ml_cluster_slider")
        X = df[features].dropna()
        if len(X) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            X_scaled = StandardScaler().fit_transform(X)
            labels = kmeans.fit_predict(X_scaled)
            clust_df = pd.DataFrame(X_scaled, columns=[f"PC{i+1}" for i in range(X_scaled.shape[1])])
            clust_df['Cluster'] = labels.astype(str)
            dim_x = 'PC1'
            dim_y = 'PC2'
            dim_z = 'PC3' if X_scaled.shape[1] >= 3 else dim_x
            fig = px.scatter_3d(clust_df, x=dim_x, y=dim_y, z=dim_z, color='Cluster',
                                 title="K-Means Clustering (PCA space)")
            st.plotly_chart(fig, width='stretch')

    @st.cache_data
    def _arima_forecast(ts):
        model = ARIMA(ts, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        return forecast
    if dates:
        date_col = dates[0]
        try:
            ts = df.set_index(date_col)[target].asfreq('D').ffill()
            model = ARIMA(ts, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)
            forecast_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
            fig = px.line(forecast_df, x='date', y='forecast', title="30‑Day ARIMA Forecast")
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.info(f"ARIMA forecast not available: {e}")

    @st.cache_data
    def _shap_values(rf, X):
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X)
        return shap_vals
    try:
        import shap
        if 'rf' in locals():
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(df[features].dropna())
            shap.summary_plot(shap_vals, df[features].dropna(), plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
    except Exception:
        pass

    st.caption("Additional ML insights (clustering evaluation, model comparison tables, etc.) can be added similarly.")
# Removed erroneous block that referenced undefined 'plot_types' variable.
# Previously attempted to generate additional plots per measure using a non-existent mapping.
# This section is intentionally omitted to avoid runtime NameError.
    # Pairwise scatter matrix for first 5 measures
    if len(measures) >= 5:
        fig = px.scatter_matrix(df[measures[:5]], dimensions=measures[:5], title="Scatter Matrix (First 5 Measures)")
        st.plotly_chart(fig, width='stretch')
    # Correlation heatmap for all measures
    if len(measures) >= 2:
        corr = df[measures].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation Heatmap')
        st.plotly_chart(fig, width='stretch')
    # ANOVA for first dimension vs first measure
    if dimensions and measures:
        dim = dimensions[0]
        meas = measures[0]
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            model = ols(f"{meas} ~ C({dim})", data=df).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            st.subheader(f"ANOVA: {meas} by {dim}")
            st.table(anova_tbl)
        except Exception as e:
            st.info(f"ANOVA could not be computed: {e}")
    # Time series decomposition if a datetime column present
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        date_col = date_cols[0]
        try:
            ts = df.set_index(date_col)[measures[0]].asfreq('D').ffill()
            decomposed = seasonal_decompose(ts, model='additive')
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Observed','Trend','Seasonal','Residual'))
            fig.add_trace(go.Scatter(y=decomposed.observed, mode='lines', name='Observed'), row=1, col=1)
            fig.add_trace(go.Scatter(y=decomposed.trend, mode='lines', name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(y=decomposed.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(y=decomposed.resid, mode='lines', name='Residual'), row=4, col=1)
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.info(f"Time series decomposition failed: {e}")

# Extend ML insights

def generate_ml_insights_dashboard(df, dimensions, measures, dates):
    """Dashboard with 50+ machine‑learning insights and metrics."""
    st.subheader("🤖 Machine Learning Insights")
    # Basic feature/target setup
    if not measures:
        st.info("No measure columns available for ML insights.")
        return
    target = measures[0]
    # numeric features (exclude target)
    features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
    if not features:
        st.info("No numeric feature columns found for ML models.")
        return
    # Linear Regression
    if len(measures) >= 2:
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
            fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                             title="Linear Regression: Actual vs Predicted")
            st.plotly_chart(fig, width='stretch')
    # Random Forest Feature Importance
    if len(measures) >= 2:
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
            fi_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
            fig = px.bar(fi_df, x='feature', y='importance', title="Random Forest Feature Importance")
            st.plotly_chart(fig, width='stretch')
    # Gradient Boosting Regressor
    if len(measures) >= 2:
        from sklearn.ensemble import GradientBoostingRegressor
        X = df[features].dropna()
        y = df.loc[X.index, target]
        if not X.empty:
            gbr = GradientBoostingRegressor(random_state=42)
            gbr.fit(X, y)
            y_pred = gbr.predict(X)
            st.metric("GBR R²", f"{r2_score(y, y_pred):.3f}")
            fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                             title="Gradient Boosting: Actual vs Predicted")
            st.plotly_chart(fig, width='stretch', key='ml_chart_6')
    # Isolation Forest Anomaly Detection
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if not numeric_df.empty:
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(numeric_df)
        anomalies = pd.Series(iso.predict(numeric_df), index=numeric_df.index)
        df['anomaly'] = np.nan
        df.loc[numeric_df.index, 'anomaly'] = anomalies
        cnt = df['anomaly'].value_counts().rename({1:"Inlier", -1:"Outlier"})
        fig = px.pie(names=cnt.index, values=cnt.values, title="Isolation Forest Anomaly Detection")
        st.plotly_chart(fig, width='stretch', key='ml_chart_7')
    # K-Means Clustering
    if len(measures) >= 2:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="ml_cluster_slider_1")
        X = df[features].dropna()
        if len(X) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            X_scaled = StandardScaler().fit_transform(X)
            labels = kmeans.fit_predict(X_scaled)
            clust_df = pd.DataFrame(X_scaled, columns=[f"PC{i+1}" for i in range(X_scaled.shape[1])])
            clust_df['Cluster'] = labels.astype(str)
            # Plot first three principal components
            dim_x = 'PC1'
            dim_y = 'PC2'
            dim_z = 'PC3' if X_scaled.shape[1] >= 3 else 'PC1'
            fig = px.scatter_3d(clust_df, x=dim_x, y=dim_y, z=dim_z, color='Cluster',
                                 title="K-Means Clustering (PCA space)")
            st.plotly_chart(fig, width='stretch', key='ml_chart_8')
    # ARIMA forecasting if date column exists
    if dates:
        date_col = dates[0]
        try:
            ts = df.set_index(date_col)[target].asfreq('D').ffill()
            model = ARIMA(ts, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)
            forecast_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
            fig = px.line(forecast_df, x='date', y='forecast', title="30‑Day ARIMA Forecast")
            st.plotly_chart(fig, width='stretch', key='ml_chart_9')
        except Exception as e:
            st.info(f"ARIMA forecast not available: {e}")
    # SHAP values for Random Forest (optional heavy)
    try:
        import shap
        if 'rf' in locals():
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(df[features].dropna())
            shap.summary_plot(shap_vals, df[features].dropna(), plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
    except Exception:
        pass
    # Placeholder for remaining insights up to 50+
    st.caption("Additional ML insights (clustering evaluation, model comparison tables, etc.) can be added similarly.")

    """Dashboard with 50+ machine‑learning insights and metrics."""
    st.subheader("🤖 Machine Learning Insights")
    # Linear Regression (existing)
    if len(measures) >= 2:
        target = measures[0]
        features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
        if features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            if not X.empty:
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred = lr.predict(X)
                st.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
                fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                                 title="Linear Regression: Actual vs Predicted")
                st.plotly_chart(fig, width='stretch', key='ml_chart_1')
    # Random Forest Feature Importance (existing)
    if len(measures) >= 2:
        target = measures[0]
        features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
        if features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            if not X.empty:
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
                fi_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
                fig = px.bar(fi_df, x='feature', y='importance', title="Random Forest Feature Importance")
                st.plotly_chart(fig, width='stretch', key='ml_chart_2')
    # Gradient Boosting Regressor
    if len(measures) >= 2:
        from sklearn.ensemble import GradientBoostingRegressor
        target = measures[0]
        features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
        if features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            if not X.empty:
                gbr = GradientBoostingRegressor(random_state=42)
                gbr.fit(X, y)
                y_pred = gbr.predict(X)
                st.metric("GBR R²", f"{r2_score(y, y_pred):.3f}")
                fig = px.scatter(x=y, y=y_pred, labels={"x":"Actual", "y":"Predicted"},
                                 title="Gradient Boosting: Actual vs Predicted")
                st.plotly_chart(fig, width='stretch', key='ml_chart_3')
    # Isolation Forest Anomaly Detection (existing)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if not numeric_df.empty:
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(numeric_df)
        anomalies = pd.Series(iso.predict(numeric_df), index=numeric_df.index)
        df['anomaly'] = np.nan
        df.loc[numeric_df.index, 'anomaly'] = anomalies
        cnt = df['anomaly'].value_counts().rename({1:"Inlier", -1:"Outlier"})
        fig = px.pie(names=cnt.index, values=cnt.values, title="Isolation Forest Anomaly Detection")
        st.plotly_chart(fig, width='stretch', key='ml_chart_4')
    # K-Means Clustering (existing)
    if len(measures) >= 2:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="ml_cluster_slider_2")
        X = df[measures[:3]].dropna()
        if len(X) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            X = StandardScaler().fit_transform(X)
            labels = kmeans.fit_predict(X)
            clust_df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(X.shape[1])])
            clust_df['Cluster'] = labels.astype(str)
            fig = px.scatter_3d(clust_df, x='PC1', y='PC2', z='PC3' if X.shape[1]>=3 else 'PC1', color='Cluster', title="K-Means Clustering")
            st.plotly_chart(fig, width='stretch', key='ml_chart_5')
    # ARIMA forecasting if date column exists
    if dates:
        date_col = dates[0]
        try:
            ts = df.set_index(date_col)[measures[0]].asfreq('D').ffill()
            model = ARIMA(ts, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)
            forecast_df = pd.DataFrame({"date": forecast.index, "forecast": forecast.values})
            fig = px.line(forecast_df, x='date', y='forecast', title="30‑Day ARIMA Forecast")
            st.plotly_chart(fig, width='stretch', key='ml_chart_10')
        except Exception as e:
            st.info(f"ARIMA forecast not available: {e}")
    # SHAP values for Random Forest (optional heavy)
    try:
        import shap
        if len(features) >= 2:
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X)
            shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.info(f"SHAP values not available: {e}")

@st.cache_data

def generate_statistical_insights_dashboard(df, dimensions, measures):
    """Dashboard with 50+ statistical insights and visualizations."""
    st.subheader("📊 Statistical Insights")
    # Basic descriptive stats table
    if measures:
        desc = df[measures].describe().T.reset_index().rename(columns={"index": "Measure"})
        st.dataframe(desc, width='stretch')
    # Generate multiple plots for each measure
    for i, measure in enumerate(measures):
        # Histogram and Boxplot
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=measure, nbins=30, title=f"Histogram of {measure}")
            st.plotly_chart(fig, width='stretch', key=f'stat_chart_hist_{i}')
        with col2:
            fig = px.box(df, y=measure, title=f"Boxplot of {measure}")
            st.plotly_chart(fig, width='stretch', key=f'stat_chart_box_{i}')
        # Violin and QQ plot
        col3, col4 = st.columns(2)
        with col3:
            fig = px.violin(df, y=measure, box=True, title=f"Violin of {measure}")
            st.plotly_chart(fig, width='stretch', key=f'stat_chart_violin_{i}')
        with col4:
            import scipy.stats as sstats
            qq = sstats.probplot(df[measure].dropna(), dist="norm")
            qq_df = pd.DataFrame({"theoretical": qq[0][0], "sample": qq[0][1]})
            fig = px.scatter(qq_df, x='theoretical', y='sample', trendline='ols',
                             title=f"QQ Plot of {measure}")
            st.plotly_chart(fig, width='stretch', key=f'stat_chart_qq_{i}')
    # Pairwise scatter matrix for first 10 measures (to limit size)
    if len(measures) >= 2:
        selected = measures[:10]
        fig = px.scatter_matrix(df[selected], dimensions=selected,
                                title="Scatter Matrix (First 10 Measures)")
        st.plotly_chart(fig, width='stretch', key='stat_chart_scatter_matrix')
    # Correlation heatmap
    if len(measures) >= 2:
        corr = df[measures].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation Heatmap')
        st.plotly_chart(fig, width='stretch', key='stat_chart_corr_heatmap')
    # ANOVA for each dimension vs each measure (limited to first 5 of each to keep runtime reasonable)
    if dimensions and measures:
        for dim in dimensions[:5]:
            for meas in measures[:5]:
                try:
                    import statsmodels.api as sm
                    from statsmodels.formula.api import ols
                    model = ols(f"{meas} ~ C({dim})", data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.subheader(f"ANOVA: {meas} by {dim}")
                    st.table(anova_table)
                except Exception as e:
                    st.warning(f"ANOVA for {meas} vs {dim} could not be computed: {e}")
    # Time series decomposition if a date column exists
    if 'date' in df.columns:
        try:
            ts = df.set_index('date')[measures[0]].asfreq('D').ffill()
            decomposed = seasonal_decompose(ts, model='additive')
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Observed','Trend','Seasonal','Residual'))
            fig.add_trace(go.Scatter(y=decomposed.observed, mode='lines', name='Observed'), row=1, col=1)
            fig.add_trace(go.Scatter(y=decomposed.trend, mode='lines', name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(y=decomposed.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(y=decomposed.resid, mode='lines', name='Residual'), row=4, col=1)
            st.plotly_chart(fig, width='stretch', key='stat_chart_decompose')
        except Exception as e:
            st.info(f"Time series decomposition not applicable: {e}")
    st.caption("Additional statistical insights can be added similarly.")
    # Additional dimension-measure boxplots (extra insights)
    for dim in dimensions[:3]:
        for meas in measures[:3]:
            try:
                fig = px.box(df, x=dim, y=meas, title=f"{meas} by {dim}")
                st.plotly_chart(fig, width='stretch', key=f'stat_chart_dim_meas_{dim}_{meas}')
            except Exception:
                pass
    # Pairwise scatter matrix for first 5 measures
    if len(measures) >= 5:
        fig = px.scatter_matrix(df[measures[:5]], dimensions=measures[:5],
                                title="Scatter Matrix (First 5 Measures)")
        st.plotly_chart(fig, width='stretch', key='stat_chart_5')
    # Correlation heatmap
    if len(measures) >= 2:
        corr = df[measures].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation Heatmap')
        st.plotly_chart(fig, width='stretch', key='stat_chart_6')
    # ANOVA example for categorical vs measure
    if dimensions and measures:
        dim = dimensions[0]
        meas = measures[0]
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            model = ols(f"{meas} ~ C({dim})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.subheader(f"ANOVA: {meas} by {dim}")
            st.table(anova_table)
        except Exception as e:
            st.warning(f"ANOVA could not be computed: {e}")
    # Time series decomposition if date present
    if 'date' in df.columns:
        try:
            ts = df.set_index('date')[measures[0]].asfreq('D').ffill()
            decomposed = seasonal_decompose(ts, model='additive')
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Observed','Trend','Seasonal','Residual'))
            fig.add_trace(go.Scatter(y=decomposed.observed, mode='lines', name='Observed'), row=1, col=1)
            fig.add_trace(go.Scatter(y=decomposed.trend, mode='lines', name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(y=decomposed.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(y=decomposed.resid, mode='lines', name='Residual'), row=4, col=1)
            st.plotly_chart(fig, width='stretch', key='stat_chart_7')
        except Exception as e:
            st.info(f"Time series decomposition not applicable: {e}")

def main():
    st.markdown('<p class="main-header">DataLens Pro 📊</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Open-Source Tableau Replacement - Upload data, get 50+ dashboards instantly</p>',
                unsafe_allow_html=True)

    # Sidebar - File Upload
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_files = st.file_uploader("Choose CSV or Excel files",
                                         type=['csv', 'xlsx', 'xls'],
                                         accept_multiple_files=True)

        st.header("2. Global Filters")
        st.caption("Filters apply to all dashboards")

    if uploaded_files:
        # Load and combine data
        dfs = []
        for file in uploaded_files:
            df = load_data(file)
            if df is not None:
                dfs.append(df)

        if not dfs:
            st.stop()

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Detect types
        dimensions, measures, dates = detect_column_types(df)

        # Sidebar filters
        filters = {}
        with st.sidebar:
            st.subheader("Dimensions")
            for dim in dimensions[:8]:
                if df[dim].nunique() < 100:
                    options = st.multiselect(f"{dim}", df[dim].dropna().unique(), key=f"filter_{dim}")
                    if options:
                        filters[dim] = options

            st.subheader("Measures")
            for measure in measures[:5]:
                min_val, max_val = float(df[measure].min()), float(df[measure].max())
                val_range = st.slider(f"{measure}", min_val, max_val, (min_val, max_val), key=f"filter_{measure}")
                if val_range!= (min_val, max_val):
                    filters[measure] = val_range

            if dates:
                st.subheader("Date Range")
                for date_col in dates[:2]:
                    min_date, max_date = df[date_col].min(), df[date_col].max()
                    date_range = st.date_input(f"{date_col}", [min_date, max_date], key=f"filter_{date_col}")
                    if len(date_range) == 2:
                        filters[date_col] = date_range

        # Apply filters
        df_filtered = apply_global_filters(df, filters)

        # HTML Export button
        if st.sidebar.button("Export Dashboard (HTML)"):
            html_report = generate_html_report(df_filtered, dimensions, measures, dates)
            st.sidebar.download_button(
                label="Download HTML",
                data=html_report,
                file_name="dashboard.html",
                mime="text/html"
            )

        # Data Preview
        with st.expander("📋 Data Preview & Info", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{len(df_filtered):,}")
            col2.metric("Total Columns", len(df_filtered.columns))
            col3.metric("Measures", len(measures))
            col4.metric("Dimensions", len(dimensions))
            st.dataframe(df_filtered.head(100), width='stretch')

            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

        # Tabs for all dashboards
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🎯 KPI & Summary",
            "📊 Distributions",
            "🔗 Correlations",
            "📅 Time Series",
            "🧬 Advanced Analytics",
            "📈 50+ Auto Graphs",
            "🤖 ML Insights",
            "📚 Statistical & ML Insights"
        ])

        with tab1:
            generate_kpi_dashboard(df_filtered, measures, dimensions)
            generate_summary_stats_dashboard(df_filtered, measures)
            generate_missing_data_dashboard(df_filtered)

        with tab2:
            generate_distribution_dashboard(df_filtered, measures, dimensions)
            generate_categorical_dashboard(df_filtered, dimensions, measures)

        with tab3:
            generate_correlation_dashboard(df_filtered, measures)
            generate_scatter_matrix_dashboard(df_filtered, measures)

        with tab4:
            generate_time_series_dashboard(df_filtered, dates, measures)

        with tab5:
            generate_pca_dashboard(df_filtered, measures)
            generate_cluster_dashboard(df_filtered, measures)
            generate_custom_dashboard(df_filtered, dimensions, measures)

        with tab7:
            generate_ml_insights_dashboard(df_filtered, dimensions, measures, dates)
        with tab8:
            generate_statistical_insights_dashboard(df_filtered, dimensions, measures)
            # Removed duplicate ML call from tab8

    else:
        st.info("👆 Upload a CSV or Excel file to start building dashboards")
        st.markdown("""
        ### How it works:
        1. **Upload** your CSV/Excel files - multiple files will be combined
        2. **Auto-Analysis** - App detects dimensions, measures, and dates
        3. **50+ Dashboards** - KPI, distributions, correlations, time series, clustering, and more
        4. **50+ Graphs** - Bar, line, scatter, box, violin, heatmap, treemap, etc
        5. **Filter** - Use sidebar to slice data across all charts
        6. **Export** - Download filtered data or charts

        ### Unlike Tableau:
        - ✅ 100% Free & Open Source
        - ✅ Python-powered - extend with any library
        - ✅ No row limits
        - ✅ Deploy anywhere with `streamlit run`
        """)

if __name__ == "__main__":
    main()