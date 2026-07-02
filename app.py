import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
import datetime

# 1. Page Configuration & Premium Styling
st.set_page_config(
    page_title="Ticket Triage & Acceleration Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium dashboard aesthetics
st.markdown("""
<style>
    /* Styling headers and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #852DF4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #8892B0;
        margin-bottom: 2rem;
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 240px; /* Enforces equal height */
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px);
        border-color: rgba(133, 45, 244, 0.6);
        box-shadow: 0 8px 30px rgba(133, 45, 244, 0.2);
    }
    
    .kpi-val {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 1rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
    }
    
    .speedup-badge {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ID = "ticket-triage-dashboard"
LOOKER_STUDIO_URL = "https://datastudio.google.com/reporting/82124fce-4aa0-4530-828a-8f3f41e4aab7"

# Helper for BigQuery Client
@st.cache_resource
def get_bq_client():
    try:
        return bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        st.error(f"Failed to create BigQuery Client: {e}")
        return None

bq_client = get_bq_client()

# 2. Sidebar Navigation & Links
st.sidebar.markdown(f'<div style="text-align: center;"><h2 style="font-weight: 700;">🎫 Triage Demo</h2></div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 External Reports")
st.sidebar.link_button("🌐 Open Looker Studio Dashboard", LOOKER_STUDIO_URL, use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Technology Stack")
st.sidebar.info("""
- **Backend Database**: GCP BigQuery
- **Compute Acceleration**: NVIDIA L4 GPU + RAPIDS cuDF
- **Frontend Dashboard**: Streamlit
- **Host Platform**: GCP Cloud Run
""")

# 3. Main Dashboard Layout
st.markdown('<h1 class="main-title">Ticket Triage & Pipeline Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Live Demonstration of GPU-Accelerated Feature Engineering and Ticket SLA Risk Profiling</p>', unsafe_allow_html=True)

# Tabs
tab_overview, tab_risk, tab_forecast = st.tabs([
    "🚀 Project Overview & Acceleration",
    "⚠️ Live Risk-Ranked Tickets",
    "📈 7-Day Volume Forecast"
])

# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW & ACCELERATION
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    This project demonstrates a production-grade **Support Ticket Triage and Analytics Pipeline** designed to ingest, process, and analyze support tickets at scale.
    
    Using synthetic data scaled to **5,000,000 tickets**, we showcase how modern GPU acceleration (NVIDIA RAPIDS cuDF) optimizes the processing bottleneck, while downstream forecasting and risk scoring prioritize workloads for operations teams.
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚡ GPU Acceleration Benchmark")
    st.markdown("Comparing feature engineering execution times on 5,000,000 records using standard **CPU Pandas** vs **GPU cuDF** (RAPIDS-accelerated Pandas):")
    
    # KPI Grid
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val">6.28s</div>
            <div class="kpi-label">Pandas CPU Execution</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #A78BFA;">3.61s</div>
            <div class="kpi-label">cuDF GPU Execution</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Benchmark Chart and Gauge Chart side-by-side
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            x=["Pandas (CPU)", "cuDF (GPU)"],
            y=[6.2797, 3.6053],
            marker_color=["#ef4444", "#8b5cf6"],
            text=["6.28s", "3.61s"],
            textposition="auto",
            width=0.4
        ))
        fig_bench.update_layout(
            title="Processing Time Comparison",
            yaxis_title="Execution Time (seconds)",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        
    with col_chart2:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 1.74,
            delta = {'reference': 1.0, 'position': "top", 'relative': True, 'valueformat': ".1%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pipeline Speedup Factor", 'font': {'size': 20, 'color': '#ffffff', 'family': 'Outfit'}},
            number = {'suffix': "x", 'font': {'size': 50, 'color': '#10B981', 'family': 'Outfit'}},
            gauge = {
                'axis': {
                    'range': [0.5, 2.5], 
                    'tickwidth': 2, 
                    'tickcolor': "#8892B0",
                    'tickvals': [0.5, 1.0, 1.5, 2.0, 2.5],
                    'ticktext': ["0.5x", "1.0x (CPU)", "1.5x", "2.0x", "2.5x"]
                },
                'bar': {'color': "#8b5cf6", 'thickness': 0.6},
                'bgcolor': "rgba(255,255,255,0.03)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0.5, 1.0], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [1.0, 1.5], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [1.5, 2.5], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#10B981", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.74
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#ffffff", 'family': "Outfit"},
            height=380,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: LIVE RISK-RANKED TICKETS
# ---------------------------------------------------------------------------
with tab_risk:
    st.markdown("### ⚠️ Live SLA Breach Risk Scoring")
    st.markdown("Retrieving currently open tickets with dynamic risk scoring. Risk bands are calculated dynamically using percentile ranks (Critical: Top 5%, High: Next 15%, Medium: Next 35%, Low: Bottom 45%).")
    
    if bq_client is not None:
        query_risk = """
        SELECT ticket_id, team, category, priority, risk_score, risk_band, computed_at
        FROM `ticket-triage-dashboard.ticket_analytics.ticket_risk_scores`
        ORDER BY risk_score DESC
        LIMIT 50
        """
        try:
            with st.spinner("Fetching live risk scores from BigQuery..."):
                df_risk = bq_client.query(query_risk).to_dataframe()
            
            # Format the computed_at column
            if "computed_at" in df_risk.columns:
                df_risk["computed_at"] = pd.to_datetime(df_risk["computed_at"]).dt.date
                
            # Table View
            st.markdown("#### Top 50 Open Tickets by Risk Score")
            st.dataframe(
                df_risk,
                column_config={
                    "ticket_id": "Ticket ID",
                    "team": "Assigned Team",
                    "category": "Category",
                    "priority": "Priority",
                    "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
                    "risk_band": "Risk Band",
                    "computed_at": "Computed At"
                },
                hide_index=True,
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error querying BigQuery table: {e}")
            st.info("Make sure the Cloud Run service account has proper BigQuery Data Viewer and Job User permissions.")
    else:
        st.warning("BigQuery client not initialized.")

# ---------------------------------------------------------------------------
# TAB 3: 7-DAY FORECAST
# ---------------------------------------------------------------------------
with tab_forecast:
    st.markdown("### 📈 7-Day Ticket Volume Forecast")
    st.markdown("Daily predictions of inbound ticket volumes per team over the next 7 days, computed from historical trend fits and moving averages.")
    
    if bq_client is not None:
        query_forecast = """
        SELECT team, forecast_date, predicted_ticket_count
        FROM `ticket-triage-dashboard.ticket_analytics.volume_forecast`
        ORDER BY forecast_date, team
        """
        try:
            with st.spinner("Fetching live volume forecast from BigQuery..."):
                df_fore = bq_client.query(query_forecast).to_dataframe()
                
            # Formatting Date
            df_fore["forecast_date"] = pd.to_datetime(df_fore["forecast_date"]).dt.strftime('%b %d, %Y')
            
            # Create interactive line chart
            fig_fore = px.line(
                df_fore,
                x="forecast_date",
                y="predicted_ticket_count",
                color="team",
                markers=True,
                labels={
                    "forecast_date": "Forecast Date",
                    "predicted_ticket_count": "Predicted Ticket Count",
                    "team": "Team"
                },
                template="plotly_dark"
            )
            fig_fore.update_layout(
                title="Predicted Daily Inbound Ticket Counts by Team",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_range=[df_fore["predicted_ticket_count"].min() * 0.95, df_fore["predicted_ticket_count"].max() * 1.05]
            )
            st.plotly_chart(fig_fore, use_container_width=True)
            
            # Expanded Table View
            with st.expander("🔍 View Raw Forecast Data"):
                st.dataframe(df_fore, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Error querying BigQuery table: {e}")
    else:
        st.warning("BigQuery client not initialized.")
