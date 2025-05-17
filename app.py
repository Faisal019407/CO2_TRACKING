# Real-Time CO₂ Emission Tracker for Factories (Streamlit Version)
"""
💡 PROJECT OVERVIEW
This Streamlit app connects to factory operational data (uploaded via CSV) and estimates carbon dioxide (CO₂) emissions using AI models. It features real-time monitoring, anomaly detection, forecasting, and actionable suggestions.

🎯 KEY OBJECTIVES:
- Track real-time CO₂ emissions using factory parameters.
- Predict future emissions based on historical trends.
- Detect emission anomalies (e.g., spikes).
- Provide AI-powered emission reduction suggestions.

⚙️ HOW TO RUN:
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Visit displayed URL (e.g., http://localhost:8501)

📂 FILES:
- app.py       (this file)
- requirements.txt
- sample_data.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
from sklearn.linear_model import LinearRegression
import openai
from PIL import Image

# Uncomment and set your OpenAI API Key
# import os
# openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="CO₂ Emission Tracker",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make UI more attractive
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-top: 2rem;
        border-bottom: 2px solid #81C784;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #F1F8E9;
        border-left: 4px solid #7CB342;
        padding: 1rem;
        margin: 1rem 0;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🌿 Real-Time CO₂ Emission Tracker for Factories</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Monitor, forecast, and reduce factory CO₂ emissions with AI-powered insights.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='color: #2E7D32;'>📁 Upload & Settings</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV with factory data", type="csv")
    show_raw = st.checkbox("Show raw data", value=False)
    show_analysis = st.checkbox("Show detailed analysis", value=True)
    st.markdown("---")
    st.markdown("**Required Columns:** timestamp, power_kW, fuel_Lph, runtime_hr, production_qty, boiler_pressure, boiler_temp, humidity (optional)")
    
    # Sample data option
    st.markdown("---")
    st.markdown("<h4 style='color: #2E7D32;'>📊 Sample Data</h4>", unsafe_allow_html=True)
    if st.button("Load Sample Data"):
        uploaded_file = "sample_data.csv"

if uploaded_file:
    # Load data from file or sample
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Process timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🔍 Analysis", "📊 Data"])

    # Data tab content
    with tab3:
        st.markdown("<h3 class='sub-header'>📊 Raw Data</h3>", unsafe_allow_html=True)
        st.dataframe(df)
        
        # Download raw data button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Raw Data CSV",
            data=csv,
            file_name="co2_tracking_data.csv",
            mime="text/csv",
            key="raw-data-download"
        )

    # Estimate CO2
    def estimate_co2(row):
        val = 0
        # Check if required columns exist and add their contribution
        if 'fuel_Lph' in row:
            val += row['fuel_Lph'] * 2.68
        if 'power_kW' in row:
            val += row['power_kW'] * 0.5
        if 'runtime_hr' in row:
            val += row['runtime_hr'] * 0.3
        if 'boiler_pressure' in row:
            val += row['boiler_pressure'] * 0.2
        if 'boiler_temp' in row:
            val += row['boiler_temp'] * 0.05
        if 'humidity' in row:
            val *= (1 + 0.01*(100-row['humidity'])/100)
        return val

    df['co2_estimated'] = df.apply(estimate_co2, axis=1)
    
    # Dashboard tab content
    with tab1:
        # Key metrics
        st.markdown("<h3 class='sub-header'>📊 Key Metrics</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Average CO₂", f"{df['co2_estimated'].mean():.2f} kg/hr")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Max CO₂", f"{df['co2_estimated'].max():.2f} kg/hr")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total CO₂", f"{df['co2_estimated'].sum():.2f} kg")
            st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Data Points", f"{len(df)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Main chart
        st.markdown("<h3 class='sub-header'>📈 CO₂ Emission Trend</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['timestamp'], df['co2_estimated'], color='green', linewidth=2)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('CO₂ (kg/hr)')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    # Anomaly detection
    mean = df['co2_estimated'].mean()
    std = df['co2_estimated'].std()
    df['z_score'] = (df['co2_estimated'] - mean) / std
    anomalies = df[df['z_score'].abs() > 2]
    
    # Continue with Dashboard tab
    with tab1:
        # Anomalies and Forecast
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 class='sub-header'>🔔 Anomalies Detected</h3>", unsafe_allow_html=True)
            if anomalies.empty:
                st.markdown("<div class='insight-box'>No significant anomalies found.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='insight-box'>Showing top anomalies based on statistical deviation.</div>", unsafe_allow_html=True)
                st.dataframe(anomalies[['timestamp','co2_estimated','z_score']].head(10))
                
                # Download anomaly report
                anomaly_csv = anomalies.to_csv(index=False)
                st.download_button(
                    label="Download Anomaly Report",
                    data=anomaly_csv,
                    file_name="co2_anomalies.csv",
                    mime="text/csv",
                    key="anomaly-download"
                )
        
        with col2:
            st.markdown("<h3 class='sub-header'>🔮 Forecast Next 5 Days</h3>", unsafe_allow_html=True)
            df['ord'] = df['timestamp'].map(datetime.toordinal)
            model = LinearRegression().fit(df[['ord']], df['co2_estimated'])
            future = pd.date_range(df['timestamp'].max(), periods=5, freq='D')
            preds = model.predict(future.map(datetime.toordinal).to_numpy().reshape(-1,1))
            forecast_df = pd.DataFrame({'timestamp': future, 'forecasted_CO2': preds})
            st.markdown("<div class='insight-box'>5-day forecast based on historical trends.</div>", unsafe_allow_html=True)
            st.line_chart(forecast_df.set_index('timestamp')['forecasted_CO2'])
            
            # Download forecast
            forecast_csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast Data",
                data=forecast_csv,
                file_name="co2_forecast.csv",
                mime="text/csv",
                key="forecast-download"
            )

    # Continue with Dashboard tab - AI Suggestions
    with tab1:
        st.markdown("<h3 class='sub-header'>💡 AI-Powered Suggestions</h3>", unsafe_allow_html=True)
        prompt = f"Suggest ways to reduce CO2 emissions given these latest values: {df.tail(5)[['co2_estimated']].to_dict(orient='records')}"
        # Uncomment for real GPT
        # response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
        # suggestion = response.choices[0].message.content
        suggestion = "Optimize boiler settings, schedule maintenance to reduce idle runtime, and consider renewable energy sources."
        st.markdown(f"<div class='insight-box'>{suggestion}</div>", unsafe_allow_html=True)
        
        # Download full report
        st.markdown("<h3 class='sub-header'>💾 Complete Analysis Report</h3>", unsafe_allow_html=True)
        
        # Create a comprehensive report
        report_data = {
            'Average CO₂ (kg/hr)': [df['co2_estimated'].mean()],
            'Max CO₂ (kg/hr)': [df['co2_estimated'].max()],
            'Min CO₂ (kg/hr)': [df['co2_estimated'].min()],
            'Total CO₂ (kg)': [df['co2_estimated'].sum()],
            'Standard Deviation': [df['co2_estimated'].std()],
            'Number of Anomalies': [len(anomalies)],
            'Date Range': [f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"],
            'Recommendation': [suggestion]
        }
        
        report_df = pd.DataFrame(report_data)
        full_report = pd.concat([report_df, forecast_df.rename(columns={'forecasted_CO2': 'CO₂ Forecast (kg/hr)'})], axis=1)
        
        # Download button for complete report
        st.download_button(
            label="Download Complete Analysis Report",
            data=full_report.to_csv(index=False),
            file_name="co2_complete_analysis.csv",
            mime='text/csv',
            key='complete-report'
        )
    # Analysis tab content
    with tab2:
        st.markdown("<h3 class='sub-header'>🔎 Detailed CO₂ Analysis</h3>", unsafe_allow_html=True)
        
        # Feature correlation analysis
        st.markdown("<h4 style='color: #388E3C;'>Feature Correlation with CO₂ Emissions</h4>", unsafe_allow_html=True)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_data = []
        
        for col in numeric_cols:
            if col != 'co2_estimated' and col != 'z_score' and col != 'ord':
                correlation = df[col].corr(df['co2_estimated'])
                corr_data.append({'Feature': col, 'Correlation with CO₂': correlation})
        
        corr_df = pd.DataFrame(corr_data).sort_values('Correlation with CO₂', ascending=False)
        
        # Display correlation chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(corr_df['Feature'], corr_df['Correlation with CO₂'], color='green')
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Factory Parameter')
        ax.set_title('Impact of Parameters on CO₂ Emissions')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width - 0.05
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    va='center', ha='left' if width > 0 else 'right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance insight
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("**Key Insights:**", unsafe_allow_html=True)
        st.markdown("* The most influential parameters on CO₂ emissions are highlighted above.")
        st.markdown("* Focus on optimizing parameters with the highest correlation values for maximum impact.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Time series analysis
        st.markdown("<h4 style='color: #388E3C;'>Time-based Patterns</h4>", unsafe_allow_html=True)
        
        # Group by day and calculate statistics
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby('date')['co2_estimated'].agg(['mean', 'min', 'max']).reset_index()
        daily_stats.columns = ['Date', 'Average CO₂', 'Min CO₂', 'Max CO₂']
        
        # Display daily stats chart
        st.line_chart(daily_stats.set_index('Date'))
        
        # Time pattern insight
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("**Time Pattern Analysis:**", unsafe_allow_html=True)
        st.markdown("* Daily fluctuations in CO₂ emissions are visualized above.")
        st.markdown("* Identify specific days with unusually high emissions for targeted improvements.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download analysis data
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Correlation Analysis",
                data=corr_df.to_csv(index=False),
                file_name="co2_correlations.csv",
                mime='text/csv',
                key='corr-download'
            )
        
        with col2:
            st.download_button(
                label="Download Daily Statistics",
                data=daily_stats.to_csv(index=False),
                file_name="co2_daily_stats.csv",
                mime='text/csv',
                key='daily-stats-download'
            )
            
else:
    st.warning("Please upload a CSV file to get started or click 'Load Sample Data' in the sidebar.")
    st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
    st.markdown("**Getting Started:**", unsafe_allow_html=True)
    st.markdown("1. Upload a CSV file with factory data using the sidebar uploader")
    st.markdown("2. Or click 'Load Sample Data' to see a demonstration")
    st.markdown("3. Explore the dashboard, analysis, and data tabs")
    st.markdown("4. Download reports and insights for your records")
    st.markdown("</div>", unsafe_allow_html=True)

# requirements.txt content for reference:
# streamlit
# pandas
# matplotlib
# numpy
# scikit-learn
# openai
# pillow
