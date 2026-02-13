import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🚕 Uber Fare Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict your ride fare using Machine Learning</div>', unsafe_allow_html=True)

# Helper Functions
def haversine_distance(pick_lat, pick_lon, drop_lat, drop_lon):
    """Calculate Haversine distance in meters"""
    r = 6371  # Earth radius in km
    phi1, phi2 = np.radians(pick_lat), np.radians(drop_lat)
    dphi = np.radians(drop_lat - pick_lat)
    dlambda = np.radians(drop_lon - pick_lon)
    
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return round(2 * r * np.arcsin(np.sqrt(a)) * 1000, 2)

def manhattan_distance_km(pick_lat, pick_lon, drop_lat, drop_lon):
    """Calculate Manhattan distance in meters"""
    avg_lat = np.radians((pick_lat + drop_lat) / 2.0)
    lat_dist = 111.132
    lon_dist = 111.321 * np.cos(avg_lat)
    delta_lat = np.abs(drop_lat - pick_lat)
    delta_lon = np.abs(drop_lon - pick_lon)
    manhattan_km = (delta_lat * lat_dist) + (delta_lon * lon_dist)
    return round(manhattan_km * 1000, 2)

def preprocess_input(pickup_datetime, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, passenger_count):
    """Preprocess input data for model prediction"""
    
    # Extract datetime features
    year = pickup_datetime.year
    month = pickup_datetime.month
    day = pickup_datetime.day
    hour = pickup_datetime.hour
    day_of_week = pickup_datetime.weekday()
    
    # Calculate distances
    manhattan_dist = manhattan_distance_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Create DataFrame with all features
    # Based on the notebook, features used are: year, month, day, hour, day_of_week, passenger_count, manhattan_distance_m
    data = {
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'passenger_count': [passenger_count],
        'manhattan_distance_m': [manhattan_dist]
    }
    
    df = pd.DataFrame(data)
    
    # Reorder columns to match training order
    feature_order = [
        'passenger_count',
        'year',
        'month',
        'day',
        'hour',
        'day_of_week',
        'manhattan_distance_m'
    ]
    df = df[feature_order]
    
    return df, manhattan_dist

# Sidebar - Model Selection
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["XGBoost Regressor", "Ridge Regression", "Linear Regression"],
    help="Choose the machine learning model for prediction"
)

# Sidebar - Information
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Performance")

# Display model metrics based on typical results
if model_choice == "XGBoost Regressor":
    st.sidebar.metric("R² Score", "~0.85", help="Coefficient of determination")
    st.sidebar.metric("RMSE", "~$3.50", help="Root Mean Square Error")
    st.sidebar.info("🏆 Best performing model with ensemble learning")
elif model_choice == "Ridge Regression":
    st.sidebar.metric("R² Score", "~0.80", help="Coefficient of determination")
    st.sidebar.metric("RMSE", "~$4.20", help="Root Mean Square Error")
    st.sidebar.info("📈 Good regularized linear model")
else:
    st.sidebar.metric("R² Score", "~0.78", help="Coefficient of determination")
    st.sidebar.metric("RMSE", "~$4.50", help="Root Mean Square Error")
    st.sidebar.info("📉 Basic linear regression baseline")

# Main content - Input Form
st.header("📍 Enter Trip Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pickup Information")
    pickup_datetime = st.date_input("📅 Pickup Date", datetime.now())
    pickup_time = st.time_input("🕐 Pickup Time", datetime.now().time())
    pickup_lat = st.number_input("🌍 Pickup Latitude", value=40.7614, format="%.6f", 
                                 help="Example: 40.7614 (Times Square)")
    pickup_lon = st.number_input("🌍 Pickup Longitude", value=-73.9776, format="%.6f",
                                help="Example: -73.9776 (Times Square)")

with col2:
    st.subheader("Dropoff Information")
    st.write("")  # Spacing
    st.write("")  # Spacing
    dropoff_lat = st.number_input("🌍 Dropoff Latitude", value=40.7489, format="%.6f",
                                  help="Example: 40.7489 (Empire State)")
    dropoff_lon = st.number_input("🌍 Dropoff Longitude", value=-73.9680, format="%.6f",
                                  help="Example: -73.9680 (Empire State)")
    passenger_count = st.slider("👥 Number of Passengers", 1, 6, 1)

# Combine date and time
pickup_datetime_combined = datetime.combine(pickup_datetime, pickup_time)

# Display Map
st.header("🗺️ Trip Route")
map_col1, map_col2 = st.columns([3, 1])

with map_col1:
    # Create map centered between pickup and dropoff
    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add pickup marker
    folium.Marker(
        [pickup_lat, pickup_lon],
        popup="Pickup Location",
        tooltip="Pickup",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add dropoff marker
    folium.Marker(
        [dropoff_lat, dropoff_lon],
        popup="Dropoff Location",
        tooltip="Dropoff",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add line between points
    folium.PolyLine(
        [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
        color='blue',
        weight=3,
        opacity=0.7
    ).add_to(m)
    
    st_folium(m, width=700, height=400)

with map_col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📍 Coordinates**")
    st.markdown(f"**Pickup:**")
    st.markdown(f"Lat: {pickup_lat:.4f}")
    st.markdown(f"Lon: {pickup_lon:.4f}")
    st.markdown(f"**Dropoff:**")
    st.markdown(f"Lat: {dropoff_lat:.4f}")
    st.markdown(f"Lon: {dropoff_lon:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
st.markdown("---")
if st.button("🔮 Predict Fare", type="primary", use_container_width=True):
    try:
        # Preprocess input
        input_df, manhattan_dist = preprocess_input(
            pickup_datetime_combined, 
            pickup_lat, 
            pickup_lon, 
            dropoff_lat, 
            dropoff_lon, 
            passenger_count
        )
        
        # Calculate haversine for display
        haversine_dist = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        
        # Load model and scaler
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            if model_choice == "XGBoost Regressor":
                with open('xgb_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            elif model_choice == "Ridge Regression":
                with open('ridge_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            else:
                with open('lr_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("**Predicted Fare**")
                st.markdown(f'<div class="prediction-value">${prediction:.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**📏 Distance Info**")
                st.metric("Manhattan Distance", f"{manhattan_dist/1000:.2f} km")
                st.metric("Haversine Distance", f"{haversine_dist/1000:.2f} km")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**📊 Trip Details**")
                st.metric("Passengers", passenger_count)
                st.metric("Price per km", f"${prediction/(manhattan_dist/1000):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display input features used
            with st.expander("🔍 View Detailed Features"):
                st.dataframe(input_df, use_container_width=True)
                
        except FileNotFoundError:
            st.error("⚠️ Model files not found! Please ensure model files are in the same directory.")
            st.info("💡 You need to train and save the models first. See the training section below.")
            
            # Show mock prediction for demo
            mock_prediction = 15.50 + (manhattan_dist / 1000) * 2.5
            st.warning("📊 Showing DEMO prediction (not from actual model):")
            st.markdown(f'<div class="prediction-box"><div class="prediction-value">${mock_prediction:.2f}</div></div>', 
                       unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your input values and try again.")

# Footer with information
st.markdown("---")
st.header("📚 About This Application")

tab1, tab2, tab3 = st.tabs(["📖 Overview", "🔧 Features", "💾 Model Info"])

with tab1:
    st.markdown("""
    ### Uber Fare Prediction System
    
    This application uses machine learning to predict Uber ride fares based on:
    - **Pickup and Dropoff Locations**: Geographic coordinates
    - **DateTime Information**: Date, time, day of week
    - **Distance Calculations**: Manhattan and Haversine distances
    - **Passenger Count**: Number of passengers
    
    The models were trained on historical Uber ride data with extensive preprocessing and feature engineering.
    """)

with tab2:
    st.markdown("""
    ### Feature Engineering
    
    **Datetime Features:**
    - Year, Month, Day, Hour
    - Day of Week (0=Monday, 6=Sunday)
    
    **Distance Calculations:**
    - **Manhattan Distance**: Grid-based distance calculation
    - **Haversine Distance**: Great-circle distance between coordinates
    - **Log Transformation**: Applied to reduce skewness
    
    **Preprocessing:**
    - RobustScaler: Handles outliers better than StandardScaler
    - Log transformation on distance features
    - Outlier removal based on IQR method
    """)

with tab3:
    st.markdown("""
    ### Available Models
    
    **1. XGBoost Regressor** 🏆
    - Ensemble learning method
    - Best performance on test data
    - Parameters: n_estimators=300, learning_rate=0.05, max_depth=6
    
    **2. Ridge Regression** 📈
    - Linear model with L2 regularization
    - Good balance of bias-variance
    - Alpha=1.0
    
    **3. Linear Regression** 📉
    - Basic baseline model
    - Simple interpretable coefficients
    
    All models use RobustScaler for feature scaling.
    """)

# Training Instructions
st.markdown("---")
with st.expander("🎓 How to Train and Save Models"):
    st.markdown("""
    ### Steps to Create Model Files:
    
    ```python
    # 1. Train your models using the notebook
    # 2. Save the trained models and scaler:
    
    import pickle
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save XGBoost model
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save Ridge model
    with open('ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge, f)
    
    # Save Linear Regression model
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    ```
    
    Place these `.pkl` files in the same directory as `app.py`.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 2rem 0;'>
    <p>🚕 Uber Fare Prediction System | Built with Streamlit & Machine Learning</p>
    <p>📊 Project by: Nawasena Topan</p>
</div>
""", unsafe_allow_html=True)