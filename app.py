import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import os
import json
import subprocess
from streamlit_lottie import st_lottie
from folium.plugins import MarkerCluster

# Set page configuration
st.set_page_config(page_title="SmartRoute Optimizer", layout="wide")
# -------------------- Custom CSS for Animations --------------------
st.markdown("""
    <style>
    h1 {
        text-align: center;
        font-family: Arial, sans-serif;
        animation: fadeIn 2s ease-in-out;
    }

    .stButton>button {
        background-color: #009688;
        color: white;
        transition: transform 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #00796b;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        font-size: 18px;
        font-weight: bold;
        color: #FF5722;
        animation: blink 1s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar Time Slot Selection --------------------
st.sidebar.title("⏳ Select Time Slot")
time_slot_options = {
    "7:00-9:30": "07:00:00-09:30:00",
    "9:30-12:00": "09:30:00-12:00:00",
    "12:00-2:30": "12:00:00-14:30:00"
}

# Store the previous time slot in session state
if "prev_time_slot" not in st.session_state:
    st.session_state.prev_time_slot = None

selected_time_slot = st.sidebar.selectbox("Choose a time slot:", list(time_slot_options.keys()))

# -------------------- Sidebar File Upload --------------------
st.sidebar.title("📂 Upload Shipment Data")
shipment_file = st.sidebar.file_uploader("📦 Upload Shipment Data (Excel - .xlsx)", type=["xlsx"])

# -------------------- Load Lottie Animation --------------------
def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottie_file("delivery_truck.json")

# Display animation and title
col1, col2 = st.columns([1, 5])
with col1:
    st_lottie(lottie_animation, height=100, key="delivery_animation")
with col2:
    st.markdown("<h1 style='margin-right: 300px; margin-top: 0px;'>SmartRoute Optimizer</h1>", unsafe_allow_html=True)

# -------------------- Cache Processed Data --------------------
@st.cache_data
def process_uploaded_file(shipment_file, time_slot):
    """Process the uploaded file and return the optimized dataframe."""
    if shipment_file:
        shipment_path = "Shipment_Data.xlsx"
        with open(shipment_path, "wb") as f:
            f.write(shipment_file.getbuffer())

        # Run backend Python script to generate optimized trips
        with st.spinner("🔄 Optimizing routes... Please wait!"):
            subprocess.run(["python", "backend.py", time_slot])  # Pass selected time slot to backend

        output_excel = "optimized_trips.xlsx"
        if os.path.exists(output_excel):
            optimized_df = pd.read_excel(output_excel)

            # Ensure 'Shipment ID' is treated as a string and grouped correctly
            optimized_df["Shipment ID"] = optimized_df["Shipment ID"].astype(str)
            optimized_df['TRIP ID'] = optimized_df['TRIP ID'].fillna(method='ffill')

            # Transform data for display
            formatted_df = optimized_df.groupby("TRIP ID").agg({
                "Shipment ID": lambda x: ", ".join(sorted(x.dropna(), key=int)),  
                "MST_DIST": "first",
                "TRIP_TIME": "first",
                "Vehical_Type": "first",
                "CAPACITY_UTI": "first",
                "TIME_UTI": "first",
                "COV_UTI": "first"
            }).reset_index()

            formatted_df.rename(columns={"Shipment ID": "Shipments"}, inplace=True)
            return optimized_df, formatted_df

    return None, None

# -------------------- Main Logic --------------------
if shipment_file:
    st.sidebar.success("✅ Shipment data uploaded successfully!")

    # Check if time slot has changed
    time_slot_changed = st.session_state.prev_time_slot != selected_time_slot

    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != shipment_file or time_slot_changed:
        st.session_state.uploaded_file = shipment_file
        st.session_state.prev_time_slot = selected_time_slot  # Update session state
        optimized_df, formatted_df = process_uploaded_file(shipment_file, time_slot_options[selected_time_slot])
        st.session_state.optimized_df = optimized_df
        st.session_state.formatted_df = formatted_df
    else:
        optimized_df = st.session_state.optimized_df
        formatted_df = st.session_state.formatted_df

    if optimized_df is not None and formatted_df is not None:
        st.success("✅ Optimization completed! Showing results below.")

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📊 Optimized Delivery Trips", "🗺️ View Trip Routes"])

        with tab1:
            st.subheader("📊 Optimized Delivery Trips")
            st.dataframe(formatted_df)

        with tab2:
            st.subheader("🗺️ View Trip Routes")

            selected_trip = st.selectbox("Select a Trip ID to visualize:", formatted_df["TRIP ID"].unique())

            if selected_trip:
                trip_data = optimized_df[optimized_df["TRIP ID"] == selected_trip]
                m = folium.Map(location=[trip_data.iloc[0]["Latitude"], trip_data.iloc[0]["Longitude"]], zoom_start=12)
                marker_cluster = MarkerCluster().add_to(m)

                # Store location
                store_lat, store_lon = 19.075887, 72.877911
                folium.Marker([store_lat, store_lon], tooltip="Warehouse", icon=folium.Icon(color="green")).add_to(marker_cluster)

                shipment_coordinates = [[store_lat, store_lon]]

                for idx, row in trip_data.iterrows():
                    lat, lon = row["Latitude"], row["Longitude"]
                    shipment_id = row["Shipment ID"]

                    folium.Marker(
                        [lat, lon],
                        tooltip=f"Shipment {shipment_id} (Sequence {idx + 1})",
                        icon=folium.Icon(color="blue"),
                    ).add_to(marker_cluster)

                    shipment_coordinates.append([lat, lon])

                folium.PolyLine(
                    locations=shipment_coordinates,
                    color="blue",
                    weight=2.5,
                    opacity=1,
                ).add_to(m)

                folium_static(m, width=700, height=500)
            else:
                st.warning("⚠️ Please select a trip to visualize.")
    else:
        st.error("❌ Optimized trips file not found. Please check the backend script.")
else:
    st.warning("⚠️ Please upload a shipment data file to proceed.")