# SmartRoute Optimizer

**SmartRoute Optimizer** is a web application built with **Streamlit** to optimize delivery routes for shipments. It uses **geospatial algorithms** to assign vehicles, calculate optimal routes, and visualize trip details on an interactive map. The app allows users to upload shipment data, select time slots, and get optimized trip routes.

---

## Features

- **Upload Shipment Data**: Upload your shipment data in `.xlsx` format.
- **Time Slot Selection**: Select a time slot for the optimization process.
- **View Optimized Trips**: View optimized delivery trips with vehicle details and other performance metrics.
- **Visualize Trip Routes on Map**: View the route of each trip on an interactive map with shipment locations and the route path.

---

## Prerequisites

To run this app locally, make sure you have the following installed:

- **Python 3.8+**
- **Streamlit**  
  Install with:  
  ```bash
  pip install streamlit
  ```
- **Other required libraries**  
  Install with:  
  ```bash
  pip install pandas folium openpyxl streamlit-folium streamlit-lottie subprocess
  ```

---

## File Structure

```
SmartRoute-Optimizer/
│
├── app.py                # Main Streamlit app
├── backend.py            # Backend Python script for processing data
├── Shipment_Data.xlsx    # Sample input shipment data file
└── optimized_trips.xlsx  # Optimized trips output after backend processing
```

---

## Installation & Setup

### Step 1: Clone this repository

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/SmartRoute-Optimizer.git
cd SmartRoute-Optimizer
```

### Step 2: Install the necessary packages

Install all required dependencies:

```bash
pip install pandas folium openpyxl streamlit-folium streamlit-lottie subprocess
```

Alternatively, you can manually install the necessary libraries using the `pip install` command.

### Step 3: Prepare your input file

You will need to upload your own **shipment data file** in **.xlsx format**. This file should contain columns for `Shipment ID`, `Latitude`, `Longitude`, `Delivery Timeslot`, and other relevant data.

### Step 4: Run the Streamlit App

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

This will start the app and open a browser window to access the Streamlit interface.

---

## Usage

1. **Upload the Shipment Data**: Click the **"📦 Upload Shipment Data"** section in the sidebar and upload the `.xlsx` file containing your shipment data.
   
2. **Select a Time Slot**: Choose a time slot from the dropdown for which you want to optimize routes.
   
3. **Click Start Optimization**: Once the shipment data is uploaded, click **"🚀 Start Optimization"**. The backend script will run, and optimized trip data will be displayed.

4. **View Optimized Trips**: The trips will be displayed with details like `TRIP ID`, `Vehical_Type`, `MST_DIST`, `TRIP_TIME`, and performance metrics.

5. **Visualize Routes**: After selecting a `TRIP ID`, you can see the trip's route on a **folium interactive map**.

---

## How It Works

### Backend

The backend script (`backend.py`) is responsible for processing the uploaded shipment data and generating optimized trips based on vehicle capacity, trip distance, and other parameters. The script runs a **genetic algorithm** to optimize the vehicle assignments and routes.

1. **Input**: The script takes shipment data (with latitudes and longitudes) and calculates optimal routes for the vehicles.
2. **Output**: It generates an Excel file (`optimized_trips.xlsx`) that includes trip IDs, shipment IDs, route distances, and performance metrics.
3. **Vehicle Assignment**: The script assigns vehicles based on their capacities and the number of shipments to be delivered.

### Frontend (Streamlit)

The frontend app provides a user-friendly interface for uploading the shipment data, selecting the time slot, and displaying the optimized trip data. Users can:

- View the list of optimized trips in a table.
- Select a trip to visualize its route on an interactive map.

---

## Example

**Shipment Data File**: The `Shipment_Data.xlsx` file should have columns like:

| Shipment ID | Latitude | Longitude | Delivery Timeslot | ... |
|-------------|----------|-----------|-------------------|-----|
| S1          | 19.0760  | 72.8777   | 9:30-12:00        | ... |
| S2          | 19.0820  | 72.8820   | 9:30-12:00        | ... |
| ...         | ...      | ...       | ...               | ... |

**Optimized Trips Output**: The generated `optimized_trips.xlsx` file will contain columns like:

| TRIP ID | Shipment ID | MST_DIST | TRIP_TIME | Vehical_Type | CAPACITY_UTI | TIME_UTI | COV_UTI |
|---------|-------------|----------|-----------|--------------|--------------|----------|---------|
| T101    | S1, S2      | 15.2     | 45        | 3W           | 0.8          | 0.9      | 0.95    |
| T102    | S3, S4      | 18.4     | 60        | 4W-EV        | 0.75         | 0.85     | 0.92    |

---

## Contributing

Feel free to contribute by forking this repository and submitting a pull request. Contributions are welcome!

---

## License

This project is open-source and available under the [MIT License](LICENSE).

