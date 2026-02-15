import pandas as pd
import numpy as np
import os
import streamlit as st
from datetime import datetime, timedelta

DATA_PATH = "data/airline_dataset.csv"

# Global dictionary for Airport Coordinates (Lat, Lon)
AIRPORT_COORDINATES = {
    'DEL': {'lat': 28.5562, 'lon': 77.1000, 'city': 'Delhi'},
    'BOM': {'lat': 19.0912, 'lon': 72.8656, 'city': 'Mumbai'},
    'BLR': {'lat': 13.1986, 'lon': 77.7066, 'city': 'Bangalore'},
    'HYD': {'lat': 17.2403, 'lon': 78.4294, 'city': 'Hyderabad'},
    'CCU': {'lat': 22.6547, 'lon': 88.4467, 'city': 'Kolkata'},
    'MAA': {'lat': 12.9941, 'lon': 80.1709, 'city': 'Chennai'},
    'DXB': {'lat': 25.2532, 'lon': 55.3657, 'city': 'Dubai'},
    'LHR': {'lat': 51.4700, 'lon': -0.4543, 'city': 'London'},
    'SIN': {'lat': 1.3644, 'lon': 103.9915, 'city': 'Singapore'},
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'city': 'New York'},
    'GOA': {'lat': 15.3800, 'lon': 73.8314, 'city': 'Goa'},
    'PNQ': {'lat': 18.5821, 'lon': 73.9197, 'city': 'Pune'}
}

def normalize_columns(df):
    """Smartly renames columns to match what our code expects."""
    column_map = {
        'Flight Distance': ['flight distance', 'distance', 'dist', 'miles', 'flight_distance'],
        'Class': ['class', 'travel class', 'cabin_type', 'fare type'],
        'Departure Delay in Minutes': ['departure delay', 'dep_delay', 'delay'],
        'Arrival Delay in Minutes': ['arrival delay', 'arr_delay'],
        'Origin': ['origin', 'source', 'from'],
        'Destination': ['destination', 'dest', 'to']
    }
    
    df.columns = [str(c).strip() for c in df.columns]
    current_cols_lower = {c.lower(): c for c in df.columns}
    
    for standard_name, variations in column_map.items():
        if standard_name not in df.columns:
            for v in variations:
                if v in current_cols_lower:
                    df.rename(columns={current_cols_lower[v]: standard_name}, inplace=True)
                    break
    return df

def engineer_network_features(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if 'Date' not in df.columns or df['Date'].isnull().all():
        if 'validFrom' in df.columns:
             df['Date'] = pd.to_datetime(df['validFrom'], errors='coerce').fillna(datetime.now())
        else:
            end_date = datetime.now()
            days_back = np.random.randint(0, 365, size=len(df))
            df['Date'] = [end_date - timedelta(days=int(d)) for d in days_back]
            df['Date'] = pd.to_datetime(df['Date'])

    airport_codes = list(AIRPORT_COORDINATES.keys())
    
    if 'Origin' not in df.columns:
        df['Origin'] = np.random.choice(airport_codes, size=len(df))
    if 'Destination' not in df.columns:
        df['Destination'] = np.random.choice(airport_codes, size=len(df))
    
    df['Route'] = df['Origin'].astype(str) + " - " + df['Destination'].astype(str)
    
    df['Origin_Lat'] = df['Origin'].map(lambda x: AIRPORT_COORDINATES.get(x, {}).get('lat'))
    df['Origin_Lon'] = df['Origin'].map(lambda x: AIRPORT_COORDINATES.get(x, {}).get('lon'))
    df['Dest_Lat'] = df['Destination'].map(lambda x: AIRPORT_COORDINATES.get(x, {}).get('lat'))
    df['Dest_Lon'] = df['Destination'].map(lambda x: AIRPORT_COORDINATES.get(x, {}).get('lon'))
    
    return df

def engineer_financial_features(df):
    df = normalize_columns(df)
    df = engineer_network_features(df)
    
    if 'Flight Distance' not in df.columns:
        df['Flight Distance'] = np.random.randint(200, 4000, size=len(df))
    
    if 'Class' not in df.columns:
        df['Class'] = np.random.choice(['Eco', 'Business', 'Eco Plus'], size=len(df))

    if 'Departure Delay in Minutes' not in df.columns:
        conditions = [
            np.random.rand(len(df)) < 0.7,
            np.random.rand(len(df)) < 0.9
        ]
        choices = [0, np.random.randint(5, 30, size=len(df))]
        df['Departure Delay in Minutes'] = np.select(conditions, choices, default=np.random.randint(30, 120, size=len(df)))

    df['Ticket_Price'] = 50 + (df['Flight Distance'] * 0.12)
    df['Ticket_Price'] = df['Ticket_Price'] * np.random.uniform(0.8, 1.5, size=len(df))
    
    class_multiplier = df['Class'].map({'Eco': 1, 'Business': 2.5, 'Eco Plus': 1.5}).fillna(1)
    df['Revenue'] = df['Ticket_Price'] * class_multiplier
    
    df['Fuel_Cost'] = df['Flight Distance'] * 0.04 * np.random.uniform(0.9, 1.1, size=len(df))
    df['Profit'] = df['Revenue'] - df['Fuel_Cost']
    
    return df

def load_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            df = engineer_financial_features(df)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

# --- EMBEDDED DUMMY GENERATOR (FIXES THE ERROR) ---
def generate_dummy_data_file():
    """Generates sample data directly without needing external files."""
    try:
        # Create data directory if not exists
        if not os.path.exists("data"):
            os.makedirs("data")

        n_rows = 1000
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_rows),
            'Customer Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n_rows),
            'Age': np.random.randint(18, 80, n_rows),
            'Type of Travel': np.random.choice(['Personal', 'Business'], n_rows),
            'Class': np.random.choice(['Eco', 'Business', 'Eco Plus'], n_rows),
            'Flight Distance': np.random.randint(100, 4000, n_rows),
            'Departure Delay in Minutes': np.random.randint(0, 100, n_rows),
            'Arrival Delay in Minutes': np.random.randint(0, 100, n_rows),
        }
        df = pd.DataFrame(data)
        
        # Save
        file_path = "data/airline_dataset.csv"
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to generate data: {e}")
        return False