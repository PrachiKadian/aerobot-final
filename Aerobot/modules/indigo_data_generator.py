import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_indigo_dataset():
    print("🚀 Generating large-scale Indigo dataset...")
    
    # Configuration
    ROW_COUNT = 10000  # Big dataset
    AIRLINE = "IndiGo"
    AIRPORTS = ['DEL', 'BOM', 'BLR', 'HYD', 'CCU', 'MAA', 'DXB', 'SIN', 'LHR', 'JFK', 'GOA', 'PNQ']
    CLASSES = ['Eco', 'Business', 'Eco Plus']
    
    data = []
    
    # Start from 1 year ago
    start_date = datetime.now() - timedelta(days=365)
    
    print(f"   - Simulating {ROW_COUNT} flights...")
    
    for _ in range(ROW_COUNT):
        # 1. Route & Schedule
        date = start_date + timedelta(days=random.randint(0, 365))
        flight_num = f"6E-{random.randint(100, 9999)}"
        origin = random.choice(AIRPORTS)
        dest = random.choice([a for a in AIRPORTS if a != origin])
        travel_class = random.choices(CLASSES, weights=[70, 20, 10], k=1)[0] # Mostly Eco
        
        # 2. Physics (Distance & Time)
        # Approximate distance in km
        dist = random.randint(500, 9000) 
        # Duration: ~800km/h + 30 mins taxi/takeoff
        duration_mins = int(dist / 800 * 60) + 30 
        
        # 3. Time Slots
        hour = random.randint(0, 23)
        minute = random.choice([0, 5, 10, 15, 30, 45, 50])
        
        sched_dep = datetime(date.year, date.month, date.day, hour, minute)
        sched_arr = sched_dep + timedelta(minutes=duration_mins)
        
        # 4. The "Real World" Chaos (Delays)
        # 75% On Time, 20% Minor Delay, 5% Major Delay
        rand_val = random.random()
        
        if rand_val < 0.75:
            # On time or slightly early
            dep_delay = random.randint(-10, 5)
        elif rand_val < 0.95:
            # Minor delay (10-45 mins)
            dep_delay = random.randint(10, 45)
        else:
            # Major delay (1-4 hours)
            dep_delay = random.randint(60, 240)
            
        # Arrival delay is Departure Delay +/- some time made up in air
        arr_delay = dep_delay + random.randint(-15, 10)
        
        # Calculate Actual Times
        act_dep = sched_dep + timedelta(minutes=dep_delay)
        act_arr = sched_arr + timedelta(minutes=arr_delay)
        
        # 5. Append Row
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Airline": AIRLINE,
            "Flight_Number": flight_num,
            "Origin": origin,
            "Destination": dest,
            "Class": travel_class,
            "Flight Distance": dist,
            # Time Logs
            "Scheduled_Departure": sched_dep.strftime("%H:%M"),
            "Actual_Departure": act_dep.strftime("%H:%M"),
            "Scheduled_Arrival": sched_arr.strftime("%H:%M"),
            "Actual_Arrival": act_arr.strftime("%H:%M"),
            # Metrics for Dashboard
            "Departure Delay in Minutes": max(0, dep_delay), # Only positive for stats
            "Arrival Delay in Minutes": max(0, arr_delay)
        })
        
    df = pd.DataFrame(data)
    
    # Save
    if not os.path.exists("data"):
        os.makedirs("data")
        
    file_path = "data/indigo_large_dataset.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Success! Generated '{file_path}' with {ROW_COUNT} rows.")
    print("   -> Go to your Dashboard and upload this file!")

if __name__ == "__main__":
    generate_indigo_dataset()