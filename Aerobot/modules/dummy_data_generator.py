import pandas as pd
import numpy as np
import os

def generate_dummy_csv():
    print("Generating dummy airline data...")
    
    # Create data directory if not exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Generate 1000 rows
    n_rows = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_rows),
        'Customer Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n_rows),
        'Age': np.random.randint(18, 80, n_rows),
        'Type of Travel': np.random.choice(['Personal', 'Business'], n_rows),
        'Class': np.random.choice(['Eco', 'Business', 'Eco Plus'], n_rows),
        'Flight Distance': np.random.randint(100, 4000, n_rows),
        'Inflight wifi service': np.random.randint(1, 6, n_rows),
        'Ease of Online booking': np.random.randint(1, 6, n_rows),
        'Gate location': np.random.randint(1, 6, n_rows),
        'Food and drink': np.random.randint(1, 6, n_rows),
        'Online boarding': np.random.randint(1, 6, n_rows),
        'Seat comfort': np.random.randint(1, 6, n_rows),
        'Inflight entertainment': np.random.randint(1, 6, n_rows),
        'On-board service': np.random.randint(1, 6, n_rows),
        'Leg room service': np.random.randint(1, 6, n_rows),
        'Baggage handling': np.random.randint(1, 6, n_rows),
        'Checkin service': np.random.randint(1, 6, n_rows),
        'Inflight service': np.random.randint(1, 6, n_rows),
        'Cleanliness': np.random.randint(1, 6, n_rows),
        'Departure Delay in Minutes': np.random.randint(0, 100, n_rows),
        'Arrival Delay in Minutes': np.random.randint(0, 100, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Save
    file_path = "data/airline_dataset.csv"
    df.to_csv(file_path, index=False)
    print(f"Success! Data saved to {file_path}")

if __name__ == "__main__":
    generate_dummy_csv()