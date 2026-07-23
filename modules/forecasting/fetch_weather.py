# =============================================
#              MODULE IMPORTS
# =============================================

import requests
import pandas as pd
from pathlib import Path

# =============================================
#              FUNCTIONAL MODULE
# =============================================

def fetch_northern_region_weather():
    print("Initiating Open-Meteo API connection for the Northern Grid...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # 1. Define the Major Load Centers of the Northern Region
    cities = {
        "Delhi": {"lat": 28.6139, "lon": 77.2090},
        "Lucknow": {"lat": 26.8467, "lon": 80.9462},
        "Jaipur": {"lat": 26.9124, "lon": 75.7873}
    }
    
    # Format coordinates into comma-separated strings for the API
    lats = ",".join([str(c["lat"]) for c in cities.values()])
    lons = ",".join([str(c["lon"]) for c in cities.values()])
    
    # 2. The exact query parameters spanning 5+ years
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": "2019-01-01", 
        "end_date": "2024-04-30",   
        "hourly": "temperature_2m", # Dry bulb temperature
        "timezone": "Asia/Kolkata"  # Force local Indian Standard Time
    }
    
    # 3. Execute the GET request
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"API Error: {response.text}")
        return
        
    data = response.json()
    
    # 4. Extract data for all three cities and merge them
    # Open-Meteo returns a list of dictionaries when multiple coordinates are requested
    df_list = []
    
    for i, city_name in enumerate(cities.keys()):
        city_data = data[i]['hourly']
        temp_df = pd.DataFrame({
            'Datetime': pd.to_datetime(city_data['time']),
            f'{city_name}_T2M': city_data['temperature_2m']
        })
        # Set datetime as index for easy merging
        temp_df.set_index('Datetime', inplace=True)
        df_list.append(temp_df)
        
    # Combine all city columns into one DataFrame
    combined_df = pd.concat(df_list, axis=1)
    
    # 5. Create the Regional Feature (The Average Temperature)
    combined_df['Northern_Region_Avg_T2M'] = combined_df.mean(axis=1)
    
    # 6. Save the CSV
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    file_path = project_root / 'data' / 'raw_data' / 'Northern_Region_Hourly_Weather_2019_2024.csv'
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.reset_index().to_csv(file_path, index=False)
    
    print(f"Success! Saved {len(combined_df)} hourly records to {file_path}")
    print("\nData Preview:")
    print(combined_df.head())

# ==================================================
#              TESTING & EXECUTION
# ==================================================

if __name__ == "__main__":
    fetch_northern_region_weather()