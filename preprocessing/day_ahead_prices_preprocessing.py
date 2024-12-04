import os
import pandas as pd

# Define input folder and output file
input_folder = '../data/raw/day_ahead_prices'  # Adjust the relative path if needed
output_file = '../data/processed/day_ahead_prices/hub_prices.csv'  # Adjust the output path

# Initialize an empty DataFrame for final output
final_df = pd.DataFrame()

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_folder, file_name)
        
        # Read the CSV, skipping metadata rows starting with "C"
        df = pd.read_csv(
            file_path, 
            skiprows=5,  # Skip metadata rows
            header=None, 
            names=[
                "Type", "Date", "Hour Ending", "Location ID", "Location Name", 
                "Location Type", "LMP", "Energy Component", "Congestion Component", "Marginal Loss Component"
            ]
        )

        # Filter for the hub '.H.INTERNAL_HUB'
        hub_data = df[df["Location Name"] == '.H.INTERNAL_HUB']

        # Pivot to get 24 hourly prices in columns for the given date
        if not hub_data.empty:
            hub_data = hub_data.pivot(index='Date', columns='Hour Ending', values='LMP')
            hub_data.reset_index(inplace=True)
            hub_data.columns.name = None  # Clean up column names

            # Convert 'Date' column to datetime
            hub_data['Date'] = pd.to_datetime(hub_data['Date'])

            # Filter to only include weekdays (Monday=0, ..., Sunday=6)
            hub_data = hub_data[hub_data['Date'].dt.weekday < 5]
            
            # Append to the final DataFrame
            final_df = pd.concat([final_df, hub_data], ignore_index=True)

# Save the aggregated data to a CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output folder exists
final_df.to_csv(output_file, index=False)
print(f"Aggregated data saved to {output_file}")
