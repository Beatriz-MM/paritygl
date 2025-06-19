# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 21/04/2025
# Description: Converts JSON files with Galician comments to CSV format.
# Python version: 3.10.12

import os
import pandas as pd
import json

# IMPORTANT: Set the path where the JSON files are located
input_directory = ""

for filename in os.listdir(input_directory):
    if filename.startswith('gl_comments_') and filename.endswith('.json'):
        
        with open(os.path.join(input_directory, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Convert the JSON data to a pandas DataFrame
        df = pd.json_normalize(data)
        
        # Generate the name for the output CSV file
        csv_filename = f'csv_{filename.replace(".json", ".csv")}'
        
        # Save the DataFrame as a CSV file
        df.to_csv(os.path.join(input_directory, csv_filename), index=False, encoding='utf-8-sig')
        
        # Verify data integrity between the original JSON and the CSV
        df_csv = pd.read_csv(os.path.join(input_directory, csv_filename), encoding='utf-8-sig')
        if len(data) == len(df_csv):
            print(f"Conversion completed successfully for {filename}. Saved as {csv_filename}.")
        else:
            print(f"Error: Number of items in CSV does not match original JSON for {filename}.")
            print(f"Items in JSON: {len(data)}, Items in CSV {len(df_csv)}")
