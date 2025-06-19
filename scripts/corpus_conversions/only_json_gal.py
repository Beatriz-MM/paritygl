# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 21/04/2025
# Description: Filters comments in Galician and Portuguese from multiple JSON files,
# converts their text to lowercase, and saves the filtered content to new JSON files. 
# Python version: 3.10.12

import pandas as pd
import glob
import os

# IMPORTANT: Set the path where the JSON files are located
folder_path = ""

# Find all JSON files, named "sinemojis_comments_" in the folder
json_files = glob.glob(os.path.join(folder_path, 'noemojis_comments_*.json'))

for file_path in json_files:
    try: 
        data = pd.read_json(file_path)
    
        # Filter only Galician and Portuguese texts
        filtered_data = data[data['language'].isin(['gl', 'pt'])]

        # Keep only the required fields: id, language, and text
        filtered_data = filtered_data[['id', 'language', 'text']]

        # Convert text to lowercase
        filtered_data['text'] = filtered_data['text'].str.lower().fillna('')
    
        base_name = os.path.basename(file_path)
        new_file_name = 'gl_comments_' + base_name[len('noemojis_comments_'):]
        filtered_file_path = os.path.join(folder_path, new_file_name)
    
        filtered_data.to_json(filtered_file_path, orient='records', lines=False, force_ascii=False, indent=4)
    
        print(f"Filtered file saved as: {filtered_file_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
