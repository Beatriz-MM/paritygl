# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Extracts relevant fields (postUrl, id, text) from multiple JSON files of a specific category, and 
# removes basic user mentions (@) as an initial anonymization step to clean the data for model training.
# Python version: 3.10.12

import json

#Path or Paths to the raw (unfiltered and uncleaned) JSON datasets of a specific category
path1 = ""
path2 = ""
path3 = ""
json_paths = [path1, path2, path3]

# Path to the output JSON file where all cleaned posts of the selected category will be combined
destination_json_path = ""
 
cleaned_comments = []

for json_path in json_paths:
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        if all(key in item for key in ["postUrl", "id", "text"]):
            # Remove words starting with "@" from the "text" field
            clean_text = ' '.join(word for word in item["text"].split() if not word.startswith('@'))
            
            # Create a new dictionary with the required keys and add it to the list
            cleaned_comments.append({
                "postUrl": item["postUrl"],
                "id": item["id"],
                "text": clean_text
            })
with open(destination_json_path, 'w', encoding='utf-8') as outfile:
    json.dump(cleaned_comments, outfile, indent=4, ensure_ascii=False)


# Verify the results
with open(destination_json_path, 'r', encoding='utf-8') as file:
    cleaned_data = json.load(file)

if len(cleaned_comments) == len(cleaned_data):
    print("All relevant elements have been successfully cleaned and saved")
else:
    print("ERROR! Some relevant elements may be missing in the cleaned file")
