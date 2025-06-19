# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Performs a more precise anonymization step by removing user mentions (@username) 
# from JSON text fields, while preserving inclusive forms like '@s'.
# Python version: 3.10.12


import os
import json
import re

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def remove_mention(text):
    """
    Replace mentions with an empty string and preserve 
    inclusive language like '@s' (e.g., l@s, vosotr@s)

    Args:
        text (str): Input text containing potential mentions.
    Returns:
        str: Text without user mentions.
    """ 
    words = text.split()
    cleaned_words = []
    for word in words:
        if '@' in word:
            # Find all mentions starting with '@' followed by alphanumeric characters or underscores
            possible_mentions = re.findall(r'@\w+', word)
            for mention in possible_mentions:
                # Remove the '@' at the beginning
                username = mention[1:]  
                # Exclude inclusive-language mentions like '@s' or '@S'
                if username.lower() == 's':  
                    continue
                    
                # Replace the mention with an empty string
                word = word.replace(f'@{username}', '')
        cleaned_words.append(word)

    result = ' '.join(cleaned_words)
    # Clean up redundant spaces that may result from mention removal
    return re.sub(r'\s{2,}', ' ', result).strip()


# IMPORTANT: Set the correct input file path
input_path = ""

# IMPORTANT: Set the output file path manually, ensuring the filename starts with "nomentions_"
output_path = ""

data = load_json(input_path)
processed_data = []
for item in data:
    # Ensure the item has the required fields
    if all(key in item for key in ["postUrl", "id", "text"]):
        item["text"] = remove_mention(item["text"])
        processed_data.append(item)

save_json(processed_data, output_path)
print(f"Processed file {input_path}. Result saved in {output_path}")