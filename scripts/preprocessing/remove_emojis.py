# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Removes emojis from the "text" field of JSON files and saves the cleaned data.
# Python version: 3.10.12

import json
import os
import re
import emoji

# Compile a regex pattern to match any emoji
EMOJI_PATTERN = re.compile(f"[{re.escape(''.join(emoji.EMOJI_DATA.keys()))}]")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def remove_emojis(text):
    """
    Replace the emojis with an empty string and preserve the rest of the text.

    Args:
        text (str): The input text containing potential emojis.
    Returns:
        str: The input text with all emojis removed.
    """
    return EMOJI_PATTERN.sub(r'', text) 

def process_json(input_path, output_path):
    """
    Process a JSON file by removing emojis from the "text" field of each item
    and saving the cleaned data to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path where the processed JSON file will be saved.
    """
    data = load_json(input_path)
    processed_data = []

    for item in data:
        if 'text' in item:
            text_without_emojis = remove_emojis(item['text'])
            item['text'] = text_without_emojis
        processed_data.append(item)

    save_json(processed_data, output_path)
    print(f"Processed data saved in {output_path}")


# Directory containing JSON files starting with "langdetect_"
json_files_directory = ""

if not json_files_directory:
    print("Error: Please specify a valid directory path.")
else:
    directory = json_files_directory
    for file in os.listdir(json_files_directory):
        if file.startswith("langdetect_") and file.endswith(".json"):

            # Generate output filename by replacing "langdetect_" with "noemojis_"
            output_filename = file.replace("langdetect_", "noemojis_")
                
            input_path = os.path.join(json_files_directory, file)
            output_path = os.path.join(json_files_directory, output_filename)
                
            process_json(input_path, output_path)