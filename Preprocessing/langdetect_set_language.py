# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Detects the language of each comment in a JSON file using the ftlangdetect library
# and saves the results with the detected language included.
# Python version: 3.10.12

from ftlangdetect import detect
import os
import json 


def open_file(comments_file):
    with open(comments_file, 'r', encoding='utf-8') as file:
        return json.load(file)


def detect_language(comments_file, output_file):
    """
    Detects the language of each comment using ftlangdetect and adds it as a new field.
    If the language is not Galician ('gl'), Portuguese ('pt'), or Spanish ('es'), it is labeled 'other'.

    Args:
        comments_file (str): Path to the input JSON file containing comments with a 'text' field.
        output_file (str): Path to the output JSON file where each comment includes a 'language' field.
    """ 
    comments = open_file(comments_file)
    comments_with_language = []

    for comment in comments:
        text = comment.get('text', '')

        detected_lang_tuple = detect(text)
        detected_language = detected_lang_tuple['lang']

        if detected_language == 'gl':
            comment['language'] = 'gl'
        elif detected_language == 'pt':
            comment['language'] = 'pt'
        elif detected_language == 'es':
            comment['language'] = 'es'
        else:
            comment['language'] = 'other'

        reordered_comment = {
            "postUrl": comment["postUrl"],
            "id": comment["id"],
            "language": comment["language"],
            "text": comment["text"]
        }
        comments_with_language.append(reordered_comment)

    with open(output_file, 'w', encoding='utf-8') as file_out:
        json.dump(comments_with_language, file_out, ensure_ascii=False, indent=4)


# IMPORTANT: Set the path to the directory containing JSON files with mentions removed
files_directory = ""

# Process all JSON files in the directory that start with "nomentions_"
for file in os.listdir(files_directory):
    if file.startswith("nomentions_") and file.endswith(".json"):

        # Generate output filename by replacing "nomentions_" with "langdetect_"
        new_name = file.replace("nomentions_", "langdetect_")
        full_input_path = os.path.join(files_directory, file)
        full_output_path = os.path.join(files_directory, new_name)

        print(f"Processing {file}...")
        detect_language(full_input_path, full_output_path)
        print(f"File saved as {new_name}")