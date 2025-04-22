# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Extracts unique emojis from comment JSON files and saves them to a text file.
# Python version: 3.10.12

import os
import json
import re
import emoji

# IMPORTANT: insert the corresponding path to the folder where all the JSON files are located
path_comments = ""

def get_emoji_regexp():
    """
    Generate a compiled regular expression pattern to match all emojis 
    defined in the emoji.EMOJI_DATA set.

    Returns:
        re.Pattern: A compiled regular expression object that matches emojis.
    """ 
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)


def extract_emojis(text, emoji_pattern):
    """
    Extract all emojis from a given text using a provided regex pattern.

    Args:
        text (str): The input text containing potential emojis.
        emoji_pattern (re.Pattern): Compiled regex pattern to detect emojis.
    Returns:
        list: A list of emojis found in the text.
    """ 
    return emoji_pattern.findall(text)


# Set to store unique emojis (no duplicates)
unique_emojis = set()

# Process each JSON file in the directory
for filename in os.listdir(path_comments):
    # In my case, it looks for files that start with "comments_" and end with ".json"
    if filename.startswith('comments_') and filename.endswith('.json'):
        filepath = os.path.join(path_comments, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Extract emojis from each "text" field
            for item in data:
                text = item.get("text", "")
                emojis_in_text = extract_emojis(text, get_emoji_regexp())
                unique_emojis.update(emojis_in_text)

output_file_path = os.path.join(path_comments, 'emojis.txt')

# Save the emojis to a text file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for emoji_char in unique_emojis:
        file.write(f"{emoji_char}\n")

print(f"Emojis saved in {output_file_path}. Total unique emojis: {len(unique_emojis)}")