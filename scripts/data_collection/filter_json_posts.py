# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 03/04/2025
# Description: Clean the original JSON file to create a new one with URLs, associated account, 
# comments and publication date, filtering posts from September 2023 onward.  
# Python version: 3.10.12

import json
from datetime import datetime


def filter_by_date (json_data, start_date):
    """
    Filter posts with timestamp equal to or after the given start date.

    Args:
        json_data (list): List of post dictionaries.
        start_date (datetime): Minimum date for filtering.
    Returns:
        list: Filtered posts.
    """
    filtered_posts_data = []
    for data in json_data:
        if data.get("timestamp"):
            post_date = datetime.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
            if post_date >= start_date:
                filtered_posts_data.append({
                    "inputUrl": data.get("inputUrl"),
                    "url": data.get("url"),  
                    "commentsCount": data.get("commentsCount"),
                    "timestamp": data.get("timestamp")
                })
    return filtered_posts_data

# IMPORTANT: insert the corresponding path
# Example: "~/Posts/parte3/dataset_instagram-post-scraper_2024-04-14_22-56-56-638.json"
json_path = ""

# Specify encoding="utf-8" to correctly read the JSON file in Galician and Spanish
with open(json_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)

# Filter by date: September 2023
start_date = datetime(2023, 9, 1)
filtered_posts_data = filter_by_date(json_data, start_date)

#for posts_data in filtered_posts_data:
  #   print("Perfil:", posts_data["inputUrl"])
  #  print("URL:", posts_data["url"])
  #  print("Número de comentarios:", posts_data["commentsCount"])
  #  print("Fecha:", posts_data["timestamp"])
  #  print()


with open("filtered_posts_data.json", "w", encoding="utf-8") as file:
    json.dump(filtered_posts_data, file, indent=4)
print("The filtered profiles have been saved in 'filtered_posts_data.json'")

