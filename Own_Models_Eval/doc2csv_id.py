# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 22/05/2025
# Description:Converts a .txt file into a CSV of lowercase negative text samples with sequential IDs for NLP tasks.
# Python version: 3.10.12

import pandas as pd

input_txt_path = ""
output_csv_path = ""

# Read lines, remove extra spaces, convert to lowercase
with open(input_txt_path, "r", encoding="utf-8") as file:
    lines = [line.strip().lower() for line in file if line.strip()]

df = pd.DataFrame({
    "id": list(range(1, len(lines) + 1)),  # IDs start at 1
    "text": lines
})

df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")