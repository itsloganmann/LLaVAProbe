import zipfile
import json
import pandas as pd
import os

# Define paths
zip_path = "../data_processing/data/raw/v2_Annotations_Train_mscoco.zip"  # Change to your ZIP file path
extract_path = "extracted_data"
save_directory = "../data_processing/data/raw"  
json_output_path = os.path.join(save_directory, "output.json")

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Ensure save directory exists
os.makedirs(save_directory, exist_ok=True)
print("File exists:", os.path.exists(zip_path))

# Convert CSV files in the ZIP to JSON
json_data = {}
for file in os.listdir(extract_path):
    if file.endswith(".csv"):  # Process CSV files
        csv_path = os.path.join(extract_path, file)
        df = pd.read_csv(csv_path)
        json_data[file] = df.to_dict(orient="records")  # Convert DataFrame to JSON format

# Save JSON to data/raw
with open(json_output_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Converted JSON saved to {json_output_path}")