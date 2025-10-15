import os
import json
import zipfile
from pathlib import Path
import argparse

def replace_root_path(zip_file_path, database_dir):
    """
    Read a zip file, replace 'video_file_root' in JSON files, and save to the specified directory.
    
    Args:
        zip_file_path: Path to the zip file.
        database_dir: Directory for the database.
    """
    zip_file_name = Path(zip_file_path).stem

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Iterate through all files in the zip
        for file_name in zip_ref.namelist():
            if file_name.endswith('.json'):
                # Read the JSON file
                with zip_ref.open(file_name) as json_file:
                    data = json.load(json_file)
                
                # Replace video_file_root
                new_root = os.path.join(database_dir, zip_file_name)
                data['video_file_root'] = new_root
                
                # Create output directory
                json_name = Path(file_name).stem
                output_dir = os.path.join(database_dir, zip_file_name, json_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the JSON file
                output_path = os.path.join(output_dir, 'database.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"Processed: {file_name} -> {output_path}")

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Replace video_file_root path in JSON files inside a zip archive')
    parser.add_argument('zip_file', type=str, help='Path to the zip file')
    parser.add_argument('database_dir', type=str, help='Path to the database directory')
    
    args = parser.parse_args()
    
    zip_file = args.zip_file
    database_dir = args.database_dir

    replace_root_path(zip_file, database_dir)