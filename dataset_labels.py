import os
import re
import csv
from pathlib import Path
import concurrent.futures

# Configuration
LOG_DIR = Path("dataset/abc_logs")
OUTPUT_CSV = Path("dataset/labels.csv")

# Regex Patterns (Tailored exactly to your successful log output)
# Extracts the recipe string
recipe_pattern = re.compile(r"^RECIPE:\s*(.*)")

# Extracts area, delay, and power numbers from the stats line
stats_pattern = re.compile(r"area\s*=\s*([0-9.]+)\s+delay\s*=\s*([0-9.]+).*power\s*=\s*([0-9.]+)")

def process_log_file(filepath):
    # Extract circuit name and run ID from the filename (e.g., "max_run71.log")
    base_name = filepath.stem
    parts = base_name.split("_run")
    
    if len(parts) != 2:
        return None, f"Warning: Unrecognized filename format {filepath.name}. Skipping."
        
    circuit_name = parts[0]
    run_id = parts[1]
    
    recipe_str = None
    area = None
    delay = None
    power = None
    
    # Read the file and apply Regex
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Check for Recipe
                if recipe_str is None:
                    recipe_match = recipe_pattern.match(line)
                    if recipe_match:
                        # Clean up the trailing semicolon and spaces
                        recipe_str = recipe_match.group(1).strip().rstrip(';')
                        continue
                
                # Check for Stats
                stats_match = stats_pattern.search(line)
                if stats_match:
                    area = float(stats_match.group(1))
                    delay = float(stats_match.group(2))
                    power = float(stats_match.group(3))
                    break # Found the stats, no need to read the rest of the file
    except Exception as e:
        return None, f"Error reading {filepath.name}: {e}"
    
    # If we successfully found all data points, return the row
    if recipe_str and area is not None and delay is not None and power is not None:
        return [circuit_name, run_id, recipe_str, power, area, delay], None
    else:
        return None, f"Warning: Missing data in {filepath.name}. Skipping."

def parse_all_logs():
    if not LOG_DIR.exists():
        print(f"Error: Directory {LOG_DIR} not found.")
        return

    # Iterate through all log files
    log_files = list(LOG_DIR.glob('*.log'))
    print(f"Found {len(log_files)} log files. Parsing using multiprocessing...")

    dataset_rows = []
    
    # Use ProcessPoolExecutor to distribute file reading across CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_log_file, log_files)
        
        for row, error_msg in results:
            if row:
                dataset_rows.append(row)
            if error_msg:
                pass # You can uncomment this if you want to see warnings: print(error_msg)

    # Sort the dataset iteratively by Circuit Name, then by Run ID numerically
    dataset_rows.sort(key=lambda x: (x[0], int(x[1])))

    # Make sure output dir exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Write everything to a clean CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write headers
        writer.writerow(['Circuit', 'Run_ID', 'Recipe', 'Power', 'Area', 'Delay'])
        # Write data
        writer.writerows(dataset_rows)
        
    print(f"Success! Extracted {len(dataset_rows)} valid records and saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    parse_all_logs()