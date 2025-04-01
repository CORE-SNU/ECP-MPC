import os
import json
import glob


def compute_average(lst):
    try:
        if not lst:  # if the list is empty, return nan
            return float('nan')
        return sum(lst) / len(lst)
    except Exception:
        return float('nan')


# Define input and output directories
input_dir = './metric'
output_dir = os.path.join(input_dir, 'processed')
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file in the input directory
for filepath in glob.glob(os.path.join(input_dir, "*.json")):
    with open(filepath, 'r') as infile:
        data = json.load(infile)

    # Create a new dictionary with averaged values for each key
    avg_data = {key: compute_average(values) for key, values in data.items()}

    # Build output filepath in the processed folder using the same base filename
    output_filepath = os.path.join(output_dir, os.path.basename(filepath))

    with open(output_filepath, 'w') as outfile:
        json.dump(avg_data, outfile, indent=4)

    print(f"Processed {filepath} -> {output_filepath}")