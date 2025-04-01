import os
import json
import pandas as pd
import numpy as np

# Define the expected datasets and controllers.
datasets = ['zara1', 'zara2', 'univ', 'hotel', 'eth']
controllers = ['mpc', 'cp-mpc', 'cc', 'eacp-mpc']

records = []
input_dir = './metric/processed'

for dataset in datasets:
    for controller in controllers:
        filename = f"{dataset}_{controller}.json"
        filepath = os.path.join(input_dir, filename)

        # Default to NaN in case of missing file or error.
        collisions = np.nan
        cost = np.nan
        travel_time = np.nan
        infeasible = np.nan
        miscoverage = np.nan

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                # Retrieve values from JSON (keys: "collision", "cost", "time", "infeasible", "miscoverage")
                collisions = data.get("collision", np.nan)
                cost = data.get("cost", np.nan)
                travel_time = data.get("time", np.nan)
                infeasible = data.get("infeasible", np.nan)
                miscoverage = data.get("miscoverage", np.nan)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        records.append({
            "Dataset": dataset.capitalize(),  # Capitalize first letter of dataset name
            "Controller": controller.upper(),  # Convert controller name to uppercase
            "Collisions": collisions,
            "Cost": cost,
            "Travel Time": travel_time,
            "Infeasible": infeasible,
            "Miscoverage": miscoverage
        })

# Create a DataFrame with a MultiIndex for Dataset and Controller.
df = pd.DataFrame(records)
df = df.set_index(["Dataset", "Controller"])

# Format each column with the desired number of decimal places.
# For "Collisions" and "Infeasible": 3 decimal places.
df['Collisions'] = df['Collisions'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "nan")
df['Infeasible'] = df['Infeasible'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "nan")
df['Miscoverage'] = df['Miscoverage'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "nan")

# For "Cost", "Travel Time", and "Miscoverage": 2 decimal places.
df['Cost'] = df['Cost'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "nan")
df['Travel Time'] = df['Travel Time'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "nan")


# Generate the LaTeX table code.
# Note: Since the numeric columns are now strings with custom formatting, we do not use the float_format argument.
latex_table = df.to_latex(multirow=True, multicolumn=True, escape=False)

# Write the LaTeX code to a text file.
output_file = 'output_table.txt'
with open(output_file, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table code written to {output_file}")