import os
import pandas as pd

def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Parse timestamp and remove seconds
    df['ts'] = pd.to_datetime(df['ts']).dt.strftime('%Y-%m-%d %H:%M')

    # Remove duplicate entries
    df = df.drop_duplicates(subset=['ts'])

    # Write to output CSV file
    df.to_csv(output_file, index=False)

def organize_dataset(input_folder, output_folder):
    for month_folder in os.listdir(input_folder):
        month_path = os.path.join(input_folder, month_folder)
        output_month_path = os.path.join(output_folder, month_folder)

        # Create output month folder if it doesn't exist
        if not os.path.exists(output_month_path):
            os.makedirs(output_month_path)

        for file_name in os.listdir(month_path):
            if file_name.endswith(".csv"):
                input_file = os.path.join(month_path, file_name)
                output_file = os.path.join(output_month_path, file_name)

                process_file(input_file, output_file)

# Define input and output folders
input_folder = "Training Dataset"
output_folder = "Ordered Dataset"

# Organize the dataset
organize_dataset(input_folder, output_folder)
