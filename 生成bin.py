import pandas as pd
import struct
import numpy as np
import os

def save_programme_to_bin(excel_path, output_bin_path):
    """
    Save the 'Programme' column from an Excel file to a binary file, with each value stored as a 4-byte integer.

    Args:
        excel_path (str): Path to the input Excel file (e.g., 'training_set.xlsx').
        output_bin_path (str): Path to the output binary file (e.g., 'programme_labels.bin').

    Returns:
        None
    """
    # Validate input file existence
    if not os.path.exists(excel_path):
        raise IOError(f"Excel file not found: {excel_path}")

    # Read Excel file
    try:
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
    except Exception as e:
        raise IOError(f"Error reading Excel file {excel_path}: {str(e)}")

    # Extract Programme column
    if 'Programme' not in df.columns:
        raise ValueError("Excel file must contain a 'Programme' column")

    # Check for missing values in Programme column
    if df['Programme'].isnull().any():
        raise ValueError("Programme column contains missing values")

    # Convert to numpy array and ensure integer type
    programme_labels = df['Programme'].values
    labels_array = np.array(programme_labels, dtype=np.int32)

    # Validate that all labels are integers
    if not np.all(labels_array == labels_array.astype(int)):
        raise ValueError("All Programme values must be integers")

    # Save to binary file
    try:
        with open(output_bin_path, 'wb') as bin_file:
            for label in labels_array:
                # Pack integer into 4-byte binary format ('i' for 32-bit signed integer)
                binary_data = struct.pack('i', label)
                bin_file.write(binary_data)
        print(f"Successfully saved {len(labels_array)} Programme labels to {output_bin_path}")
    except Exception as e:
        raise IOError(f"Error writing to binary file {output_bin_path}: {str(e)}")


# Example usage
if __name__ == "__main__":
    excel_path = "cw3要交的/training_set.xlsx"
    output_bin_path = "programme_labels.bin"
    save_programme_to_bin(excel_path, output_bin_path)