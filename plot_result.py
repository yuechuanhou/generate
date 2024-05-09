import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# def plot_data(data, label, window_length, polyorder):
#     """Plots the data with a given label, normalizing the timestamps to start from the same point and smoothing."""
#     data_np = np.array(data)
#     timestamps = data_np[:, 0]
#     values = data_np[:, 2]  # Assuming the value is in the second column

#     # Normalize timestamps so that the first timestamp starts at 0
#     normalized_timestamps = timestamps - timestamps[0]

#     # Check if the window length is valid for the data size and polyorder
#     if window_length > len(values) or window_length < polyorder + 1:
#         raise ValueError("Window length must be less than or equal to the data size and greater than polyorder.")

#     # Apply Savitzky-Golay filter to smooth the values
#     smoothed_values = savgol_filter(values, window_length, polyorder)

#     plt.plot(normalized_timestamps, smoothed_values, marker='o', linestyle='-', label=label)

def plot_data(data, label, window_length, polyorder):
    """Plots the y-values of the data against their index with a given label, applying smoothing."""
    data_np = np.array(data)
    values = data_np[:, 2]  # Assuming the value is in the second column
    indices = np.arange(len(values))

    # Ensure the window length is an odd number and less than the number of data points
    if window_length % 2 == 0:
        window_length += 1 
    if window_length > len(values):
        window_length = len(values) - 1 if len(values) % 2 else len(values) - 2

    # Apply Savitzky-Golay filter to smooth the values
    smoothed_values = savgol_filter(values, window_length, polyorder)

    plt.plot(indices, smoothed_values, marker='o', linestyle='-', label=label)



def main():
    file_paths = [
        'urbanscene_training_30.json',
        'urbanscene_training_60.json',
        'urbanscene_training_90.json'
    ]
    legend_names = ['30 degrees', '60 degrees', '90 degrees'] 
    output_dir = './plots'  
    output_filename = 'combined_psnr.png'  
    full_output_path = os.path.join(output_dir, output_filename)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(16, 9))

    # Loop through each file path and plot the data with custom legend names
    # for i, file_path in enumerate(file_paths):
    #     data = read_json_file(file_path)
    #     # Adjust the window_length and polyorder accordingly
    #     window_length = 51  
    #     polyorder = 3       
    #     plot_data(data, label=legend_names[i], window_length=window_length, polyorder=polyorder)

    window_length = 51  # Choose an odd number
    polyorder = 3       # Choose the polynomial order

    for i, file_path in enumerate(file_paths):
        data = read_json_file(file_path)
        plot_data(data, label=legend_names[i], window_length=window_length, polyorder=polyorder)

    plt.title("Average PSNR")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)
    
    # Save the combined plot to the specified path
    plt.savefig(full_output_path)
    print(f"Figure saved to {full_output_path}")
    
    # plt.show()

if __name__ == "__main__":
    main()

