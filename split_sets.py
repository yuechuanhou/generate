import os
import random

# Set the path to the directory containing the folders you want to rename
path_to_directory = "/home/yuechuan/vlr/project/neu-nbv/scripts/neural_rendering/data/dataset/urbanscene_cutscene_60"

# Fetch all the folder names in the directory
folder_names = [name for name in os.listdir(path_to_directory) if os.path.isdir(os.path.join(path_to_directory, name))]

# Sort the folders
folder_names.sort()

# First, rename all folders to a temporary name
temp_prefix = "temp_rename_"
for i, folder_name in enumerate(folder_names):
    temp_name = f"{temp_prefix}{i}"
    old_path = os.path.join(path_to_directory, folder_name)
    new_path = os.path.join(path_to_directory, temp_name)
    os.rename(old_path, new_path)
    folder_names[i] = temp_name  # Update the list with temporary folder names

# Then, rename them to the final 'scanX' format
for i, temp_name in enumerate(folder_names):
    final_name = f"scan{i}"
    old_path = os.path.join(path_to_directory, temp_name)
    new_path = os.path.join(path_to_directory, final_name)
    os.rename(old_path, new_path)
    folder_names[i] = final_name  # Update the list with final folder names

# Shuffle the folder list for random splitting
random.shuffle(folder_names)

# Define the splitting logic
total_folders = len(folder_names)
train_split = int(0.70 * total_folders)
test_split = int(0.15 * total_folders)

train_set = folder_names[:train_split]
test_set = folder_names[train_split:train_split + test_split]
validation_set = folder_names[train_split + test_split:]

# Function to write folder names to a file in the specified directory
def write_to_file(file_name, folder_set):
    file_path = os.path.join(path_to_directory, file_name)
    with open(file_path, 'w') as file:
        for folder in folder_set:
            file.write(folder + '\n')

# Write each set to its respective file in the specified directory
write_to_file("train.lst", train_set)
write_to_file("test.lst", test_set)
write_to_file("val.lst", validation_set)

print("Folders have been divided into train, test, and validation sets and recorded in .lst files in the specified directory.")