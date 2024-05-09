import os

# Set the path to the directory containing the folders you want to rename
path_to_directory = "/home/yuechuan/vlr/project/neu-nbv/scripts/neural_rendering/data/dataset/urbanscene_cutscene_60"

# Fetch all the folder names in the directory
folder_names = [name for name in os.listdir(path_to_directory) if os.path.isdir(os.path.join(path_to_directory, name))]

# Sort the folders in your desired order
# This example sorts them alphabetically, but you can modify this part to suit your needs
# folder_names.sort()

# Loop through the folders and rename them
for i, folder_name in enumerate(folder_names):
    # Define the new name
    new_name = f"scan{i}"

    # Define the full old and new paths
    old_path = os.path.join(path_to_directory, folder_name)
    new_path = os.path.join(path_to_directory, new_name)

    # Rename the folder
    os.rename(old_path, new_path)
    print(f"Renamed '{folder_name}' to '{new_name}'")

print("All folders have been renamed.")
