import kagglehub
import os
import shutil

# Get the current working directory
cur_dir = os.getcwd()

# Download latest version
path = kagglehub.dataset_download("suriyaganesh/resume-dataset-structured")

# The directory that will hold the CSVs
data_directory = os.path.join(cur_dir, "data")

if os.path.isdir(data_directory):
    pass
else:
    os.makedirs(data_directory)

# Move the files into local directory
for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.move(os.path.join(path, file), os.path.join(data_directory, file))
