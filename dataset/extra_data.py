"""
Data can be downloaded here:
https://drive.google.com/drive/folders/1R787vkWaMe5nsWyLpbXTG55aUv4YteTo

Extract the data (npy files) to /dataset/alzaib and run this script directly 
to append his data to clean data that was produced by data_processing.py, or 
just use this data.

Credits to https://github.com/Alzaib/Autonomous-Self-Driving-Car-GTA-5 for the data.
"""

import os 
import numpy as np
from h5py import File

output = File("clean_data/clean_data.h5", "a")

files = os.listdir("alzaib")

def append_data(data):
    output["images"].resize((output["images"].shape[0] + data.shape[0]), axis = 0)
    output["feedbacks"].resize((output["feedbacks"].shape[0] + data.shape[0]), axis = 0)
    output["images"][-data.shape[0]:] = [x[0] for x in data]
    output["feedbacks"][-data.shape[0]:] = [x[1] for x in data]

for file in files:
    data = np.load("alzaib/" + file, allow_pickle=True)
    print(file)
    append_data(data)