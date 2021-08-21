from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import random

processed_data = File("clean_data/clean_data.h5", "r")
balanced_data = File("clean_data/balanced_data.h5", "w")

balanced_data.create_dataset('images', shape=(0, 120, 160), dtype='uint8', chunks=True, maxshape=(None, 120, 160))
balanced_data.create_dataset('feedbacks', shape=(0, 2), dtype='float64', chunks=True, maxshape=(None, 2))

def save_data(data):
    balanced_data["images"].resize((balanced_data["images"].shape[0] + data.shape[0]), axis = 0)
    balanced_data["feedbacks"].resize((balanced_data["feedbacks"].shape[0] + data.shape[0]), axis = 0)
    balanced_data["images"][-data.shape[0]:] = [x[0] for x in data]
    balanced_data["feedbacks"][-data.shape[0]:] = [x[1] for x in data]

yAxes = [f[0] for f in processed_data["feedbacks"]]

# Create 26 intervals between -1 and 1.
interval_edges = np.linspace(np.min(yAxes), np.max(yAxes), 26)

# Dict where:
# Key will be the category, example: "<interval-start> to <interval-end>"
# Value will be an array which contains indexes of items
# which belong to the interval.
#
# Example:
#
# {
#   "-1 to -0.9": [34, 643, 29, ...]
#   ...
# }
#
categories = {}
# Create empty arrays for each range
for j in range(len(interval_edges)-1):
    categories[f"{interval_edges[j]} to {interval_edges[j+1]}"] = []

# Loop through the data and check to which category does
# each data item
for i in range(processed_data["feedbacks"].shape[0]):
    yAxis = processed_data["feedbacks"][i][0]
    for j in range(len(interval_edges)-1):
        if yAxis > interval_edges[j] and yAxis < interval_edges[j+1]:
            categories[f"{interval_edges[j]} to {interval_edges[j+1]}"].append(i)

for key in categories:
    random.shuffle(categories[key])
    print(f"Processing interval {key}")
    train_data = []
    for i in categories[key][:3000]:
        im = processed_data["images"][i]
        feedback = processed_data["feedbacks"][i]
        train_data.append([im, feedback])
    save_data(np.array(train_data))
