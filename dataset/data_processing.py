import os
import numpy as np 
from h5py import File
import cv2 

RAW_DATA_ROOT = "./raw_data"
CLEAN_DATA_ROOT = "./clean_data"

output = File(os.path.join(CLEAN_DATA_ROOT, "clean_data.h5"), 'w')

output.create_dataset('images', shape=(0, 120, 160), dtype='uint8', chunks=True, maxshape=(None, 120, 160))
output.create_dataset('feedbacks', shape=(0, 2), dtype='float64', chunks=True, maxshape=(None, 2))


def normalize_axes(feedback):
    """
    Takes in feedback [steering_angle, brake, throttle].

    Returns [yAxis, xAxis].

    yAxis is steering angle and xAxis is brake/throttle. 

    Axes are normalized in range between -1 and 1.
    """
    steering_angle, brake, throttle = feedback

    yAxis = steering_angle / 32767

    # If there was braking then output it as xAxis state,
    # otherwise output the throttle as xAxis state
    if brake != 0:
        xAxis = -brake / 255
    else:
        xAxis = throttle / 255

    feedback = np.array([yAxis, xAxis])

    return feedback

def process_image(image):
    image = cv2.resize(image, (160,120))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def save_data(data):
    output["images"].resize((output["images"].shape[0] + data.shape[0]), axis = 0)
    output["feedbacks"].resize((output["feedbacks"].shape[0] + data.shape[0]), axis = 0)
    output["images"][-data.shape[0]:] = [x[0] for x in data]
    output["feedbacks"][-data.shape[0]:] = [x[1] for x in data]

def process_data(data):
    for i, item in enumerate(data):
        image, feedback = item
        feedback = normalize_axes(feedback)
        image = process_image(image)
        data[i] = [image, feedback]
    return data

def main():
    raw_filenames = os.listdir(RAW_DATA_ROOT)

    for i, raw_file in enumerate(raw_filenames):
        print(f"Processing file {i}/{len(raw_filenames)}")
        data = np.load(os.path.join(RAW_DATA_ROOT, raw_file), allow_pickle=True)
        data = process_data(data)
        save_data(data)

if __name__ == "__main__":
    main()