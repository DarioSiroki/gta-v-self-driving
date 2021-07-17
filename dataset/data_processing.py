"""
This script is used for cleaning and balancing data
"""
import os
import numpy as np 

RAW_DATA_ROOT = "./raw_data"
CLEAN_DATA_ROOT = "./clean_data"

# normalize yAxis to be in range [-8, 8]
def normalize_yAxis(y):
    return round(y/(32768/8))

# normalize xAxis to be in range [-8, 8]
def normalize_xAxis(x):
    return round(x/(255/8))

# takes in an item [image, [steering_angle, brake, throttle]]
# returns [image, [yAxis, xAxis]]
def normalize_axes(item):
    steering_angle, brake, throttle = item[1]

    yAxis = normalize_yAxis(steering_angle)

    # if there was braking then output it as xAxis state,
    # otherwise output the throttle as xAxis state
    if brake != 0:
        xAxis = normalize_xAxis(brake)
    else:
        xAxis = normalize_xAxis(throttle)

    # modify the original item and return it
    item[1] = [yAxis, xAxis]
    return item

def save_data(lefts, rights, straights, iteration):
    # shuffle data first because the data will be cut to be balanced
    # this way if you cut 10k straights down to 5k, you get a variety of screens
    # instead of only the scenes shown in first 5k samples
    np.random.shuffle(lefts)
    np.random.shuffle(rights)
    np.random.shuffle(straights)

    # find the minimum length so lefts, rights and straights are of same size
    lengths = [len(lefts), len(rights), len(straights)]
    min_length_index = np.argmin(lengths)
    min_length = lengths[min_length_index]
    print(f"Iteration: {iteration}, lengths: {lengths}, min length: {min_length}")

    # size down so they are all same length
    lefts = lefts[:min_length]
    rights = rights[:min_length]
    straights = straights[:min_length]

    # normalize x and y axis
    lefts = [normalize_axes(item) for item in lefts]
    rights = [normalize_axes(item) for item in rights]
    straights = [normalize_axes(item) for item in straights]

    # convert to numpy arrays, concat, shuffle
    lefts = np.array(lefts)
    rights = np.array(rights)
    straights = np.array(rights)
    training_data = np.concatenate((lefts, rights, straights))
    np.random.shuffle(training_data)

    # save
    output_location = os.path.join(CLEAN_DATA_ROOT, f"clean_data-{iteration}.npy")
    np.save(output_location, training_data)


def main():
    raw_filenames = os.listdir(RAW_DATA_ROOT)

    if not(os.path.isdir(CLEAN_DATA_ROOT)):
        os.mkdir(CLEAN_DATA_ROOT)

    # accumulators
    lefts = []
    rights = []
    straights = []
    iteration = 0

    for i, raw_file in enumerate(raw_filenames):
        # save every 20 processed raw files (or if it's last iteration)
        # 20 files will cap around 4.5gb RAM usage
        if( i != 0 and i % 20 == 0) or (i == len(raw_filenames) - 1):
            save_data(lefts, rights, straights, iteration)
            iteration += 1
            lefts = []
            rights = []
            straights = []

        data = np.load(os.path.join(RAW_DATA_ROOT, raw_file), allow_pickle=True)
        
        for item in data:
            _, axis_state = item
            yAxis = axis_state[0]
            # controller sometimes yields left or right even though
            # there is no user input, so a range from -2000 to 2000 is 
            # considered a straight
            if yAxis < -2000:
                lefts.append(item)
            elif yAxis > 2000:
                rights.append(item)
            else:
                straights.append(item)

if __name__ == "__main__":
    main()