import numpy as np
from PIL import ImageGrab
import cv2
import time
from utils.gamepad import Gamepad
import os
import sys

# init gamepad
gp = Gamepad()

# fps counter stuff
last_time = time.time()
fps = 0

# open output file
if not os.path.isdir("data"):
    os.mkdir("data")

# training data will be appended here
training_data = []

print("Press RB to unpause")

while(True):
    if gp.pause_main_loop:
        continue
    # FPS counter
    time_passed = time.time() - last_time
    if time_passed >= 1:
        print("FPS", fps, end="\r", flush=True)
        fps = 0
        last_time = time.time()
    else:
        fps += 1

    # Grab image and resize it
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 1920, 1080)))
    screen = cv2.resize(screen, (480, 270))

    # Append screen and gamepad feedback to training data
    gp_feedback = [gp.steering_angle, gp.brake, gp.throttle]
    data = np.array([screen, gp_feedback], dtype="object")
    training_data.append(data)

    # Debug: show on screen
    cv2.imshow('_', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    # Save data every 500 iters
    if len(training_data) == 500:
        with open(f"data/{round(time.time())}.npy", 'ab') as output:
            print("saving data")
            np.save(output, training_data)
            training_data = []

    # Exit listener, quit with "q" key or xbox gamepad Y button
    if cv2.waitKey(25) & 0xFF == ord('q') or gp.exit_main_loop:
        gp.kill()
        cv2.destroyAllWindows()
        break
