import numpy as np
from PIL import ImageGrab
import cv2
import time
from utils.gamepad import Gamepad
import os

# Initialize gamepad
gp = Gamepad()

# Create output dir if it doesn't exist
if not os.path.isdir("data"):
    os.mkdir("data")

# Training data will be appended here
training_data = []

print("Press RB to pause/unpause. Currently paused.")

while(True):
    if gp.pause_main_loop:
        time.sleep(0.1)
        continue
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
            print(f"{time.strftime('%H:%M:%S')} Saving data")
            np.save(output, training_data)
            training_data = []

    # Exit listener, quit with "q" key or xbox gamepad Y button
    if cv2.waitKey(25) & 0xFF == ord('q') or gp.exit_main_loop:
        gp.kill()
        cv2.destroyAllWindows()
        break
