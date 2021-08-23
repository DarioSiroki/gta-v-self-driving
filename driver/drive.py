from tensorflow.keras.models import load_model
from os import path
import numpy as np
import cv2
from PIL import ImageGrab
from fps_counter import FpsCounter
import pyxinput
import win32api as wapi
import time
from pymem import Pymem
import yolo
is_eights = False

MODEL_PATH = path.join("..", "train", "models", "plss.h5")
FPS_COUNTER = FpsCounter()
GAMEPAD = pyxinput.vController(True)
pm = Pymem('GTA5.exe')
yolo_model, classes, colors, output_layers = yolo.load_yolo()


def is_paused(paused):
    pause_key_pressed = wapi.GetAsyncKeyState(ord("U"))
    if pause_key_pressed:
        paused = not paused
        # Reset because it will keep pressing stuff.
        set_axes(0, 0)
        print(f"Paused: {paused}")

    # Avoid CPU fan noise (:
    if paused:
        time.sleep(0.1)

    return paused

def fix_deadzone(yAxis):
    if is_eights:
        return int(yAxis * 4095)
    # -0.2 to 0.2 won't even move the wheels, it's a deadzone
    if yAxis > -0.2 and yAxis < 0.2:
        if yAxis < 0:
            yAxis -= 0.2
        elif yAxis > 0:
            yAxis += 0.2
    return yAxis

def set_axes(yAxis, xAxis):
    yAxis = fix_deadzone(yAxis)
    GAMEPAD.set_value('AxisLx', yAxis)
    GAMEPAD.set_value('TriggerR', xAxis)


def drive(model):
    xyz_old = np.zeros(3)
    paused = False
    
    while True:
        paused = is_paused(paused)
        if paused: continue

        screen = np.array(ImageGrab.grab(bbox=(0, 26, 1920, 1106)))

        yolo_screen = cv2.resize(screen, (416, 416))
        yolo_screen = cv2.cvtColor(yolo_screen, cv2.COLOR_BGR2RGB)

        pilotnet_screen = cv2.resize(screen, (160, 120))
        pilotnet_screen = np.array([cv2.cvtColor(pilotnet_screen, cv2.COLOR_BGR2GRAY)])
        pilotnet_screen = pilotnet_screen.reshape(-1,160,120,1)
        pred = model.predict(pilotnet_screen, batch_size=1)[0]
        yAxis, xAxis = pred


        blob, outputs = yolo.detect_objects(yolo_screen, yolo_model, output_layers)
        boxes, confs, class_ids = yolo.get_box_dimensions(outputs, 416, 416)
        yolo.draw_labels(boxes, confs, colors, class_ids, classes, yolo_screen)

        if yolo.should_break(outputs, classes): xAxis = -1
        set_axes(yAxis, xAxis)
        FPS_COUNTER.increment()
        print(FPS_COUNTER.get_fps(), yAxis, xAxis)

        cv2.imshow("_", yolo_screen)

        # Get them through cheat engine
        # Offsets for GTA V 1.57: https://i.imgur.com/U16t3Zz.png
        x = pm.read_float(0x2A1C5A0C390)
        y = pm.read_float(0x2A1C5A0C394)
        z = pm.read_float(0x2A1C5A0C398)
        xyz_new = np.array([x, y, z])

        speed = np.sqrt(np.sum((xyz_new-xyz_old)**2, axis=0)) * 5 * 3.6
        xyz_old = xyz_new

        if speed < 40:
            GAMEPAD.set_value('TriggerR', 127)
        else:
            GAMEPAD.set_value('TriggerR', 0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def main():
    model = load_model(MODEL_PATH)
    drive(model)

if __name__ == "__main__":
    main()