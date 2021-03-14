from inputs import get_gamepad
from threading import Thread
from time import sleep


class Gamepad:
    def __init__(self):
        self.steering_angle = self.brake = self.throttle = 0
        self.pause_main_loop = True
        self.exit_main_loop = False
        self.run = True  # when this is set to False the thread running loop method will terminate
        Thread(target=self.loop).start()

    def loop(self):
        while self.run:
            sleep(0.001)
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X":
                    self.steering_angle = event.state
                elif event.code == "ABS_Z":
                    self.brake = event.state
                elif event.code == "ABS_RZ":
                    self.throttle = event.state
                elif event.code == "BTN_TR":
                    if event.state == 1:
                        self.pause_main_loop = not self.pause_main_loop
                elif event.code == "BTN_NORTH":
                    self.exit_main_loop = True

    def kill(self):
        self.run = False
