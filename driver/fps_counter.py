import time 

class FpsCounter:
    def __init__(self):
        self._last_loop = time.time()
        self._fps_counter = 0
        self._current_fps = 0

    def increment(self):
        if time.time() - self._last_loop >= 1:
            self._last_loop = time.time()
            self._current_fps = self._fps_counter
            self._fps_counter = 0
        else:
            self._fps_counter += 1

    def get_fps(self):
        return str(self._current_fps)
