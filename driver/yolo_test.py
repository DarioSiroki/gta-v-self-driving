import torch
from PIL import ImageGrab
import numpy as np
import cv2
from fps_counter import FpsCounter
import yolo_classes

model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                       pretrained=True, force_reload=True)

fps_counter = FpsCounter()

while True:
    fps_counter.increment()
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))
    screen = cv2.resize(screen, (480, 270))

    results = model(screen)

    for r in results.xyxy[0]:
        r = r.detach().cpu().numpy()
        start_point = (r[0], r[1])
        end_point = (r[2], r[3])
        color = (255, 0, 0)
        cv2.rectangle(screen, start_point, end_point, color)

    cv2.putText(screen, fps_counter.get_fps(), (440, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow('_', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
