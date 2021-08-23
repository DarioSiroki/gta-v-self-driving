import cv2
import numpy as np
from h5py import File

f = File("../clean_data/clean_data.h5", "r")
l = f["images"].shape[0]

backwards = False

print("training data length: ", len(f))
for i in range(0, len(f["images"]), 1):
    print(i)
    if backwards:
        i = l - i - 1
    im = f["images"][i]
    feedback = f["feedbacks"][i]
    cv2.imshow("_", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    print(feedback)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
