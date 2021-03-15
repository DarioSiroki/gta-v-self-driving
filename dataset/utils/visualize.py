import cv2
import numpy as np

f = np.load("../data/1615768087.npy", allow_pickle=True)

print("training data length: ", len(f))
for item in f:
    im = item[0]
    feedback = item[1]
    cv2.imshow("_", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    print(feedback)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
