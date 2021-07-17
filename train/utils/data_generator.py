from tensorflow.keras.utils import Sequence
import math 
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, X, y, batch_size):
        self.x, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]

        inputs = np.array(batch_x)
        targets = np.array(batch_y)
        return inputs, targets
