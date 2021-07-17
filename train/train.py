import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from os import path, listdir
from model import build_model
from utils.data_generator import DataGenerator
from utils.gpu_is_available import gpu_is_available

MODEL_ROOT = "models"
CLEAN_DATA_ROOT = path.join("..", "dataset", "clean_data")

def parse_file(path):
    data = np.load(path, allow_pickle=True)
    X_train, X_val, y_train, y_val = train_test_split(np.array([x[0] for x in data]), np.array(
        [np.array(x[1]) for x in data]), test_size=0.2, random_state=0)

    return X_train, X_val, y_train, y_val


def load_weights(model):
    model_path = path.join(MODEL_ROOT, "model.h5")
    if not path.isfile(model_path): 
        return model
    prev_model = load_model(model_path)
    model.set_weights(prev_model.get_weights())
    return model


def train_model(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=200):
    checkpoint = ModelCheckpoint(path.join(MODEL_ROOT, 'model.h5'),
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=1.0e-4))

    model.fit(x=DataGenerator(X_train, y_train, batch_size),
              validation_data=DataGenerator(X_val, y_val, batch_size),
              steps_per_epoch=len(X_train) // batch_size,
              validation_steps=len(X_val) // batch_size,
              epochs=epochs,
              max_queue_size=1,
              callbacks=[checkpoint])


def main():
    gpu_is_available()  # will throw if it's not
    files = [path.join(CLEAN_DATA_ROOT, f) for f in listdir(CLEAN_DATA_ROOT)]
    np.random.shuffle(files)

    model = build_model()
    model = load_weights(model)
    
    for i, file_path in enumerate(files):
        print(f"Processing training file {i+1}/{len(files)}")
        X_train, X_val, y_train, y_val = parse_file(file_path)
        train_model(model, X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
