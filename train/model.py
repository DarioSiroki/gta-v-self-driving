from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential

def build_model():
  model = Sequential()
  model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(270, 480, 3)))
  model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
  model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
  model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation='elu'))
  model.add(Conv2D(64, (3, 3), activation='elu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dense(1))
  #model.summary()

  return model