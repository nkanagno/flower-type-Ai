import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import save_model
import numpy as np

import pandas as pd

# Label columns
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'] # Flower stats (width/height)


# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# train 70%
# test  30%
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# pop 'Species'
train_y = train.pop('Species')
test_y = test.pop('Species')

# Build your Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for classification
])


# Compile your model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train your model
model.fit(train, train_y,validation_split=0.1, batch_size=256,epochs=200, shuffle=True,verbose=2)





# Save the model
save_model(model, 'my_model.h5')  # Save in HDF5 format