import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras import layers


def input_func(x): return x % 10

raw_input = [i for i in range(5000)]
raw_output = [input_func(i) for i in raw_input]

#plt.plot(raw_input, raw_output)
#plt.show()

df = pd.DataFrame({
    "Input" : raw_input,
    "Output" : raw_output
})

dataset = df.copy()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("Output")
test_labels = test_features.pop("Output")

print(test_dataset.nunique())

model = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_shape=(1, )),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)

loss = keras.losses.SparseCategoricalCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)
history = model.fit(train_features, 
                    train_labels, 
                    batch_size=10, 
                    epochs=10, 
                    verbose=1, 
                    validation_split=0.2,
                    validation_data=(test_features, test_labels))

print(history)
