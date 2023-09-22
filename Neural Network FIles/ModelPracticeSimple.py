import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# DATA GENERATION

WIDTH = 5
HEIGHT = 500

sample_data = [[(i+j) for i in range(WIDTH)] for j in range(HEIGHT)]

names = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
column_names = [names[i] for i in range(WIDTH-1)] + ["TARGET"]
orig_data = pd.DataFrame(sample_data)

orig_data.columns = column_names

print(orig_data.head())

# DATA SPLITTING

dataset = orig_data.copy()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('TARGET')
test_labels = test_features.pop('TARGET')

# Tensorflow Normalizer

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Sequential Model Builder

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# Early stopping mechanism
early_stopping = keras.callbacks.EarlyStopping(
    patience = 5,
    min_delta = 0.001,
    restore_best_weights = True
)


# Training the model using the fit command

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_data = (test_features, test_labels),
    verbose=1, 
    batch_size=20,
    callbacks = [early_stopping],
    epochs=100)

# Show history of loss

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))

# Predict using trained model

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(range(len(test_labels)), test_labels, color="red")
plt.scatter(range(len(test_labels)), test_predictions, color="green")
plt.xlabel('Features Test')
plt.ylabel('Predictions and Labels')

plt.show()

# Print the test predictions

print(test_predictions)

print(test_labels)

# Save the model

# dnn_model.save('DNN_TEST_MODEL')


