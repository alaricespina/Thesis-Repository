import tensorflow as tf 

from RBM import RBM 
import os 
import random 
import pandas as pd

os.system("cls")

# Load and preprocess data
data = pd.read_csv("timeseries.csv")

input_data = data["value"].to_numpy()

# Define window size and create batches
window_size = 32
batches = []
for i in range(0, len(data) - window_size + 1, window_size):
    batch = data.iloc[i:i+window_size]
    batches.append(batch["value"].to_numpy().astype('float32'))

shuffle = True


# Shuffle batches (optional)
if shuffle:
    random.shuffle(batches)

# Define number of RBMs and their sizes
num_rbms = 3
rbm_sizes = [100, 50, 20]
epochs = 3 

print(type(batches[0]))
print(batches[0])
print(batches[0].shape, batches[1].shape)
print(batches[0].dtype)

# Create RBM layers
rbms = []
for i in range(num_rbms):
    rbms.append(RBM(rbm_sizes[i-1] if i > 0 else window_size, rbm_sizes[i]))
    # rbms.append(RBM(rbm_sizes[i-1] if i > 0 else batches[0].shape[0], rbm_sizes[i]))

# Train each RBM layer sequentially
for i, rbm in enumerate(rbms):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for epoch in range(epochs):
        # Get training data batch
        v_batch = batches[i]  # Replace with your data loading logic
        with tf.GradientTape() as tape:
            cost, dW, dvb, dhb = rbm.contrastive_divergence(v_batch)
            gradients = tape.gradient([cost], [rbm.W, rbm.vb, rbm.hb])
            optimizer.apply_gradients(zip(gradients, [rbm.W, rbm.vb, rbm.hb]))

    # Monitor and log training progress

# Extract features from top RBM layer
features = rbms[-1].visible_to_hidden(input_data)
print(features)
