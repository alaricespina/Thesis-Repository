# train_model.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer

# Load iris data
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Convert labels to one-hot
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Build a simple model
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate on training data
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"✅ Training Accuracy: {train_acc:.4f}")

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Testing Accuracy: {test_acc:.4f}")

# Save the model
model.save("iris_model.keras")
print("✅ Model saved as 'iris_model.keras'")
