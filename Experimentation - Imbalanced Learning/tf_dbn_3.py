import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# 1. RBM Class
class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, momentum=0.95, cd_steps=1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cd_steps = cd_steps

        # Initialize weights and biases
        self.W = tf.Variable(tf.random.normal([n_visible, n_hidden], 0.01), name="weights") # Random initialization
        self.bv = tf.Variable(tf.zeros([n_visible]), name="visible_bias")
        self.bh = tf.Variable(tf.zeros([n_hidden]), name="hidden_bias")

        # Create variables for momentum
        self.W_delta = tf.Variable(tf.zeros([n_visible, n_hidden]), name="weight_delta")
        self.bv_delta = tf.Variable(tf.zeros([n_visible]), name="visible_delta")
        self.bh_delta = tf.Variable(tf.zeros([n_hidden]), name="hidden_delta")

    # Gibbs Sampling functions
    def _sample_hidden(self, v):
        p_h_given_v = tf.nn.sigmoid(tf.matmul(v, self.W) + self.bh)
        return tf.nn.relu(tf.sign(p_h_given_v - tf.random.uniform(tf.shape(p_h_given_v))))

    def _sample_visible(self, h):
        p_v_given_h = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.bv)
        return tf.nn.relu(tf.sign(p_v_given_h - tf.random.uniform(tf.shape(p_v_given_h))))

    # Contrastive Divergence training step
    def train(self, X):
        v_k = tf.identity(X) # Initial state
        for step in range(self.cd_steps):
            h_k = self._sample_hidden(v_k) # Positive phase
            v_k = self._sample_visible(h_k) # Reconstructed visible units

        # Calculate probabilities for positive and negative phases
        h = self._sample_hidden(X)
        h_k = self._sample_hidden(v_k)
        # Contrastive Divergence gradient update (using TensorFlow)
        positive_grad = tf.matmul(tf.transpose(X), h)
        negative_grad = tf.matmul(tf.transpose(v_k), h_k)

        # Calculate deltas with momentum
        W_delta = self.momentum * self.W_delta + self.learning_rate * (positive_grad - negative_grad) / tf.cast(tf.shape(X)[0], tf.float32)
        bv_delta = self.momentum * self.bv_delta + self.learning_rate * tf.reduce_mean(X - v_k, 0)
        bh_delta = self.momentum * self.bh_delta + self.learning_rate * tf.reduce_mean(h - h_k, 0)

        # Apply updates (using TensorFlow)
        self.W.assign_add(W_delta)
        self.bv.assign_add(bv_delta)
        self.bh.assign_add(bh_delta)

        # Update deltas for the next iteration
        self.W_delta.assign(W_delta)
        self.bv_delta.assign(bv_delta)
        self.bh_delta.assign(bh_delta)

    def transform(self, X):
        # Transforms input X through the RBM layer
        p_h_given_v = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
        return p_h_given_v # Probability output

# 2. DBN Class (using TensorFlow for Neural Network Layers)
class DBN(tf.keras.Model):
    def __init__(self, n_visible, hidden_layer_sizes, learning_rate=0.01, momentum=0.95, cd_steps=1):
        super(DBN, self).__init__()
        self.n_visible = n_visible
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cd_steps = cd_steps

        self.rbm_layers = []
        self.fine_tuning_layers = []

        # Create RBM layers
        for i, n_hidden in enumerate(hidden_layer_sizes):

            rbm = RBM(n_visible, n_hidden, learning_rate, momentum, cd_steps) #Removed nIn equation and just put it at visible
            self.rbm_layers.append(rbm)

            # Create Dense layer for fine-tuning (adjust activation as needed)
            dense_layer = tf.keras.layers.Dense(n_hidden, activation='relu', kernel_initializer='glorot_uniform') # Changed for RELU.

            self.fine_tuning_layers.append(dense_layer) # Append here, so DBN can train later
            # Append to layers will be done in the build layers

    def pretrain(self, X, epochs=10):
        print (" Pre Training Steps")
        for i, rbm in enumerate(self.rbm_layers):
            print(f"Training RBM layer {i+1}/{len(self.rbm_layers)}")
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                for batch in self._iterate_minibatches(X, batchsize=128, shuffle=True):
                    rbm.train(batch)

    def call(self, inputs):
        # Use the DBN model for inference
        layer_input = inputs
        for layer in self.fine_tuning_layers:
            layer_input = layer(layer_input)
        return layer_input

    def _iterate_minibatches(self, inputs, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield tf.convert_to_tensor(inputs[excerpt], dtype=tf.float32)

# Example usage
def tf_dbn_example():
    # 1. Generate dataset (as before)
    X, y = make_classification(n_samples=5000, n_features=60,
                               n_informative=45, n_redundant=15,
                               n_classes=4,
                               random_state=42, weights=[0.4, 0.4, 0.1, 0.1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = BorderlineSMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # DBN parameters
    n_visible = X_train_resampled.shape[1] # Number of input features
    hidden_layer_sizes = [40, 20, 10]  # ADD n_visible as THE FIRST VALUE
    learning_rate = 0.01
    momentum = 0.95
    cd_steps = 1 # Contrastive Divergence steps
    pretrain_epochs = 5

    # 2. Create and pre-train the DBN
    dbn = DBN(n_visible, hidden_layer_sizes, learning_rate, momentum, cd_steps) # Tensorflow RBM
    dbn.pretrain(X_train_resampled, epochs=pretrain_epochs)

    # 3. Transform the data using the trained DBN
    X_train_transformed = dbn(tf.convert_to_tensor(X_train_resampled, dtype=tf.float32)).numpy() # Output of the DBN is feed in RF
    X_test_transformed = dbn(tf.convert_to_tensor(X_test, dtype=tf.float32)).numpy()

    # 4. Neural Network Classifier (Keras)
    from tensorflow import keras
    from tensorflow.keras import layers

    # Define the model
    model = keras.Sequential([
        layers.Dense(64, kernel_initializer='he_normal', activation='elu', input_shape=(X_train_transformed.shape[1],), kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),  # Batch Norm, improves convergence
        layers.Dropout(0.3),  # Adjust Dropout

        layers.Dense(32, kernel_initializer='he_normal', activation='elu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(), # Batch Norm, improves convergence
        layers.Dropout(0.3),  # Adjust Dropout

        layers.Dense(4, activation='softmax') # 4 classes, softmax output
    ])

    # 3. Model compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Check learning rate
                 loss='sparse_categorical_crossentropy', # Loss based on number of outputs
                 metrics=['accuracy'])

    # 4. Implement Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                      restore_best_weights=True)

    # 5. Model Training
    model.fit(X_train_transformed, y_train_resampled, epochs=100, batch_size=32, validation_split=0.1,
             callbacks=[early_stopping])

    # 5. Evaluate the model
    y_pred = np.argmax(model.predict(X_test_transformed), axis=1)  # Convert probabilities to class labels

    # 6. Display the report, confusion matrix, plot for the DBN + NN models
    print("\nTensorFlow DBN with Neural Network")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - TensorFlow DBN + NN")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Running the tensorflow example
tf_dbn_example()