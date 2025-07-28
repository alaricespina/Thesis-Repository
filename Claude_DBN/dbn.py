import numpy as np
import tensorflow as tf
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, n_epochs=100, batch_size=32):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Initialize weights and biases
        self.W = tf.Variable(tf.random.normal([n_visible, n_hidden], stddev=0.1))
        self.visible_bias = tf.Variable(tf.zeros([n_visible]))
        self.hidden_bias = tf.Variable(tf.zeros([n_hidden]))
    
    def sample_hidden(self, visible):
        hidden_prob = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.hidden_bias)
        hidden_sample = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob))))
        return hidden_prob, hidden_sample
    
    def sample_visible(self, hidden):
        visible_prob = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.visible_bias)
        visible_sample = tf.nn.relu(tf.sign(visible_prob - tf.random.uniform(tf.shape(visible_prob))))
        return visible_prob, visible_sample
    
    def contrastive_divergence(self, batch):
        print(f"  CD: Starting contrastive divergence with batch shape {batch.shape}")
        
        # Positive phase
        print(f"  CD: Positive phase - sampling hidden")
        h0_prob, h0_sample = self.sample_hidden(batch)
        print(f"  CD: Hidden probabilities shape: {h0_prob.shape}")
        
        # Negative phase
        print(f"  CD: Negative phase - sampling visible")
        v1_prob, v1_sample = self.sample_visible(h0_sample)
        print(f"  CD: Visible reconstruction shape: {v1_sample.shape}")
        
        print(f"  CD: Negative phase - sampling hidden again")
        h1_prob, h1_sample = self.sample_hidden(v1_sample)
        
        # Update weights and biases
        print(f"  CD: Computing gradients")
        positive_grad = tf.matmul(tf.transpose(batch), h0_prob)
        negative_grad = tf.matmul(tf.transpose(v1_sample), h1_prob)
        
        print(f"  CD: Updating weights and biases")
        self.W.assign_add(self.learning_rate * (positive_grad - negative_grad) / tf.cast(batch.shape[0], tf.float32))
        self.visible_bias.assign_add(self.learning_rate * tf.reduce_mean(batch - v1_sample, axis=0))
        self.hidden_bias.assign_add(self.learning_rate * tf.reduce_mean(h0_prob - h1_prob, axis=0))
        
        error = tf.reduce_mean(tf.square(batch - v1_sample))
        print(f"  CD: Computed error: {error}")
        return error
    
    def fit(self, X):
        X = tf.cast(X, tf.float32)
        n_samples = X.shape[0]  # Get actual Python int, not tensor
        print(f"RBM fit starting: n_samples={n_samples}, n_epochs={self.n_epochs}, batch_size={self.batch_size}")
        
        for epoch in range(self.n_epochs):
            print(f"Starting epoch {epoch}/{self.n_epochs}")
            epoch_error = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                print(f"Processing batch {n_batches+1}, samples {i}:{i+self.batch_size}")
                batch = X[i:i+self.batch_size]
                print(f"Batch shape: {batch.shape}")
                
                if batch.shape[0] > 0:  # Use .shape instead of tf.shape
                    print(f"Running contrastive divergence...")
                    batch_error = self.contrastive_divergence(batch)
                    print(f"Batch error: {batch_error}")
                    epoch_error += batch_error
                    n_batches += 1
            
            if n_batches > 0:
                avg_error = epoch_error / n_batches
                print(f"Epoch {epoch} completed, Error: {avg_error:.4f}, Batches: {n_batches}")
            else:
                print(f"Epoch {epoch}: No batches processed!")
    
    def transform(self, X):
        print(f"RBM transform: Input shape {X.shape}")
        X = tf.cast(X, tf.float32)
        print(f"RBM transform: Cast to tf.float32")
        
        # Use a simpler approach without the verbose sample_hidden
        hidden_prob = tf.nn.sigmoid(tf.matmul(X, self.W) + self.hidden_bias)
        print(f"RBM transform: Hidden prob computed, converting to numpy...")
        
        result = hidden_prob.numpy()
        print(f"RBM transform: Completed, output shape {result.shape}")
        return result


class DeepBeliefNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layers=[100, 50], learning_rate=0.1, n_epochs=100, batch_size=32):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rbms = []
    
    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float32)
        print(f"DBN fit starting with input shape: {X.shape}")
        
        # Normalize input data
        self.input_mean_ = np.mean(X, axis=0)
        self.input_std_ = np.std(X, axis=0) + 1e-8
        X = (X - self.input_mean_) / self.input_std_
        print(f"Data normalized, mean: {self.input_mean_.shape}, std: {self.input_std_.shape}")
        
        current_input = X
        
        # Train RBMs layer by layer
        for i, n_hidden in enumerate(self.hidden_layers):
            print(f"\n=== Training RBM layer {i+1} with {current_input.shape[1]} -> {n_hidden} units ===")
            
            rbm = RestrictedBoltzmannMachine(
                n_visible=current_input.shape[1],
                n_hidden=n_hidden,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size
            )
            print(f"RBM created for layer {i+1}")
            
            print(f"About to start RBM training for layer {i+1}...")
            rbm.fit(current_input)
            print(f"RBM layer {i+1} training completed!")
            
            self.rbms.append(rbm)
            
            # Transform data for next layer
            print(f"Transforming data for next layer...")
            current_input = rbm.transform(current_input)
            print(f"Layer {i+1} output shape: {current_input.shape}")
        
        print("DBN training completed!")
        return self
    
    def transform(self, X):
        X = np.array(X, dtype=np.float32)
        print(f"DBN transform: Input shape {X.shape}")
        
        # Normalize using training statistics
        X = (X - self.input_mean_) / self.input_std_
        print(f"DBN transform: Data normalized")
        
        # Forward pass through all RBM layers
        current_output = X
        for i, rbm in enumerate(self.rbms):
            print(f"DBN transform: Processing RBM layer {i+1}")
            current_output = rbm.transform(current_output)
            print(f"DBN transform: RBM layer {i+1} completed, shape: {current_output.shape}")
        
        print(f"DBN transform: All layers completed, final shape: {current_output.shape}")
        return current_output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)