import tensorflow as tf 

class RBM:
    def __init__(self, visible_units, hidden_units):
        # Initialize weight matrix and biases
        self.W = tf.Variable(tf.random.normal([visible_units, hidden_units]))
        self.vb = tf.Variable(tf.zeros([visible_units]))
        self.hb = tf.Variable(tf.zeros([hidden_units]))

    def visible_to_hidden(self, v):
        # Calculate hidden activations
        h = tf.matmul(v, self.W) + self.hb
        h = tf.sigmoid(h)
        return h

    def hidden_to_visible(self, h):
        # Calculate visible activations
        v = tf.matmul(h, tf.transpose(self.W)) + self.vb
        v = tf.sigmoid(v)
        return v

    def contrastive_divergence(self, v0):
        # Positive phase: visible -> hidden -> visible
        h0 = self.visible_to_hidden(v0)
        v1 = self.hidden_to_visible(h0)
        h1 = self.visible_to_hidden(v1)

        # Negative phase: sample hidden -> sample visible -> sample hidden
        hk = tf.random.uniform(tf.shape(h0))
        vk = self.hidden_to_visible(hk)
        hk = self.visible_to_hidden(vk)

        # Calculate positive and negative phase reconstructions
        positive_reconstruction = tf.reduce_mean(v0 * v0 - v1 * v1)
        negative_reconstruction = tf.reduce_mean(vk * vk - hk * hk)

        # Cost function and gradients
        cost = positive_reconstruction - negative_reconstruction
        dW = tf.matmul(tf.transpose(v0), h0) - tf.matmul(tf.transpose(v1), h1)
        dvb = tf.reduce_mean(v0 - v1)
        dhb = tf.reduce_mean(h0 - h1)

        return cost, dW, dvb, dhb
