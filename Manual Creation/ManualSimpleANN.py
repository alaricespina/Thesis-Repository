import numpy as np 

# 1. Define independent variables and dependent variable
input_set = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0],
                      [1,1,0],
                      [1,1,1],
                      [0,1,1],
                      [0,1,0]])

labels = np.array([[1, 0, 0, 1, 1, 0, 1]])
labels = labels.reshape(7,1)

# 2. Define Hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

# 3. Define Activation Function and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# 4. Train the model
for epoch in range(25000):
    # assign inputs from the input set, always reassign
    inputs = input_set

    # Dot Product of randomly generated weights and inputs
    XW = np.dot(inputs, weights) + bias

    # Pass the Dot Produce of Weights and Bias to the Sigmoid Function
    z = sigmoid(XW)

    # Get the error between the real and label
    error = z - labels
    # print(error.sum())

    dcost = error
    
    dpred = sigmoid_derivative(z)

    z_del = dcost * dpred
    inputs = input_set.T
    weights = weights - lr * np.dot(inputs, z_del)
    
    for num in z_del:
        bias = bias - lr*num

# 5. Make predictions
single_pt = np.array([0,1,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)