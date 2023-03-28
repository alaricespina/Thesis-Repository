# Thesis Algorithms
The algorithms contained would first specify the Main Algorithm and then specify the specific algorithms for the different parts of the main algorithm which are the following submodules:
1. `initialize_DBN`
2. `select_training_batch`
3. `get_expected_output`
4. `train_DBN`
5. `fine_tune_DBN`
6. `get_input`
7. `classify_weather`

## Main Algorithm

Main Algorithm for the creation of the DBN and the prediction of weather using the trained DBN. First the DBN is created and initially trainned using contrastive divergence and later on fine tuned through the use of backpropagation

```
# Step 1: Define the DBN architecture and parameters
input_nodes = 4  # Number of input nodes
hidden_nodes = [16, 16]  # Number of hidden nodes in each layer
output_nodes = 3  # Number of output nodes (Sunny, Cloudy, Rainy)
learning_rate = 0.01  # Learning rate for training
epochs = 50  # Number of training iterations
batch_size = 32  # Number of training examples in each batch
dropout_rate = 0.2  # Dropout rate to prevent overfitting

# Step 2: Initialize the DBN and train the model
initialize_DBN(input_nodes, hidden_nodes, output_nodes)
for i in range(epochs):
    # Select a random batch of training examples
    batch_x = select_training_batch(wind_speed, cloud_coverage, relative_humidity, temperature, batch_size)
    batch_y = get_expected_output(batch_x)  # Get the expected output for the batch
    # Train the DBN using contrastive divergence
    # train_DBN(batch_x, batch_y, learning_rate, dropout_rate)
    train_DBN(batch_x, learning_rate, dropout_rate)

# Step 3: Fine-tune the DBN using backpropagation
fine_tune_DBN(wind_speed, cloud_coverage, relative_humidity, temperature, learning_rate, epochs)

# Step 4: Use the trained DBN to classify the weather
x = get_input(wind_speed, cloud_coverage, relative_humidity, temperature)
classification = classify_weather(x)

```

## Initialize DBN

This module would simply define the different layers of the DBN Model which consists of the different Restricted Boltzmann Machines (RBM)

```
def initialize_DBN(input_nodes, hidden_nodes, output_nodes):
    # Initialize the DBN
    global dbn
    dbn = []

    # Add the input layer to the DBN
    input_layer = RBM(input_nodes, hidden_nodes[0])
    dbn.append(input_layer)

    # Add the hidden layers to the DBN
    for i in range(1, len(hidden_nodes)):
        hidden_layer = RBM(hidden_nodes[i-1], hidden_nodes[i])
        dbn.append(hidden_layer)

    # Add the output layer to the DBN
    output_layer = Sigmoid(hidden_nodes[-1], output_nodes)
    dbn.append(output_layer)
    
    # Initialize the weights and biases of the RBMs in the DBN
    for layer in dbn:
        layer.initialize_weights_and_biases()

```

## Select training batch

This part of the program would just get a sample of the training data

```
def select_training_batch(wind_speed, cloud_coverage, relative_humidity, temperature, batch_size):
    # Randomly select a batch of training examples
    num_examples = len(wind_speed)
    indices = random.sample(range(num_examples), batch_size)
    batch_x = []
    for i in indices:
        # Create a tuple containing the input values for this example
        example_x = (wind_speed[i], cloud_coverage[i], relative_humidity[i], temperature[i])
        batch_x.append(example_x)
    return batch_x

```

## Get expected output and Get Input
This part of the program would just get the necessary values for the expected output which is located in the dataset and the get input which would just get the inputs from the sensors

```
def get_input(wind_speed, cloud_coverage, relative_humidity, temperature):
    input_vector = [wind_speed, cloud_coverage, relative_humidity, temperature]
    return input_vector

```

```
def get_expected_output(batch_x):
    # Initialize an empty list to hold the expected output for each input in the batch
    batch_y = []
    for input_data in batch_x:
        # Determine the expected output for the current input data based on some predefined rule or dataset
        expected_output = determine_expected_output(input_data)
        batch_y.append(expected_output)
    return batch_y

```

## Train DBN

This part of the code would specify the pseudocode for the training of the Deep Belief Network by first training and then fine tuning
```
def train_DBN(batch_x, learning_rate, dropout_rate):
    # Initialize the visible layer with the input batch
    v = batch_x

    # Forward pass through each layer
    for i in range(num_layers):
        # Compute the probabilities of the hidden layer
        h = sample_prob(sigmoid(np.dot(v, W[i]) + b[i]))
        # Apply dropout to the hidden layer
        mask = np.random.binomial(1, 1-dropout_rate, size=h.shape)
        h *= mask / (1-dropout_rate)
        # Compute the reconstructed probabilities of the visible layer
        v = sample_prob(sigmoid(np.dot(h, W[i].T) + c[i]))

    # Compute the positive and negative associations between the input batch and the final hidden layer
    pos_associations = np.dot(batch_x.T, sample_prob(sigmoid(np.dot(batch_x, W[0]) + b[0])))
    neg_associations = np.dot(v.T, sample_prob(sigmoid(np.dot(v, W[0]) + b[0])))

    # Update the weights and biases based on the contrastive divergence objective
    dW = (pos_associations - neg_associations) / batch_x.shape[0]
    dc = np.mean(batch_x - v, axis=0)
    db = np.mean(sample_prob(sigmoid(np.dot(batch_x, W[0]) + b[0])) - sample_prob(sigmoid(np.dot(v, W[0]) + b[0])), axis=0)
    W[0] += learning_rate * dW
    c[0] += learning_rate * dc
    b[0] += learning_rate * db
```

## Fine Tune DBN
This part of the code would fine tune the Deep Belief Network through propagation
```
def fine_tune_DBN(wind_speed, cloud_coverage, relative_humidity, temperature, learning_rate, epochs)
    # Concatenate the input variables
    X = concatenate_inputs(wind_speed, cloud_coverage, relative_humidity, temperature)

    # Get the expected output
    y = get_expected_output(X)

    # Use backpropagation to fine-tune the DBN
    for i in range(epochs):
        # Perform feedforward to get the output of each layer
        layer_output = feedforward(X)

        # Compute the error at the output layer
        output_error = layer_output[-1] - y

        # Compute the error at each hidden layer
        hidden_errors = backpropagate(output_error, layer_output)

        # Update the weights and biases using the errors and the learning rate
        for j in range(len(dbn)):
            dbn[j].update_weights(hidden_errors[j], layer_output[j], learning_rate)

```

## Classify the Weather
This would contain the pseudocode for classifying the weather using the inputs combined into a single vector and make use of the trained DBN.
```
def classify_weather(x)
    # Pass the input through the DBN to get the predicted output
    input_layer = x
    for layer in dbn:
        input_layer = layer.feedforward(input_layer)
    predicted_output = input_layer
    
    # Determine the class with the highest probability as the predicted weather
    max_prob_index = argmax(predicted_output)
    if max_prob_index == 0:
        predicted_weather = "Sunny"
    elif max_prob_index == 1:
        predicted_weather = "Cloudy"
    else:
        predicted_weather = "Rainy"
        
    return predicted_weather

```