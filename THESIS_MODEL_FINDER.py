# Lahat ng mga kailangang module

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import activations

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Input Data, Baguhin kung kailangan gawin yung Baguio, Makati or Manila
RAW_DATA_CSV = pd.read_csv("QC_2016_01_01_TO_2018_08_31.csv")

# Dataset to be used
dataset = RAW_DATA_CSV.copy()

# Features ng AI (Will Change)
features_temp = ["tempmin", "tempmax", "temp"]
features_fl = ["feelslikemax", "feelslikemin", "feelslike"]
features_dew = ["dew", "humidity"]
features_wind = ["windspeed"]
features_cloud = ["cloudcover", "visibility"]

# All Features combined into 1 array
features_all = features_temp + features_fl + features_dew + features_wind + features_cloud
features_list = [features_temp,features_fl,features_dew,features_wind,features_cloud]

# Target (Y)
target = ["conditions"]

# Get the stated features from the dataset
x_raw_data = dataset[features_all]
y_raw_data = dataset[target]

# idk why i need to do this
y_raw_data = y_raw_data.conditions.to_list()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x_raw_data, y_raw_data, test_size=0.2, random_state=0)

# Encode the data into numpy arrays
le = LabelEncoder()
le.fit(y_train)
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)
x_train_encoded = x_train.to_numpy()
x_test_encoded = x_test.to_numpy()

MIN_POSSIBLE_LAYERS = 1
MAX_POSSIBLE_LAYERS = 10
MIN_POSSIBLE_UNITS = 128
MAX_POSSIBLE_UNITS = 1024+1
STEP_POSSIBLE_UNITS = 128
OUTPUT_UNITS = 4
EPOCHS = 100
ACTIVATION_ARRAY = [activations.relu, activations.sigmoid, activations.softmax,
                    activations.softplus, activations.softsign, activations.tanh,
                    activations.elu, activations.selu, activations.exponential]

OPTIMIZER_ARRAY = [keras.optimizers.SGD, keras.optimizers.RMSprop, keras.optimizers.Adam,
                   keras.optimizers.Adadelta, keras.optimizers.Adamax,
                   keras.optimizers.Adagrad, keras.optimizers.Nadam,
                   keras.optimizers.Ftrl]

LOSS_ARRAY = [keras.losses.BinaryCrossentropy(from_logits=True),
              keras.losses.CategoricalCrossentropy(from_logits=True),
              keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              keras.losses.Poisson(from_logits=True),
              keras.losses.KLDivergence(from_logits=True)
]

LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1, 1]

RESULTS = []

for num_layers in range(MIN_POSSIBLE_LAYERS, MAX_POSSIBLE_LAYERS):
    for num_units in range(MIN_POSSIBLE_UNITS, MAX_POSSIBLE_UNITS, STEP_POSSIBLE_UNITS):
        for activation in ACTIVATION_ARRAY:
            for optimizer in OPTIMIZER_ARRAY:
                for learning_rate in LEARNING_RATES:
                    for loss in LOSS_ARRAY:
                        #print(f"Parameters ======> L: {num_layers} | U: {num_units} | A: {activation} | O: {optimizer(learning_rate = learning_rate)} | LR: {learning_rate}")  
                        
                        ann_model = Sequential()

                        # Flatten Layer
                        ann_model.add(Flatten(input_shape=[11]))
                        
                        for i in range(num_layers):
                            ann_model.add(Dense(num_units, activation=activation))

                        # Output Layer
                        ann_model.add(Dense(4))

                        ann_model.compile(
                            loss = loss,
                            optimizer = optimizer(learning_rate=learning_rate),
                            metrics=["accuracy"]
                        )

                        epochs = 100

                        history = ann_model.fit(x_train_encoded, y_train_encoded, 
                            #batch_size=batch_size, 
                            epochs=epochs, 
                            validation_data = (x_test_encoded, y_test_encoded),
                            shuffle=True, verbose=0)
                        
                        history_df = pd.DataFrame(history.history)

                        characteristic = [num_layers, 
                                        num_units, 
                                        str(activation).replace("<function ","").split()[0], 
                                        str(optimizer).replace("<class 'keras.optimizers.optimizer_experimental", "").replace("'>","").split(".")[2], 
                                        learning_rate, 
                                        loss.name,
                                        history_df['accuracy'].max(),
                                        history_df['val_accuracy'].max()] 
                                        

                        RESULTS.append(characteristic)                    

RESULTS_DF = pd.DataFrame(RESULTS, columns=["Layers", "Units", "Activation", "Optimizer", "Learning Rate", "Loss Function", "Accuracy", "Val Accuracy"])
RESULTS_DF.to_csv("BlankResults.csv")
