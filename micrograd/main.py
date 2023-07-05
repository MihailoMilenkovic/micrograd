import datetime
import sys
import os
import numpy as np
import dill
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

import config
from micrograd.engine import Value
from micrograd.nn import Linear, Sigmoid, BinaryCrossEntropyLoss, Sequential, Module, Softmax, CrossEntropyLoss
from micrograd.optimizers import SGD, Adam
from micrograd.metrics import Metrics
from micrograd.training import train, test
from micrograd.dataset import Dataset

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def create_mlp_model(input_size: int, hidden_size: List[int], output_size: int,
                      nonlin: bool = True, use_bias: bool = True) -> Sequential:
    layers = []
    layers.append(Linear(input_size, hidden_size[0], nonlin=nonlin, use_bias=use_bias))
    for i in range(1, len(hidden_size)):
        layers.append(Linear(hidden_size[i-1], hidden_size[i], nonlin=nonlin, use_bias=use_bias))
    layers.append(Linear(hidden_size[-1], output_size, nonlin=False, use_bias=use_bias))
    if output_size == 1:
        layers.append(Sigmoid())
    else:
        layers.append(Softmax())
    return Sequential(*layers)

def create_optimizer(name: str, lr: float):
    if name == "sgd":
        return SGD(lr)
    elif name == "adam":
        return Adam(lr)
    else:
        raise ValueError(f"Optimizer {name} not implemented")
    
def create_criteria(name: str):
    if name == "bce":
        return BinaryCrossEntropyLoss()
    elif name == "ce":
        return CrossEntropyLoss()
    else:
        raise ValueError(f"Criteria {name} not implemented")
    
def create_metrics(metrics: List[str], name: str = "train_metrics"):
    return Metrics(metrics, name)

def create_train_test_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the first 5 samples
    print("X train:", X_train[:5])
    print("X test:", X_test[:5])
    print("y_train (one-hot):", y_train[:5])
    print("y_test (one-hot):", y_test[:5])
    
    return X_train, X_test, y_train, y_test

# def create_dataset():

#     # Load the digits dataset
#     digits = load_digits()

#     # Extract the features (X) and target labels (y)
#     X = digits.data
#     #normalize inputs to range [0,1]
#     X=X/16 
#     y = digits.target.reshape(-1, 1)  # Reshape to column vector
#     # Convert the class labels to one-hot encodings

#     print("X shape:", X.shape)
#     print("y shape:", y.shape)

#     # Shuffle the dataset
#     X, y = shuffle(X, y, random_state=42)

#     # Create an instance of OneHotEncoder
#     encoder = OneHotEncoder(sparse_output=False)

#     # Fit and transform the labels into one-hot encoded vectors
#     y = encoder.fit_transform(y)

#     train_data = []
#     train_labels = []
#     for x in X:
#         temp = []
#         for val in x:
#             temp.append(Value(val))
#         train_data.append(temp)
#     train_labels = []
#     for label in y:
#         temp = []
#         for val in label:
#             temp.append(Value(val))
#         train_labels.append(temp)
#     return train_data, train_labels

def create_directories(experiment_name: str):
    # check if root_directory has an experiments directory
    if not os.path.exists(os.path.join(root_directory, "experiments")):
        os.mkdir(os.path.join(root_directory, "experiments"))
    # check if root_directory/experiments has an experiment_name directory
    if not os.path.exists(os.path.join(root_directory, "experiments", experiment_name)):
        os.mkdir(os.path.join(root_directory, "experiments", experiment_name))
    return os.path.join(root_directory, "experiments", experiment_name)


def run_traning() -> Sequential:
    model = None
    if config.load_model:
        with open(config.load_path, 'rb') as f:
            model = dill.load(f)
            print(model)
    else:
        model = create_mlp_model(config.input_size, config.hidden_size, config.output_size)
    optimizer = create_optimizer(config.optimizer, config.learning_rate)
    criterion = create_criteria(config.loss)
    train_metrics = create_metrics(config.train_metrics)
    test_metrics = create_metrics(config.test_metrics, name="test_metrics")
    experiment_name = config.experiment_name
    model_name = experiment_name + '-{date:%Y-%m-%d_%H:%M:%S}.txt'.format(date=datetime.datetime.now()) + ".pkl"
    save_path = create_directories(experiment_name)

    if config.training_mode == "split":
        x_train, y_train, x_test, y_test = Dataset(mode="split").get_train_test_split()
        return train(x_train, y_train, model, criterion, optimizer, train_metrics, epochs=config.epochs, 
                     save_path=save_path, model_name=model_name,
                     evaluate_on_epoch=config.evaluate_on_epoch, x_test=x_test, y_test=y_test, test_metrics=test_metrics)
    
    return None

def run_testing(model: Sequential = None):
    if model is None: 
        if config.load_model:
            with open(config.load_path, 'rb') as f:
                model = dill.load(f)
                print(model)
        else:
            raise ValueError("No model provided")
    criterion = create_criteria(config.loss)
    metrics = create_metrics(config.metrics)
    x, y = Dataset(mode="split").get_dataset()
    test(x, y, model, criterion, metrics)
        

if __name__ == "__main__":
    model = None
    if config.run_training:
        model = run_traning()   
    if config.run_testing:
        run_testing(model)