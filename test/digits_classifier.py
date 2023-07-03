import sys
import os
import numpy as np
from typing import List,Optional
from pathlib import Path
import json
     
# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

import dill
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

from micrograd.engine import Value
from micrograd.nn import Linear, Sigmoid, BinaryCrossEntropyLoss, Sequential, Module, Softmax, CrossEntropyLoss
from micrograd.optimizers import SGD, Adam
from micrograd.metrics import Metrics

def preprocess_dataset():
    # Load the digits dataset
    digits = load_digits()

    # Extract the features (X) and target labels (y)
    X = digits.data
    #normalize inputs to range [0,1]
    X=X/16 
    y = digits.target.reshape(-1, 1)  # Reshape to column vector
    # Convert the class labels to one-hot encodings
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y).toarray()

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)
    return X,y

def create_train_test_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Print the first 5 samples
    print("X train:", X_train[:5])
    print("X test:", X_test[:5])
    print("y_train (one-hot):", y_train[:5])
    print("y_test (one-hot):", y_test[:5])
    

    return X_train, X_test, y_train, y_test


def convert_to_micrograd(X,y):
    # Convert the dataset to micrograd values
    train_data = []
    train_labels = []
    for x in X:
        temp = []
        for val in x:
            temp.append(Value(val))
        train_data.append(temp)
    X = train_data
    for label in y:
        temp = []
        for val in label:
            temp.append(Value(val))
        train_labels.append(temp)
    y = train_labels
    return X,y

def save_model(model: Module, path: str,train_metrics:Optional[List[str]]=None,test_metrics:Optional[List[str]]=None):
    with open(path, "wb") as f:
        dill.dump(model, f)
        print(f"Model saved to {path}")
    if train_metrics and test_metrics:
        with open(f"{path}_train_metrics.json", "w") as file:
            json.dump(train_metrics, file)
        with open(f"{path}_test_metrics.json", "w") as file:
            json.dump(test_metrics, file)

def load_model(path: str) -> Module:
    with open(path, "rb") as f:
        model = dill.load(f)
        print(f"Model loaded from {path}")
        return model

def train_model(x_train: List[List[Value]], x_test: List[List[Value]],
                y_train: List[List[Value]], y_test: List[List[Value]],
                model_type="mlp", use_bias=True, load: bool = False, 
                 epochs: int = 10, learning_rate: float = 0.05):
    
    model_path=os.path.join(root_directory,f"models/digit_classification/{model_type}_model.pkl")
    if not load:
        in_neurons=len(x_train[0])
        out_neurons=len(y_train[0])
        hidden_dim=10
        if model_type=="mlp":
            model = Sequential(Linear(in_neurons, hidden_dim, nonlin=True ,use_bias=use_bias),
                           Linear(hidden_dim, out_neurons, nonlin=False ,use_bias=use_bias),
                           Softmax())
        elif model_type=="cnn":
            #TODO: create conv2d and pooling layers
            model=Sequential(
                    Linear(in_neurons, hidden_dim, nonlin=True ,use_bias=use_bias),
                    Linear(hidden_dim, out_neurons, nonlin=False ,use_bias=use_bias),
                    Softmax())
            exit(1)
        
    else:
        model = load_model(model_path)
    
    print(f"Number of training parameters: {len(model.parameters())}")
        
    criterion = CrossEntropyLoss()
    # optimizer = SGD(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    train_metrics = Metrics(["loss", "accuracy"], name="train_metrics")
    test_metrics = Metrics(["loss", "accuracy"], name="test_metrics")
    train_metric_out=[]
    train_metric_data=[]
    test_metric_out=[]
    test_metric_data=[]
    for epoch in range(epochs):
        for i, (x_, y_) in enumerate(zip(x_train, y_train)):
            pred = model(x_)
            loss = criterion(pred, y_)
            train_metrics.record(loss, pred, y_, epoch, i)

            # backward
            loss.backward()
            optimizer.step(model.parameters())

            # zero gradients
            model.zero_grad()
            loss.destroy_graph(model.parameters())
        metric_out,metric_data=train_metrics.report(epoch, epochs)
        train_metric_out.append(metric_out)
        train_metric_data.append(metric_data)
        #record metrics on test set
        for i, (x_, y_) in enumerate(zip(x_test, y_test)):
            pred = model(x_)
            loss = criterion(pred, y_)
            test_metrics.record(loss, pred, y_, epoch, i)

            # zero gradients
            model.zero_grad()
            loss.destroy_graph(model.parameters())
        
        metric_out,metric_data=test_metrics.report(epoch, epochs)
        test_metric_out.append(metric_out)
        test_metric_data.append(metric_data)

        save_model(model,f"{model_path}_epoch_{epoch}")

    save_model(model, model_path,train_metrics=train_metric_out,test_metrics=test_metric_out)
    return train_metric_data,test_metric_data


if __name__=="__main__":
    k_fold=True
    X,y=preprocess_dataset()
    print("X:",X,"y:",y)
    if not k_fold:
        X_train, X_test, y_train, y_test=create_train_test_dataset(X,y)
        X_train,y_train=convert_to_micrograd(X_train,y_train)
        X_test,y_test=convert_to_micrograd(X_test,y_test)
        train_model(X_train,X_test,y_train,y_test,model_type="mlp",epochs=10,load=False)
    else:
        num_folds=2
        sss=StratifiedShuffleSplit(n_splits=num_folds)
        all_metrics={}
        all_metrics["train_loss"]=[]
        all_metrics["train_accuracy"]=[]
        all_metrics["test_loss"]=[]
        all_metrics["test_accuracy"]=[]
        for train,test in sss.split(X,y):
            X_train,y_train=X[train],y[train]
            X_test,y_test=X[test],y[test]
            X_train,y_train=convert_to_micrograd(X_train,y_train)
            X_test,y_test=convert_to_micrograd(X_test,y_test)
            num_epochs=1
            train_metric_data,test_metric_data=train_model(X_train,X_test,y_train,y_test,model_type="mlp",epochs=num_epochs,load=False)
            metric_data={"train":train_metric_data,"test":test_metric_data}
            for dataset_type in ["train","test"]:
                curr_metric_data=metric_data[dataset_type]
                for metric in ["loss","accuracy"]:
                    all_metrics[f"{dataset_type}_{metric}"].append(curr_metric_data[num_epochs-1][metric])
        print("Results after k-fold cross-validation:")
        for metric in all_metrics:
            avg_value=sum(all_metrics[metric])/len(all_metrics[metric])
            print(f"Average {metric}:{avg_value}")
