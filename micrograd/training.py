
import sys
import os
import numpy as np
import dill
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value
from micrograd.nn import Linear, Sigmoid, BinaryCrossEntropyLoss, Sequential, Module, Softmax, CrossEntropyLoss
from micrograd.optimizers import SGD
from micrograd.metrics import Metrics

def save_model(model: Module, path: str):
    with open(path, "wb") as f:
        dill.dump(model, f)
        print(f"Model saved to {path}")

def load_model(path: str) -> Module:
    with open(path, "rb") as f:
        model = dill.load(f)
        print(f"Model loaded from {path}")
        return model

def train(x: List[List[Value]], y: List[Value], model: Sequential, criterion: Module, optimizer,
           metrics: Metrics, epochs: int = 10, save_path: str = None, model_name: str = "model.pkl",
           evaluate_on_epoch: bool = False, x_test: List[List[Value]] = None, y_test: List[Value] = None,
           test_metrics: Metrics = None, fold_num: int = None) -> Sequential:
    
    print(f"Number of training parameters: {len(model.parameters())}")

    for epoch in range(epochs):
        for i, (x_, y_) in enumerate(zip(x, y)):
            pred = model(x_)
            loss = criterion(pred, y_)
            metrics.record(loss, pred, y_, epoch, i)
            # backward
            loss.backward()
            optimizer.step(model.parameters())
            # zero gradients
            model.zero_grad()
            loss.destroy_graph(model.parameters())

        metrics.report(epoch, epochs)
        if evaluate_on_epoch:
            assert x_test is not None and y_test is not None and test_metrics is not None
            test(x_test, y_test, model, criterion, test_metrics)
    save_model(model, os.path.join(save_path, model_name))
    metrics.save_log_file(os.path.join(save_path, f"metrics_{fold_num}.txt"))
    test_metrics.save_log_file(os.path.join(save_path, f"test_metrics_{fold_num}.txt"))
    test_metrics.save_history(os.path.join(save_path, f"test_history_{fold_num}.txt"))


def test(x: List[List[Value]], y: List[Value], model: Sequential, criterion: Module, metrics: Metrics):
    for i, (x_, y_) in enumerate(zip(x, y)):
        pred = model(x_)
        loss = criterion(pred, y_)
        metrics.record(loss, pred, y_, 0, i, x=x_)
        loss.destroy_graph(model.parameters())
    # make sure to reset any remaining gradients
    model.zero_grad()
    metrics.report(0, 1)


