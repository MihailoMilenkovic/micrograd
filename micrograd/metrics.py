import sys
import os
from typing import List

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value

class Metrics:

    def __init__(self, metrics: List[str], num_labels: int, name="train_metrics"):
        # check that metrics list consist of only valid metrics [loss, accuracy, precision, recall, f1]
        for metric in metrics:
            assert metric in ["loss", "accuracy", "mat"], f"Invalid metric {metric}"

        self.name=name
        self.metrics = metrics
        self.loss_history = {}
        self.output_history = {}
        self.input_history = {}
        self.num_labels = num_labels
        self.log = ""

    def record(self, loss: float, preds: List[Value], y: List[Value], epoch: int, iteration: int, x: List[Value] = None):
        if iteration == 0:
            self.loss_history[f"Epoch {epoch + 1}"] = []
            self.output_history[f"Epoch {epoch + 1}"] = []
            self.input_history[f"Epoch {epoch + 1}"] = []

        self.loss_history[f"Epoch {epoch + 1}"].append(loss.data)
        predicted_probs = [p.data for p in preds]
        predicted_class = max(range(len(predicted_probs)), key=lambda i: predicted_probs[i])
        labels_class = [l.data for l in y]
        labels_class = labels_class.index(1)
        self.output_history[f"Epoch {epoch + 1}"].append([predicted_class, labels_class])

        if x is not None:
            self.input_history[f"Epoch {epoch + 1}"].append(self.convert_micrograd_values_to_list(x))

    def report(self, epoch: int, total_epochs: int):
        output = f"{self.name} - epoch: {epoch + 1}/{total_epochs}"
        for metric in self.metrics:
            if metric != "mat":
                output += f" | {metric.capitalize()}: {self.calculate_metric_by_epoch(metric, epoch + 1):.4f}"
        output += "\n"
        output += "-" * len(output)
        output += "\n"
        if "mat" in self.metrics:
            # print matrix in a nice way
            for row in self.calculate_metric_by_epoch("mat", epoch + 1):
                row_string = ""
                for num in row:
                    row_string += f"{num:4}"
                output += row_string + "\n"

        print(output)
        self.log += output + "\n"
        return output
        
    def convert_micrograd_values_to_list(self, micrograd_values: List[Value]) -> List[float]:
        return [v.data for v in micrograd_values]

    def calculate_metric_by_epoch(self, metric: str, epoch: int) -> float:
        if metric == "loss":
            return self.calculate_mean_loss_by_epoch(epoch)
        elif metric == "accuracy":
            return self.calculate_multiclass_accuracy_by_epoch(epoch)
        elif metric == "mat":
            return self.calculate_multiclass_confusion_matrix_by_epoch(epoch)
        return None

    def calculate_mean_loss_by_epoch(self, epoch: int) -> float:
        key = f"Epoch {epoch}"
        losses = self.loss_history[key]
        return sum(losses) / len(losses)
    
    def calculate_multiclass_accuracy_by_epoch(self, epoch: int) -> float:
        key = f"Epoch {epoch}"
        results = self.output_history[key]
        correct = 0
        for result in results:
            pred = result[0]
            y = result[1]
            if pred == y:
                correct += 1
        return correct / len(results)
    
    def calculate_multiclass_confusion_matrix_by_epoch(self, epoch: int) -> List[List[float]]:
        key = f"Epoch {epoch}"
        results = self.output_history[key]
        confusion_matrix = [[0 for _ in range(self.num_labels)] for _ in range(self.num_labels)]
        for result in results:
            pred = result[0]
            y = result[1]
            confusion_matrix[pred][y] += 1
        return confusion_matrix
    
    def save_log_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.log)

    def save_history(self, path: str):
        with open(path, "w") as f:
            f.write(str(self.output_history) + "\n")
            f.write(str(self.input_history) + "\n")
    
    
    