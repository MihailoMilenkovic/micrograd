import os
import sys

from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

# Get the absolute path of the root directory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the Python module search path
sys.path.append(root_directory)

from micrograd.engine import Value

class Dataset:

    def __init__(self, mode: str, split_ratio: float = 0.8, k_fold: int = 10) -> None:
        self.mode = mode
        self.split_ratio = split_ratio
        self.k_fold = k_fold
        self.x, self.y = self.create_dataset()

    def get_whole_dataset(self):
        return self.x, self.y

    def create_dataset(self):
        # Load the digits dataset
        digits = load_digits()

        # Extract the features (X) and target labels (y)
        x = digits.data
        #normalize inputs to range [0,1]
        x=x/16 
        y = digits.target.reshape(-1, 1)  # Reshape to column vector
        # Convert the class labels to one-hot encodings

        print("x shape:", x.shape)
        print("y shape:", y.shape)

        # Shuffle the dataset
        x, y = shuffle(x, y, random_state=42)

        # Create an instance of OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)

        # Fit and transform the labels into one-hot encoded vectors
        y = encoder.fit_transform(y)

        train_data = []
        train_labels = []
        for sample in x:
            temp = []
            for val in sample:
                temp.append(Value(val))
            train_data.append(temp)
        train_labels = []
        for label in y:
            temp = []
            for val in label:
                temp.append(Value(val))
            train_labels.append(temp)
        return train_data, train_labels
    
    def get_train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=1-self.split_ratio , random_state=42)
        return x_train, y_train, x_test, y_test
    
    def get_k_fold_crossvalidation(self):
        sss = StratifiedShuffleSplit(n_splits=self.k_fold, test_size=1/self.k_fold, random_state=42)
        sss.get_n_splits(self.x, self.y)
        print(sss)
        splits = []
        for train_index, test_index in sss.split(self.x, self.y):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            splits.append((x_train, y_train, x_test, y_test))
        return splits