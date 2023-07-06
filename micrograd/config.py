experiment_name = "64-10-10-inf"

# running configuration
run_training = True
run_testing = False

# training mode [split, cross]
training_mode = "split"

# if traning mode is split
split_ratio = 0.8

# if training mode is cross
k_fold = 10

# wheater to evaluate on each epoch
evaluate_on_epoch = True

# model loading
load_model = False
load_path = "./experiments/mlp_classification/mlp_classification-2023-07-02_23:20:24.txt.pkl"

# model parameters
input_size = 64
hidden_size = [10,10]
output_size = 10

# loss function [bce, ce]
loss = "ce"

# training parameters
epochs = 10
learning_rate = 0.001

# choose optimizer from ["sgd", "adam"]
optimizer = "adam"

# metrics to track [loss, accuracy, mat]
train_metrics = ["loss", "accuracy"]
test_metrics = ["loss", "accuracy", "mat"]