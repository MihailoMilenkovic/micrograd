import os
os.system("mkdir data_for_vis")
for optim in ["adam","sgd"]:
    for architecture in ["64_10","64-10-10-10","64-10-10"]:
        for metric in ["train_loss","train_acc","test_loss","test_acc"]:
            os.system(f"touch ./data_for_vis/{architecture}-{optim}-{metric}.txt")

for metric in ["train_loss","train_acc","test_loss","test_acc"]:
    os.system(f"touch ./data_for_vis/64-10-10-sgd-{metric}-history.txt")