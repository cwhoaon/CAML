import os
import torch
import numpy as np


# [0, 10, 20, 30, 40, 50, 1, 5]

path = "/data3/CAML/results/metaflow_30steps_0.15lr_5seeds_0.2noise/chestX/flow_acc.npy"
accs = np.load(path)
for i in range(accs.shape[1]):
    print(accs[:,i].mean())