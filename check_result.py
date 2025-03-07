import os
import torch
import numpy as np


# [0, 10, 20, 30, 40, 50, 1, 5]
datasets = ['aircraft', 'chestX', 'paintings', 'cifar_fs']

vanilla_path = "/data4/CAML/results/huge/vanilla"
for i, dastaset in enumerate(datasets):
    file_path = os.path.join(vanilla_path, dastaset, "acc.npy")
    acc= np.load(file_path)
    print(acc.mean())


flow_huge_path = '/data4/CAML/results/huge/flow'
for i, dataset in enumerate(datasets):
    for ch in sorted(os.listdir(flow_huge_path))[1:]:
        file_path = os.path.join(flow_huge_path, ch, dataset, "acc.npy")
        acc = np.load(file_path)
        
        
        print(file_path)
        acc = acc.T
        for i in range(acc.shape[0]):
            print(acc[i].mean())        
        
        # print(acc.shape)
        # print(acc[:].mean())
        # print()