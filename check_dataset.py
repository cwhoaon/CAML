import os
import sys



path = "/common_datasets/METAFLOW_DATASETS/caml_train_datasets/fungi/train"

child = os.listdir(path)

li = []
for c in child:
    child_path = os.path.join(path, c)
    x = os.listdir(child_path)
    li.append(len(x))
print(li)

out = list(filter(lambda x: x<5, li))
print(len(out))