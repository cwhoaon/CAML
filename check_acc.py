import numpy as np
import os


src_path = "outputs/ft_trajs_full_bias_train_20steps_5e-05lr_10seeds"
domain = 'mscoco'
file_name = 'acc_000009.npy'

path = os.path.join(src_path, domain, file_name)
acc = np.load(path)
print(acc)