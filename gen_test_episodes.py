import torch
import numpy as np
import os

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from src.evaluation.datasets import samplers
from src.evaluation.datasets import dataloaders


transform = transforms.Compose([
    transforms.Resize(
        size=224,
        interpolation=InterpolationMode.BICUBIC,  # bicubic interpolation 사용
        antialias=None  # antialias 옵션 (필요시 조정)
    ),
    transforms.CenterCrop((224, 224)),  # (224, 224) 크기로 center crop
    transforms.ToTensor(),              # PIL 이미지를 tensor로 변환 (0~1 범위)
    transforms.Normalize(
        mean=[0.4815, 0.4578, 0.4082],   # 각 채널의 평균
        std=[0.2686, 0.2613, 0.2758]      # 각 채널의 표준편차
    )
])



save_folder = "/common_datasets/METAFLOW_DATASETS/test_episodes"
# os.makedirs(save_folder, exist_ok=True)

# datasets = ['aircraft', 'cifar_fs', 'chestX', 'paintings']
# for dataset_name in datasets:
#     data_path = f"/common_datasets/METAFLOW_DATASETS/caml_universal_eval_datasets/{dataset_name}/test"
#     # data_path = f"/common_datasets/METAFLOW_DATASETS/caml_train_datasets/fungi/train"
#     # data_path = f"/common_datasets/METAFLOW_DATASETS/caml_universal_eval_datasets"
    
#     dataset = dataloaders.get_dataset(data_path=data_path, is_training=False, transform_type=transform, pre=False)
#     sampler = samplers.random_sampler(
#         dataset,5,5,query_shot=16,trial=1000
#     )
#     episodes = []
#     for ids in sampler:
#         ids = torch.tensor(ids)
#         episodes.append(ids)
    
#     # print(len(episodes))
    
    
#     save_name = os.path.join(save_folder, f"{dataset_name}.pth")
#     print(save_name)
#     torch.save(episodes, save_name)


epis = torch.load(os.path.join(save_folder, 'aircraft.pth'))
print(len(epis))
print(epis[0])