import torch
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
# import ipdb

from src.datasets.fungi_dataset import FungiDataset
from src.datasets.coco_dataset import CocoDataset
from src.datasets.episodic_imagenet_dataset import EpisodicImageNet

from src.models.feature_extractors.pretrained_fe import get_fe_metadata

# image_path = '/common_datasets/METAFLOW_DATASETS/caml_train_datasets/fungi/val/1/1200575.jpg'
embedding_path = '/common_datasets/METAFLOW_DATASETS/caml_train_embeddings/fungi/cached_embeddings/timm:vit_huge_patch14_clip_224.laion2b:1280/val'


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

# timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k
class args:
    model = 'CAML'
    fe_type = 'timm:vit_huge_patch14_clip_224.laion2b:1280'
    fe_dtype = ''
    
args = args()

fe_metadata = get_fe_metadata(args)
feature_extractor = fe_metadata['fe'].eval()

print(fe_metadata['train_transform'])
print(fe_metadata['test_transform'])
image1, _ = FungiDataset(split='train', transform=transform)[1]
image2, _ = FungiDataset(split='train', transform=fe_metadata['test_transform'])[2]
# image = image.to(torch.float16)
img_embedding = feature_extractor(image1.unsqueeze(0)).squeeze(0)
img_embedding2 = feature_extractor(image2.unsqueeze(0)).squeeze(0)

# print(img_embedding.shape)
# print(img_embedding)

# dist_img = np.abs(img_embedding-img_embedding2).mean()
# print(dist_img)

embedding_folders = os.listdir(embedding_path)

mean_embedding = torch.from_numpy(np.load(os.path.join(embedding_path, 'split_average.npy')))
print(mean_embedding.shape)

selected_embedding = None
for folder_name in embedding_folders:
    if folder_name == 'split_average.npy':
        continue
    folder_path = os.path.join(embedding_path, folder_name)
    class_files = os.listdir(folder_path)
    for embedding_file in class_files:
        embedding = torch.from_numpy(np.load(os.path.join(folder_path, embedding_file)))
        #  ipdb.set_trace()
        
        # embedding = F.normalize(
        #   embedding - mean_embedding, p=2, dim=0
        # )
        
        dist = (torch.abs(img_embedding - embedding)).mean()
        
        # print(dist)
        if dist < 0.1:
            selected_embedding = embedding
            print("same")
            print(dist)
        
        # if np.allclose(img_embedding, embedding, rtol=1e-05, atol=1e-08, equal_nan=False):
        #     print('same')
        
print(img_embedding)
if selected_embedding is not None:
    print(selected_embedding)