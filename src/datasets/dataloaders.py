import torch
import os
import sys
import random

from torch.utils.data import Sampler
from typing import Any
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.datasets.episodic_imagenet_dataset import EpisodicImageNet
from src.datasets.cached_embedding_dataset import CachedEmbeddingDataset
from src.datasets.samplers import MetricSampler, custom_collate_batch_fn
from src.datasets.wikiart_dataset import WikiArt
from src.datasets.fungi_dataset import FungiDataset
from src.datasets.coco_dataset import CocoDataset

from torchvision.datasets import ImageFolder


# Full batch dataset for inf-shot finetuning
class CAMLDataset(ImageFolder):
  def __init__(self, dataset, split, transform):
    if split=='train' or split=='val':
      self.dataset_path = f"/common_datasets/METAFLOW_DATASETS/caml_train_datasets/{dataset}/{split}"
    elif split=='test':
      self.dataset_path = f"/common_datasets/METAFLOW_DATASETS/caml_universal_eval_datasets/{dataset}/{split}"
    super().__init__(self.dataset_path, transform=transform)
    self.split = split
    self.class_list = sorted(os.listdir(self.dataset_path))
    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.class_list))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
      
  def get_tgt2idx(self):
    return self.target_to_index
  


class FineTuneDataset(torch.utils.data.Dataset):
  def __init__(self, domain, split, transform, offset=0):
    self.domain = domain
    
    # embedding_cache_dir = f"/common_datasets/METAFLOW_DATASETS/caml_train_embeddings/{domain}/cached_embeddings/timm:vit_huge_patch14_clip_224.laion2b:1280"
    # task_path = f"/common_datasets/METAFLOW_DATASETS/task_descriptions/{domain}_embedding_{split}_500.pth"ß
    # self.dataset = CachedEmbeddingDataset(embedding_cache_dir, split=split)
    
    task_path = f"/common_datasets/METAFLOW_DATASETS/task_descriptions/{domain}_{split}_500.pth"
    self.dataset = CAMLDataset(domain, split, transform)
    self.split = split
    self.tasks = torch.load(task_path)
    self.n_task = len(self.tasks)
    self.target_to_index = self.dataset.get_tgt2idx()
    self.max_img_per_class = 500
    self.offset = offset
    
  def __len__(self):
    return self.n_task
  
  def __getitem__(self, idx):
    class_ids = self.tasks[idx+self.offset]
    images = []
    labels = []
    for i, class_id in enumerate(class_ids):
      images_per_class = []
      labels_per_class = []
      
      for image_id in self.sample_list(self.target_to_index[class_id.item()]):
        img, label = self.dataset[image_id]
        images_per_class.append(img)
        labels_per_class.append(i)
      images_per_class = torch.stack(images_per_class)
      labels_per_class = torch.tensor(labels_per_class).long()
      
      images.append(images_per_class)
      labels.append(labels_per_class)
    return images, labels
  
  def sample_list(self, my_list):
    if len(my_list) <= self.max_img_per_class:
        return my_list
    else:
        return random.sample(my_list, self.max_img_per_class)
      
      
  
class EpisodicDataset(torch.utils.data.Dataset):
  def __init__(self, domain, split, shot=5, query=10, transform=None, offset=0):
    self.domain = domain
    # embedding_cache_dir = f"/common_datasets/METAFLOW_DATASETS/caml_train_embeddings/{domain}/cached_embeddings/timm:vit_huge_patch14_clip_224.laion2b:1280"
    # task_path = f"/common_datasets/METAFLOW_DATASETS/task_descriptions/{domain}_embedding_{split}_500.pth"
    # self.dataset = CachedEmbeddingDataset(embedding_cache_dir, split=split)
      
    task_path = f"/common_datasets/METAFLOW_DATASETS/task_descriptions/{domain}_{split}_500.pth"
    self.dataset = CAMLDataset(domain, split, transform)
    self.split = split
    self.tasks = torch.load(task_path)
    self.n_task = len(self.tasks)
    self.target_to_index = self.dataset.get_tgt2idx()
    self.offset = offset
    self.shot = shot
    self.query = query
    
  def __len__(self):
    return self.n_task
  
  def __getitem__(self, idx):
    pass
  
  def sample_episode(self, idx, fix_seed=False):
    class_ids = self.tasks[idx+self.offset]
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    for i, class_id in enumerate(class_ids):
      support_images_per_class = []
      support_labels_per_class = []
      query_images_per_class = []
      query_labels_per_class = []
      
      img_id_list = self.sample_list(self.target_to_index[class_id.item()], self.shot+self.query) if not fix_seed else self.target_to_index[class_id.item()][:self.shot+self.query]
      if len(img_id_list) < self.shot:
        shot = len(img_id_list)-2
      else:
        shot = self.shot
      for j, image_id in enumerate(img_id_list):
        img, label = self.dataset[image_id]
        if j < shot:
          support_images_per_class.append(img)
          support_labels_per_class.append(i)
        else:
          query_images_per_class.append(img)
          query_labels_per_class.append(i)
      # print()
      # print(f"domain: {self.domain}, idx: {class_id}, total_len:{len(img_id_list)}, query_len: {len(query_images_per_class)}")
      support_images_per_class = torch.stack(support_images_per_class)
      support_labels_per_class = torch.tensor(support_labels_per_class).long()
      support_images.append(support_images_per_class)
      support_labels.append(support_labels_per_class)
      
      if len(query_images_per_class) != 0:
        query_images_per_class = torch.stack(query_images_per_class)
        query_labels_per_class = torch.tensor(query_labels_per_class).long()
        query_images.append(query_images_per_class)
        query_labels.append(query_labels_per_class)
      
    support_images = torch.cat(support_images)
    support_labels = torch.cat(support_labels)
    query_images = torch.cat(query_images)
    query_labels = torch.cat(query_labels)
    
    return support_images, support_labels, query_images, query_labels
  
  def sample_support(self, idx, fix_seed=False):
    class_ids = self.tasks[idx+self.offset]
    images = []
    masks = []
    for i, class_id in enumerate(class_ids):
      images_per_class = []
      masks_per_class = []
      
      img_id_list = self.sample_list(self.target_to_index[class_id.item()], self.shot) if not fix_seed else self.target_to_index[class_id.item()][:self.shot]
      for i in range(self.shot):
        if i < len(img_id_list):
          image_id = img_id_list[i]
          img, label = self.dataset[image_id]
          images_per_class.append(img)
          masks_per_class.append(1)
        else:
          blank_img = torch.zeros(img.shape)
          images_per_class.append(blank_img)
          masks_per_class.append(0)

      images_per_class = torch.stack(images_per_class)
      masks_per_class = torch.tensor(masks_per_class)
      
      images.append(images_per_class)
      masks.append(masks_per_class)
    
    images = torch.stack(images)
    masks = torch.stack(masks).bool()
    return images, masks
  
  
  def sample_list(self, my_list, k):
    if len(my_list) <= k:
        return my_list
    else:
        return random.sample(my_list, k)
  
  
class AggregatedDataset(torch.utils.data.Dataset):
  def __init__(self, domains, split, n_tasks_per_domain, fix_seed=False, transform=None):
    self.domains = domains
    self.split = split
    self.n_tasks_per_domain = n_tasks_per_domain
    self.fix_seed = fix_seed
    self.datasets = [EpisodicDataset(domain, split, transform=transform) for domain in self.domains]
    self.n_domains = len(domains)
    self.n_tasks = self.n_domains * self.n_tasks_per_domain
  
  def __len__(self):
    return self.n_tasks
  
  def __getitem__(self, idx):
    return self.sample_episode(idx)
  
  def sample_support(self, idx):
    domain_idx = idx // self.n_tasks_per_domain
    task_idx = idx % self.n_tasks_per_domain
    return self.datasets[domain_idx].sample_support(task_idx, self.fix_seed)
  
  def sample_episode(self, idx):
    domain_idx = idx // self.n_tasks_per_domain
    task_idx = idx % self.n_tasks_per_domain
    return self.datasets[domain_idx].sample_episode(task_idx, self.fix_seed)


class MetricDataloader(torch.utils.data.DataLoader):
  def __len__(self):
    return len(self.sampler)

def get_metric_dataloader(way,
                          shot,
                          batch_size,
                          transform,
                          split,
                          dataset: str = "imagenet",
                          dataset_kwargs={},
                          **kwargs):
  """Dataset is one of 'imagenet', 'wikiart'."""
  if (batch_size - way * shot) % way != 0:
    raise Exception(
      f'Batch size does not evenly divide into way*shot samples: ' +
      f'{(batch_size - way * shot) % way} remainder -- this needs to be 0.'
    )
  if kwargs.get('use_embedding_cache', False):
    embedding_cache_dir = kwargs.get('embedding_cache_dir')
    print('Using embeddings cached at', embedding_cache_dir)
    data = CachedEmbeddingDataset(embedding_cache_dir,
                                  split=split,
                                  **dataset_kwargs)
  else:
    # Otherwise, we load the data normally
    if dataset == "imagenet":
      data = EpisodicImageNet('../image_datasets/latest_imagenet', split=split, transform=transform)
    elif dataset == "wikiart-style":
      data = WikiArt(split=split, class_column="style", transform=transform)
    elif dataset == "wikiart-genre":
      data = WikiArt(split=split, class_column="genre", transform=transform)
    elif dataset == "wikiart-artist":
      data = WikiArt(split=split, class_column="artist", transform=transform)
    elif dataset == 'fungi':
      data = FungiDataset(split=split, transform=transform)
    elif dataset == 'coco':
      data = CocoDataset(split=split, transform=transform)
  num_workers = 50
  episodic_sampler = MetricSampler(len(data.classes),
                                   data.target_to_index,
                                   way=way,
                                   shot=shot,
                                   batch_size=batch_size)
  data_loader = MetricDataloader(data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 sampler=episodic_sampler,
                                 collate_fn=custom_collate_batch_fn)
  return data_loader


if __name__ == '__main__':
  dataset = "fungi"
  task_path = "/common_datasets/METAFLOW_DATASETS/task_descriptions/fungi_train_500.pth"
  dataset = FineTuneDataset(dataset, None, task_path, "val")
  print(dataset[4176])
  # print(dataset.__dict__)
  for k, v in dataset.__dict__.items():
    print(k)
  
  # print(dataset.target_to_index)
