from .meta_h5_dataset import FullMetaDatasetH5
from .meta_dataset.args import get_args_parser
from torch.utils.data import DataLoader
from .meta_dataset.utils import Split
import utils.deit_util as deit_util
import torch

def get_dataloaders(datasets, split, dataloader_fn, batch_size, transforms, use_embedding_cache, fe_subdir):
  rtn = []
  image_dataset_path = '../caml_train_datasets'
  for (dataset, way, shot) in datasets:
    if dataset == 'imagenet':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="imagenet",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/latest_imagenet/cached_embeddings/{fe_subdir}'))
    elif dataset == 'fungi':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="fungi",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/fungi/cached_embeddings/{fe_subdir}'))
    elif dataset == 'coco':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="coco",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/mscoco/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-style':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-style",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_style/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-genre':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-genre",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_genre/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-artist':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-artist",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_artist/cached_embeddings/{fe_subdir}'))
    elif dataset == 'metadata':
      def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
      parser = get_args_parser()
      shell_script = \
        f"--data-path meta_dataset_h5 --dataset meta_dataset --bf16 --num_ways 5 --ignore_dag_ontology --ignore_bilevel_ontology --image_size 224"
      dataset_args = parser.parse_args(shell_script.split())
      
      if split=='train':
        split = Split.TRAIN
      elif split=='val':
        split = Split.VALID
        
      dataset = FullMetaDatasetH5(dataset_args, split)
      
      num_tasks = deit_util.get_world_size()
      global_rank = deit_util.get_rank()
      
      dist_sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
      )
      dataloader = DataLoader(dataset, sampler=dist_sampler, batch_size=1, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
      
      return dataloader
      
  return rtn
