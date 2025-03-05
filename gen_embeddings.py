import torch
import numpy
import os

root = 'common_datasets/METAFLOW_DATASETS'

src_path = os.path.join(root, 'caml_universal_eval_datasets')
tgt_path = os.path.join(root, 'caml_universal_eval_embeddings')

domains = ['aircraft', 'chestX', 'cifar_fs', 'paintings']

class args:
    model = 'CAML'
    fe_type = 'timm:vit_huge_patch14_clip_224.laion2b:1280'
    fe_dtype = ''
    
args = args()

fe_metadata = get_fe_metadata(args)
feature_extractor = fe_metadata['fe'].eval()

fe_metadata['train_transform']
fe_metadata['test_transform']

class TestDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__() 
        self.target_to_index = {
        class_idx: []
        for class_idx in range(len(self.classes))
        }
        for sample_idx, target in enumerate(self.targets):
            self.target_to_index[target].append(sample_idx)
        

for domain in domains:
    root = os.path.join(src_path, domain, 'test')
    dataset = TestDataset(self, root, )

  def __init__(self, split: str = "train", transform=None):
    assert split in ['train', 'val']
    path_to_datasets = '/common_datasets/METAFLOW_DATASETS/caml_train_datasets/'
    super().__init__(f'{path_to_datasets}mscoco/{split}', transform=transform)
    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
    self.all_targets = list(self.target_to_index.keys())
    self.all_sample_ids = list(self.target_to_index.values())[0]

