import os
import random
import h5py
from PIL import Image
import json
#import cv2
import numpy as np

import torch
from .meta_dataset import config as config_lib
from .meta_dataset import sampling
from .meta_dataset.utils import Split
from .meta_dataset.transform import get_transforms
from .meta_dataset.task_transform import get_consistent_transform
from .meta_dataset import dataset_spec as dataset_spec_lib


class FullMetaDatasetH5(torch.utils.data.Dataset):
    def __init__(self, args, split=Split['TRAIN'], eval_mode=False, fix_seed=False, use_cache=False):
        super().__init__()

        # Data & episodic configurations
        data_config = config_lib.DataConfig(args)
        episod_config = config_lib.EpisodeDescriptionConfig(args)

        if split == Split.TRAIN:
            datasets = args.base_sources
            episod_config.num_episodes = args.nEpisode
        elif split == Split.VALID:
            datasets = args.val_sources
            episod_config.num_episodes = args.nValEpisode
        else:
            datasets = args.test_sources
            episod_config.num_episodes = 600

        if args.evaluate_train or args.evaluate_valid:
            self.dataset = args.base_sources[0]
        else:
            self.dataset = None

        use_dag_ontology_list = [False]*len(datasets)
        use_bilevel_ontology_list = [False]*len(datasets)
        # if episod_config.num_ways:
        #     if len(datasets) > 1 and not (args.tasks_path is not None and args.n_tasks > 0):
        #         raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
        # else:
        #     # Enable ontology aware sampling for Omniglot and ImageNet.
        #     if 'omniglot' in datasets:
        #         use_bilevel_ontology_list[datasets.index('omniglot')] = True
        #     if 'ilsvrc_2012' in datasets:
        #         use_dag_ontology_list[datasets.index('ilsvrc_2012')] = True

        episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
        episod_config.use_dag_ontology_list = use_dag_ontology_list

        if args.num_query is not None and args.num_support is not None:
            episod_config.min_examples_in_class = args.num_query + args.num_support
        elif args.num_support is not None:
            episod_config.min_examples_in_class = args.num_support + 1

        # dataset specifications
        all_dataset_specs = []
        for dataset_name in datasets:
            dataset_records_path = os.path.join(data_config.path, dataset_name)
            dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
            all_dataset_specs.append(dataset_spec)

        num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])
        print(f"=> There are {num_classes} classes in the {split} split of the combined datasets")

        self.datasets = datasets
        if eval_mode:
            self.transforms = get_transforms(data_config, Split['TEST'])
        else:
            self.transforms = get_transforms(data_config, split)

        self.class_map = {} # 2-level dict of h5 paths
        self.class_h5_dict = {} # 2-level dict of opened h5 files
        self.class_samplers = {} # 1-level dict of samplers, one for each dataset
        self.class_images = {} # 2-level dict of image ids, one list for each class

        for i, dataset_name in enumerate(datasets):
            dataset_spec = all_dataset_specs[i]
            base_path = dataset_spec.path
            class_set = dataset_spec.get_classes(split) # class ids in this split
            num_classes = len(class_set)

            record_file_pattern = dataset_spec.file_pattern
            assert record_file_pattern.startswith('{}'), f'Unsupported {record_file_pattern}.'

            self.class_map[dataset_name] = {}
            self.class_h5_dict[dataset_name] = {}
            self.class_images[dataset_name] = {}

            for class_id in class_set:
                data_path = os.path.join(base_path, record_file_pattern.format(class_id))
                self.class_map[dataset_name][class_id] = data_path.replace('tfrecords', 'h5')
                self.class_h5_dict[dataset_name][class_id] = None # closed h5 is None
                self.class_images[dataset_name][class_id] = [str(j) for j in range(dataset_spec.get_total_images_per_class(class_id))]

            self.class_samplers[dataset_name] = sampling.EpisodeDescriptionSampler(
                dataset_spec=dataset_spec,
                split=split,
                episode_descr_config=episod_config,
                use_dag_hierarchy=episod_config.use_dag_ontology_list[i],
                use_bilevel_hierarchy=episod_config.use_bilevel_ontology_list[i],
                ignore_hierarchy_probability=args.ignore_hierarchy_probability,
                n_tasks=args.n_tasks,
                tasks_path=(os.path.join(args.tasks_path, f'{dataset_name}.pth') if args.tasks_path else None),)
            
        if args.n_tasks > 0:
            self.n_tasks_per_dataset = {dataset: len(self.class_samplers[dataset].class_ids) for dataset in datasets}
            if self.dataset is not None:
                self.n_tasks = self.n_tasks_per_dataset[self.dataset]
            else:
                self.n_tasks = sum([self.n_tasks_per_dataset[dataset_name] for dataset_name in datasets])
        else:
            self.n_tasks = 0
        self.eval_mode = eval_mode
        self.fix_seed = fix_seed

        if self.eval_mode and self.n_tasks > 0:
            self.len = self.n_tasks
        else:
            self.len = episod_config.num_episodes * len(datasets) # NOTE: not all datasets get equal number of episodes per epoch

        if args.fp16:
            self.dtype = torch.float16
        elif args.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float

        self.apply_data_transform = args.apply_data_transform
        if self.apply_data_transform:
            self.data_transforms = get_transforms(data_config, Split['TRAIN'])
        else:
            self.data_transforms = None
        self.apply_class_transform = args.apply_class_transform
        self.apply_task_transform = args.apply_task_transform
        self.data_config = data_config
        self.use_cache = use_cache

    def __len__(self):
        return self.len

    def get_next(self, source, class_id, idx, transforms=None, cache=False):
        cache = self.use_cache or cache

        # fetch h5 path
        h5_path = self.class_map[source][class_id]

        # load h5 file if None
        if cache:
            if self.class_h5_dict[source][class_id] is None: # will be closed in the end of main.py
                self.class_h5_dict[source][class_id] = h5py.File(h5_path, 'r')

            h5_file = self.class_h5_dict[source][class_id]
            record = h5_file[idx]
            x = record['image'][()]
        else:
            with h5py.File(h5_path, 'r') as h5_file:
                x = h5_file[str(idx)]['image'][()]

        transforms = transforms or self.transforms

        if transforms:
            x = Image.fromarray(x)
            # x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.
            x = transforms(x).to(dtype=self.dtype)

        return x
    
    def _load_images(self, source, class_id, max_samples):
        h5_path = self.class_map[source][class_id]
        with h5py.File(h5_path, 'r') as h5_file:
            images = []
            for idx in self.class_images[source][class_id][:max_samples]:
                x = h5_file[str(idx)]['image'][()]
                x = Image.fromarray(x)
                x = self.transforms(x).to(dtype=self.dtype)
                images.append(x)
            images = torch.stack(images, dim=0)
        return images
    
    def __getitem__(self, idx):
        return self.sample_episode(idx)

    def _global_to_local_task_index(self, global_task_idx):
        dataset = None
        task_idx = global_task_idx
        for source in self.datasets:
            if task_idx < self.n_tasks_per_dataset[source]:
                dataset = source
                break

            task_idx -= self.n_tasks_per_dataset[source]
        assert dataset is not None
        
        return dataset, task_idx

    def _local_to_global_task_index(self, dataset, task_idx):
        assert dataset in self.datasets
        global_task_idx = 0
        for _dataset in self.datasets:
            if dataset == _dataset:
                global_task_idx += task_idx
                break

            global_task_idx += self.n_tasks_per_dataset[_dataset]

        return global_task_idx
    
    def sample_support(self, global_task_idx):
        assert self.eval_mode and self.n_tasks > 0
        source, task_idx = self._global_to_local_task_index(global_task_idx)
        sampler = self.class_samplers[source]
        episode_description = sampler.sample_episode_description(task_idx)
        episode_description = tuple((class_id + sampler.class_set[0], num_support) for class_id, num_support, _ in episode_description)

        support_images = []
        for class_id, nb_support in episode_description:
            support_images_ = []
            assert nb_support  <= len(self.class_images[source][class_id]), \
                f'Failed fetching {nb_support} images from {source} at class {class_id}.'
            
            if not self.fix_seed:
                random.shuffle(self.class_images[source][class_id])

            # support
            for j in range(0, nb_support):
                x = self.get_next(source, class_id, self.class_images[source][class_id][j])
                support_images_.append(x)
            
            support_images.append(torch.stack(support_images_, dim=0))
        
        support_images = torch.stack(support_images, dim=0)

        return support_images

    def sample_episode(self, global_task_idx):
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # select which dataset to form episode
        if self.eval_mode and self.n_tasks > 0:
            if self.dataset is not None:
                source = self.dataset
                task_idx = global_task_idx
                global_task_idx = self._local_to_global_task_index(source, task_idx)
            else:
                source, task_idx = self._global_to_local_task_index(global_task_idx)
        else:
            source = np.random.choice(self.datasets)
            task_idx = None
        sampler = self.class_samplers[source]

        # episode details: (class_id, nb_supp, nb_qry)
        if self.n_tasks > 0:
            if not self.eval_mode:
                task_idx = np.random.randint(0, self.n_tasks_per_dataset[source])
                global_task_idx = self._local_to_global_task_index(source, task_idx)
            task_index = torch.tensor(global_task_idx, dtype=torch.long)

        episode_description = sampler.sample_episode_description(task_idx)
        episode_description = tuple( # relative ids --> abs ids
            (class_id + sampler.class_set[0], num_support, num_query)
            for class_id, num_support, num_query in episode_description)
        episode_classes = list({class_ for class_, _, _ in episode_description})

        data_index = []
        if self.apply_task_transform:
            task_transform = get_consistent_transform(self.data_config)
        for class_id, nb_support, nb_query in episode_description:
            assert nb_support + nb_query <= len(self.class_images[source][class_id]), \
                f'Failed fetching {nb_support + nb_query} images from {source} at class {class_id}.'
            
            if not self.fix_seed:
                random.shuffle(self.class_images[source][class_id])

            if self.apply_class_transform:
                class_transform = get_consistent_transform(self.data_config)

            # support
            for j in range(0, nb_support):
                #print('support fetch:', sup_added, class_id)
                x = self.get_next(source, class_id, self.class_images[source][class_id][j], self.data_transforms)
                if self.apply_task_transform:
                    x = task_transform(x)
                if self.apply_class_transform:
                    x = class_transform(x)
                support_images.append(x)
                data_index.append(int(self.class_images[source][class_id][j]))

            # query
            for j in range(nb_support, nb_support + nb_query):
                x = self.get_next(source, class_id, self.class_images[source][class_id][j])
                query_images.append(x)

            support_labels.extend([episode_classes.index(class_id)] * nb_support)
            query_labels.extend([episode_classes.index(class_id)] * nb_query)

        support_images = torch.stack(support_images, dim=0)
        query_images = torch.stack(query_images, dim=0)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        data = (support_images, support_labels, query_images, query_labels)
        if self.n_tasks > 0:
            data_index = torch.tensor(data_index, dtype=torch.long)
            data = (*data, task_index, data_index)

        return data

    def sample_full_episode(self, global_task_idx, max_samples=1000):
        assert self.eval_mode and self.n_tasks > 0
        source, task_idx = self._global_to_local_task_index(global_task_idx)
        sampler = self.class_samplers[source]
        episode_description = sampler.sample_episode_description(task_idx)
        class_ids = [class_id + sampler.class_set[0] for class_id, _, _ in episode_description]

        images = []
        labels = []
        for class_id in class_ids:
            x = self._load_images(source, class_id, max_samples)
            images.append(x)
            labels.append(class_ids.index(class_id) * torch.ones(len(x), dtype=torch.long))
        
        return images, labels
    

class FinetuneMetaDatasetH5(FullMetaDatasetH5):
    def __init__(self, args, split=Split['TRAIN']):
        super().__init__(args, split, eval_mode=True)

    def __getitem__(self, idx):
        return self.sample_full_episode(idx)