import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
from timm.utils import accuracy
from tqdm import tqdm
import random
from copy import deepcopy
from functools import partial
from pprint import pprint

from src.datasets.dataloaders import FineTuneDataset
from src.models.feature_extractors.pretrained_fe import get_fe_metadata
from src.models.CAML import CAML
from src.models.feature_extractors.customed_fe import VisionTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(x: list, y: list, max_support, max_query, batch_size, transform=None):
    x_s, y_s, x_q, y_q = [], [], [], []
    for _ in range(batch_size):
        x_s_, y_s_, x_q_, y_q_ = [], [], [], []
        for c in range(len(x)):
            ids = torch.randperm(len(x[c][0]))
            k = len(ids) // 2
            ids_s = ids[:min(k, max_support)]
            ids_q = ids[k:k + min(k, max_query)]
            x_s_.append(x[c][0][ids_s])
            y_s_.append(y[c][0][ids_s])
            x_q_.append(x[c][0][ids_q])
            y_q_.append(y[c][0][ids_q])

        if transform is not None:
            x_s.append(transform(torch.cat(x_s_)))
            y_s.append(torch.cat(y_s_))
            x_q.append(transform(torch.cat(x_q_)))
            y_q.append(torch.cat(y_q_))
        else:    
            x_s.append(torch.cat(x_s_))
            y_s.append(torch.cat(y_s_))
            x_q.append(torch.cat(x_q_))
            y_q.append(torch.cat(y_q_))

    x_s = torch.stack(x_s)
    y_s = torch.stack(y_s)
    x_q = torch.stack(x_q)
    y_q = torch.stack(y_q)
    
    return x_s, y_s, x_q, y_q


def finetune(model, target_params_keys, batch, n_steps, lr, max_support, max_query, batch_size, save_traj, device, dtype):
    model.train()
    opt = torch.optim.Adam([param for name, param in model.named_parameters() if name in target_params_keys], lr=lr)
    
    x, y = batch
    if save_traj:
        traj = [torch.stack([deepcopy(param.data).cpu() for name, param in model.named_parameters() if name in target_params_keys])]
    accs = []
    losses = []
    with tqdm(range(n_steps), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}", leave=False) as pbar:
        for _ in pbar:
            x_s, y_s, x_q, y_q = split_data(x, y, max_support, max_query, batch_size)
            x_s, y_s, x_q, y_q = x_s.to(device=device, dtype=dtype), y_s.to(device=device), x_q.to(device=device, dtype=dtype), y_q.to(device=device)
            
            inp = torch.cat([x_s, x_q], dim=1)
            n_s = y_s.size(1)
            y_q = y_q.flatten()
            bias_idx = torch.arange(batch_size)
            
            output = model(inp, y_s, n_s, bias_idx)
            loss = criterion(output, y_q)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(f'Loss: {loss.item():.4f}')
            
            if save_traj:
                tgt_params = torch.stack([deepcopy(param.data).cpu() for name, param in model.named_parameters() if name in target_params_keys])
                traj.append(tgt_params)
            acc = accuracy(output, y_q)[0].item()
            accs.append(acc)
            losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        x_s, y_s, x_q, y_q = split_data(x, y, max_support, max_query, batch_size)
        x_s, y_s, x_q, y_q = x_s.to(device=device, dtype=dtype), y_s.to(device=device), x_q.to(device=device, dtype=dtype), y_q.to(device=device)
        
        inp = torch.cat([x_s, x_q], dim=1)
        n_s = y_s.size(1)
        y_q = y_q.flatten()
        bias_idx = torch.arange(batch_size)
        
        output = model(inp, y_s, n_s, bias_idx)
        loss = criterion(output, y_q).item()
        acc = accuracy(output, y_q)[0].item()

    if save_traj:
        traj = torch.stack(traj)
    accs = torch.tensor(accs)
    losses = torch.tensor(losses)

    if save_traj:
        traj_data = {'traj': traj, 'accs': accs, 'losses': losses}
    else:
        traj_data = {'accs': accs, 'losses': losses}

    return loss, acc, traj_data



########################################################################################################################
########################################################################################################################
########################################################################################################################



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['imagenet1k', 'mscoco', 'fungi', 'wikiart_artist', 'wikiart_genre', 'wikiart_style'])
    parser.add_argument('--save_traj', action='store_true', default=False)
    parser.add_argument('--target_param', type=str, choices=[
        'te',
        'te_bias',
        'te_qkv_bias',
        'te_even_qkv_bias',
        'te_q_bias',
        'te_even_q_bias',
        'fe',
        'fe_bias',
        'fe_qkv_bias',
        'fe_even_qkv_bias',
        'fe_q_bias',
        'fe_even_q_bias',
    ])
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_support', type=int, default=5)
    parser.add_argument('--max_query', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--n_tasks', type=int, default=500)
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--model_path', type=str, default='/common_datasets/METAFLOW_DATASETS/caml_pretrained_models/CAML_Laion2b')
    parser.add_argument('--fe_dtype', type=str, default='bfloat16')
    parser.add_argument('--model', type=str, default='CAML')
    parser.add_argument('--fe_type', type=str, default="timm:vit_huge_patch14_clip_224.laion2b:1280")
    parser.add_argument('--freeze_fe', action='store_true', default=False)
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(0)

    ############################################################
    # setting device and dtype
    device = torch.device('cuda')
    dtype = torch.bfloat16
    print(f"device: {device}, dtype: {dtype}")
    
    
    ############################################################
    # loading CAML model with backbone
    
    # fe_type = "timm:vit_large_patch14_clip_224.openai:1024"
    # fe_type = "timm:vit_huge_patch14_clip_224.laion2b:1280"
    # fe_type = "cache:timm:vit_huge_patch14_clip_224.laion2b:1280"
    
    print(f"Creating model: CAML with {args.fe_type}")
    
    fe_metadata = get_fe_metadata(args)
    feature_extractor = fe_metadata['fe']
    
    # load custom fe
    if not args.freeze_fe:
        feature_extractor = VisionTransformer(
            patch_size=14,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            pre_norm=True, 
            norm_layer=nn.LayerNorm,
            num_classes=0,
            n_bias=args.batch_size,
            weight_init='skip'
        ).to(device=device, dtype=dtype)
        
        fe_dict = fe_metadata['fe'].state_dict()
        for b in range(32):
            for i, c in enumerate(['q', 'k', 'v']):
                fe_dict[f'blocks.{b}.attn.qkv_bias.{c}_bias'] = fe_dict[f'blocks.{b}.attn.qkv.bias'][i*1280:(i+1)*1280].unsqueeze(0).repeat_interleave(args.batch_size, dim=0)
            del fe_dict[f'blocks.{b}.attn.qkv.bias']
        feature_extractor.load_state_dict(fe_dict, strict=True)
    
    model = CAML(
        feature_extractor=feature_extractor,
        fe_dim=fe_metadata['fe_dim'],
        fe_dtype=dtype,
        train_fe=not args.freeze_fe, # whether to update the feature encoder weights during meta-training
        encoder_size="laion",
        num_bias=args.batch_size,
        dropout=0.0,
        label_elmes=True,
        device=device,
        set_transformer=False
    )
    
    # load checkpoint
    model_path = os.path.join(args.model_path, 'model.pth')
    model_dict = torch.load(model_path, map_location='cuda')
    for b in range(24):
        for i, c in enumerate(['q', 'k', 'v']):
            model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{b}.self_attention.in_proj_bitfit_bias.{c}_bias'] = \
                model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{b}.self_attention.in_proj_bias'][i*1536:(i+1)*1536].unsqueeze(0).repeat_interleave(args.batch_size, dim=0)
    model_dict["transformer_encoder.elmes_scale"] = model_dict['transformer_encoder.etf_scale']
    model_dict["transformer_encoder.label_elmes"] = model_dict["transformer_encoder.label_etf"]
    del model_dict['transformer_encoder.label_etf']
    del model_dict['transformer_encoder.etf_scale']
    model.load_state_dict(model_dict, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"checkpoint loaded: {model_path}")
    
    
    
    # print(model.feature_extractor)
    
    # exit()
    
    # param_keys = model.state_dict()
    # print(param_keys['feature_extractor.blocks.21.attn.qkv.bias'].shape)
    # with open("parameters_with_fe.txt", "w") as f:
    #     for key in param_keys:
    #         f.write(key + "\n")
    
    
    ############################################################
    # loading FULL batch dataset
    domain = args.dataset
    split = args.split
    
    train_transforms = fe_metadata['train_transform']
    test_transforms = fe_metadata['test_transform']
    
    dataset = FineTuneDataset(domain, split, train_transforms, offset=args.offset)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, pin_memory=True)    
    print(f"dataset loaded: {domain} {split}")
    
    
    ############################################################
    # select target parameter
    model_dict = deepcopy(model.state_dict())
    
    def te_get_block_num(param_key):
        s = param_key.split('.')[3]
        match = re.search(r'\d+$', s)
        if match:
            return int(match.group(0))
        return None
    
    def fe_get_block_num(param_key):
        block_ids = re.findall(r'\d+', param_key)[0]
        return int(block_ids)
    
    
    def te_condition(param_key):
        return 'transformer_encoder' in param_key
    def te_bias_condition(param_key):
        return full_condition(param_key) and 'bias' in param_key
    def te_qkv_bias_condition(param_key):
        return bias_condition(param_key) and 'bitfit' in param_key
    def te_even_qkv_bias_condition(param_key):
        return qkv_bias_condition(param_key) and te_get_block_num(param_key)%2==0
    def te_q_bias_condition(param_key):
        return qkv_bias_condition(param_key) and 'q_bias' in param_key
    def te_even_q_bias_condition(param_key):
        return even_qkv_bias_condition(param_key) and 'q_bias' in param_key
    
    def fe_condition(param_key):
        return 'feature_extractor' in param_key
    def fe_bias_condition(param_key):
        return fe_condition(param_key) and 'bias' in param_key
    def fe_qkv_bias_condition(param_key):
        return fe_bias_condition(param_key) and 'qkv' in param_key
    def fe_even_qkv_bias_condition(param_key):
        return fe_qkv_bias_condition(param_key) and fe_get_block_num(param_key)%2==0
    def fe_q_bias_condition(param_key):
        return fe_qkv_bias_condition(param_key) and 'q_bias' in param_key
    def fe_even_q_bias_condition(param_key):
        return fe_q_bias_condition(param_key) and fe_get_block_num(param_key)%2==0
    
    if args.target_param == 'te':
        filter_target_params = te_condition
    elif args.target_param == 'te_bias':
        filter_target_params = te_bias_condition
    elif args.target_param == 'te_qkv_bias':
        filter_target_params = te_qkv_bias_condition
    elif args.target_param == 'te_even_qkv_bias':
        filter_target_params = te_even_qkv_bias_condition
    elif args.target_param == 'te_q_bias':
        filter_target_params = te_q_bias_condition
    elif args.target_param == 'te_even_q_bias':
        filter_target_params = te_even_q_bias_condition

    elif args.target_param == 'fe':
        filter_target_params = fe_condition
    elif args.target_param == 'fe_bias':
        filter_target_params = fe_bias_condition
    elif args.target_param == 'fe_qkv_bias':
        filter_target_params = fe_qkv_bias_condition
    elif args.target_param == 'fe_even_qkv_bias':
        filter_target_params = fe_even_qkv_bias_condition
    elif args.target_param == 'fe_q_bias':
        filter_target_params = fe_q_bias_condition
    elif args.target_param == 'fe_even_q_bias':
        filter_target_params = fe_even_q_bias_condition
        
    
    target_params_keys = list(filter(filter_target_params, model.state_dict().keys()))
    
    print(f"target param: {args.target_param}")
    
    ############################################################
    # prepare finetuning
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.lr is not None:
        save_root = f"outputs/ft_trajs_{args.target_param}_{args.split}_{args.n_steps}steps_{args.lr}lr_{args.n_seeds}seeds_{args.noise}noise"
    else:
        save_root = f"outputs/ft_trajs_{args.target_param}_{args.split}_lr_searching"
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save directory: {save_dir}")

    # search lrs
    if args.lr is None:
        best_lrs = {}
        if args.dataset in best_lrs:
            best_lr = best_lrs[args.dataset]
            print(f'Use Best LR: {best_lr}')
        else:
            # lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
            lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
            best_lr = None
            best_acc = 0
            n_search_steps = 20
            
            for lr in lrs:
                total_acc = 0
                
                losses = []
                accs = []
                with tqdm(dataloader, total=n_search_steps, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
                        leave=False, desc=f'Searching LR={lr}') as pbar:
                    losses_per_task = []
                    accs_per_task = []
                    for task_idx, batch in enumerate(pbar):
                        model.load_state_dict(model_dict, strict=True)
                        
                        for key, param in model.named_parameters():
                            if key in target_params_keys:
                                with torch.no_grad():
                                    param += torch.randn_like(param) * args.noise
                        
                        loss, acc, traj_data = finetune(model, target_params_keys, batch, args.n_steps, lr, args.max_support, args.max_query, args.batch_size, args.save_traj, device, dtype)
                        total_acc += acc
                        
                        accs_per_task.append(traj_data['accs'])
                        losses_per_task.append(traj_data['losses'])
                        
                        if task_idx == n_search_steps - 1:
                            break
                    losses_per_task = torch.stack(losses_per_task)
                    accs_per_task = torch.stack(accs_per_task)
                    losses.append(losses_per_task)
                    accs.append(accs_per_task)
                    
                losses = torch.stack(losses)
                accs = torch.stack(accs)
                    
                np.save(os.path.join(save_dir, f'acc_{lr}lr.npy'), accs.numpy())
                np.save(os.path.join(save_dir, f'loss_{lr}lr.npy'), losses.numpy())
                total_acc /= n_search_steps
                print(f'LR: {lr}, Acc: {total_acc}')
                if total_acc > best_acc:
                    best_acc = total_acc
                    best_lr = lr
                    
            print(f'Best LR: {best_lr}, Best Acc: {best_acc}')
    else:
        best_lr = args.lr
        print(f'learning rate: {best_lr}')
        
    if args.lr is None:
        exit()
    
    print("start finetuning!!")
    total_loss = 0
    total_acc = 0
    total = len(dataloader) if args.n_tasks is None else min(args.n_tasks, len(dataloader))
    with tqdm(dataloader, total=total, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}") as pbar:
        for task_idx, batch in enumerate(pbar):
            
            if os.path.exists(os.path.join(save_dir, f'ckpt_{task_idx:06d}.npy')):
                continue
            
            if args.save_traj:
                trajs = []
            accs = []
            losses = []
            
            n_seed_batches = args.n_seeds // args.batch_size
            
            for seed in range(n_seed_batches):
                model.load_state_dict(model_dict, strict=True)
                
                for key, param in model.named_parameters():
                    if key in target_params_keys:
                        with torch.no_grad():
                            param += torch.randn_like(param) * args.noise

                loss, acc, traj_data = finetune(model, target_params_keys, batch, args.n_steps, best_lr, args.max_support, args.max_query, args.batch_size, args.save_traj, device, dtype)
                total_loss += loss / n_seed_batches
                total_acc += acc / n_seed_batches

                if args.save_traj:
                    trajs.append(traj_data['traj'])
                accs.append(traj_data['accs'])
                losses.append(traj_data['losses'])
            
            if args.save_traj:
                trajs = torch.stack(trajs)
            accs = torch.stack(accs)
            losses = torch.stack(losses)
            
            if args.save_traj:
                torch.save(trajs, os.path.join(save_dir, f'ckpt_{task_idx+args.offset:06d}.pth'))
            np.save(os.path.join(save_dir, f'acc_{task_idx+args.offset:06d}.npy'), accs.numpy())
            np.save(os.path.join(save_dir, f'loss_{task_idx+args.offset:06d}.npy'), losses.numpy())

            pbar.set_description(f'Avg loss: {total_loss / (task_idx + 1):.4f}, Avg acc: {total_acc / (task_idx + 1):.4f}')

            if args.n_tasks is not None and task_idx == args.n_tasks - 1:
                break

    print(f'Avg loss: {total_loss / total}, Avg acc: {total_acc / total}')