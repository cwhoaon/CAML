import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dataloaders import FineTuneDataset, AggregatedDataset, EpisodicDataset
from src.models.feature_extractors.pretrained_fe import get_fe_metadata
from src.models.CAML import CAML

from src.evaluation.datasets import dataloaders
from pmf_models.utils import DiffAugment


from meta_flow import Flow, FlowDataset, train_collate_fn

def to_device(data, device):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
            
    return to_device_wrapper(data)

def process_batch(inp, way=5, shot=5, query_shot=16, device=torch.device('cuda'), dtype=torch.bfloat16):
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).to(device)
    support_labels = torch.arange(way).repeat(shot, 1).T.flatten().to(device)
    bias_idx = torch.arange(1).to(device)
    inp = inp.to(device, dtype)
    valid_batch = inp, support_labels, target, bias_idx
    return valid_batch


@torch.no_grad()
def evaluate_single_batch(model, batch, device=torch.device('cuda'), dtype=torch.bfloat16):
    model.eval()
    inp, y_s, y_q, bias_idx = batch
    n_s = 25
    y_q_pred = model(inp, y_s, n_s, bias_idx)

    acc = (y_q_pred.argmax(-1) == y_q).float().mean()
    loss = F.cross_entropy(y_q_pred, y_q)
    
    return acc, loss

def find_lr(model, dataloader, model_dict, way=5, shot=5, device=torch.device('cuda'), dtype=torch.bfloat16):
    best_acc = 0
    best_lr = 0
    for lr in [0, 0.00001, 0.00005, 0.0001, 0.001]:
        print(f"Testing: {lr}")
        acc_lr = []
        for i, (inp, _) in enumerate(dataloader):
            if i >= 5:
                break
            
            valid_batch = process_batch(inp)
            inp, y_s, y_q, bias_idx = valid_batch
            x_s = inp[:way*shot]
            finetune_batch = x_s, y_s, bias_idx
            
            if lr != 0:
                model = finetune(model, finetune_batch, lr, model_dict)
            acc, loss = evaluate_single_batch(model, valid_batch)
            acc_lr.append(acc)
        
        acc = torch.stack(acc_lr).mean()
        print(f"Acc: {acc}")
            
        if acc >= best_acc:
            best_acc = acc
            best_lr = lr

    print(f"Selected lr: {best_lr}, best_acc: {best_acc.item()}")
    return best_lr


def finetune(model, finetune_batch, lr, model_dict, num_iter=50, device=torch.device('cuda'), dtype=torch.bfloat16):
    model.train()
    model.load_state_dict(model_dict, strict=False)
    
    aug_prob = 0.9
    aug_types = ['color', 'translation']
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.
            )
    
    x_s, y_s, bias_idx = finetune_batch
    n_s = x_s.shape[0]
    
    for i in range(num_iter):
        opt.zero_grad()
        
        z = DiffAugment(x_s, aug_types, aug_prob, detach=True)
        inp = torch.cat([x_s, z], dim=0)
        z_pred = model(inp, y_s, n_s, bias_idx)
        loss = criterion(z_pred, y_s)
        loss.backward()
        opt.step()
    
    return model
        
    

def evaluate(eval_version, model, eval_dataloader, euler_steps, p_init, dt, model_dict, way=5, shot=5, query_shot=16, device=torch.device('cuda'), dtype=torch.bfloat16):
    accs, losses = [], []
    
    if eval_version == 'finetune':
        lr = find_lr(model, eval_dataloader, model_dict)
        print(f"Selected lr: {lr}")
    
    
    for valid_step, (inp, _) in tqdm(enumerate(eval_dataloader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}", leave=False):
        valid_batch = process_batch(inp)
        
        if eval_version == 'vanilla':
            acc, loss = evaluate_single_batch(model, valid_batch, device=device, dtype=dtype)
            accs.append(acc)
            losses.append(loss)
        
        if eval_version == 'flow':
            inp, support_labels, target, bias_idx = valid_batch
            
            with torch.no_grad():
                x_s_feat = model.get_feature_vector(inp[:way*shot])
            p_traj = flow.inference(p_init, x_s_feat, y=support_labels.unsqueeze(0), euler_step=max(euler_steps), dt=dt, get_traj=True)
            
            # print(p_traj.shape)
            # print(euler_steps)
            
            acc_steps = []
            loss_steps = []
            
            for euler_step in euler_steps:
                if euler_step == 0:
                    p = p_init
                else:
                    p = p_traj[euler_step-1]
                
                for b in range(p_init.shape[1]):
                    model.transformer_encoder.encoder.layers[b*2].self_attention.in_proj_bitfit_bias.q_bias.data = p[:, b]
                acc, loss = evaluate_single_batch(model, valid_batch, device=device, dtype=dtype)
                acc_steps.append(acc)
                loss_steps.append(loss)
            acc_steps = torch.stack(acc_steps)
            loss_steps = torch.stack(loss_steps)
            accs.append(acc_steps)
            losses.append(loss_steps)
        
        if eval_version == 'finetune':
            inp, y_s, y_q, bias_idx = valid_batch
            x_s = inp[:way*shot]
            finetune_batch = x_s, y_s, bias_idx
            model = finetune(model, finetune_batch, lr, model_dict, device=device, dtype=dtype)
            acc, loss = evaluate_single_batch(model, valid_batch)
            accs.append(acc)
            losses.append(loss)

        # pbar.update()
    accs = torch.stack(accs).cpu().float()
    losses = torch.stack(losses).cpu().float()
    print()
    print(f"Acc: {accs.mean()}, Loss: {losses.mean()}")
    return accs, losses


##############################################################################################################
##############################################################################################################



def parse_args(shell_script=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval_version', type=str, choices=['vanilla', 'flow', 'finetune'])
    parser.add_argument('--datasets', nargs="+", default=['aircraft', 'chestX', 'cifar_fs', 'paintings'])
    
    parser.add_argument('--euler_steps', type=int, nargs='+', default=[0, 1, 5, 10, 20, 100])
    parser.add_argument('--n_tasks', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--backbone_type', type=str, default='resnet18')
    parser.add_argument('--update_backbone', '-ub', action='store_true', default=False)
    parser.add_argument('--regression_mode', action='store_true', default=False)
    parser.add_argument('--ignore_time', action='store_true', default=False)
    
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--model_path', type=str, default='/common_datasets/METAFLOW_DATASETS/caml_pretrained_models/CAML_Laion2b')
    parser.add_argument('--flow_path', type=str, default='/data3/CAML/outputs_metaflow/')
    parser.add_argument('--fe_type', type=str, default="timm:vit_huge_patch14_clip_224.laion2b:1280")
    parser.add_argument('--fe_dtype', type=str, default='bfloat16')
    parser.add_argument('--model', type=str, default='CAML')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--exp_tag', type=str)
    parser.add_argument('--is_gen', action="store_true", default=False)
    
    
    if shell_script is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shell_script.split())

    return args


if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device('cuda')
    dtype = torch.bfloat16
    
    #######################################################
    # Load classifier
    fe_metadata = get_fe_metadata(args)
    model = CAML(
        feature_extractor=fe_metadata['fe'],
        fe_dim=fe_metadata['fe_dim'],
        fe_dtype=dtype,
        train_fe=False, # whether to update the feature encoder weights during meta-training
        encoder_size="laion",
        num_bias=1,
        dropout=0.0,
        label_elmes=True,
        device=device,
        set_transformer=False
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    model_path = os.path.join(args.model_path, 'model.pth')
    model_dict = torch.load(model_path, map_location='cuda')
    for i in range(24):
        bias = model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias'].unsqueeze(0)
        d = bias.shape[1] // 3
        model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.q_bias'] = bias[:,:d]
        model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.k_bias'] = bias[:,d:2*d]
        model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.v_bias'] = bias[:,2*d:]
    model_dict["transformer_encoder.elmes_scale"] = model_dict['transformer_encoder.etf_scale']
    model_dict["transformer_encoder.label_elmes"] = model_dict["transformer_encoder.label_etf"]
    del model_dict['transformer_encoder.label_etf']
    del model_dict['transformer_encoder.etf_scale']
    
    model.load_state_dict(model_dict, strict=False)
    
    p_init= None
    if args.eval_version == 'flow':
        p_init = [model.transformer_encoder.encoder.layers[b*2].self_attention.in_proj_bitfit_bias.q_bias.data for b in range(12)]
        p_init = torch.cat(p_init).unsqueeze(0)
    
        
    #######################################################
    # Load flow model
    dt=None
    if args.eval_version == "flow":
        if args.regression_mode:
            args.ignore_time = True
            args.euler_steps = [0, 1]
            dt = 1 / (n_timesteps - 1)
        else:
            n_pseudo_timesteps = 51
            args.euler_steps = [i for i in range(0, n_pseudo_timesteps, 10)] + [1, 5]
            dt = 1 / (n_pseudo_timesteps - 1)

        n_modules = 12
        n_params = 1536
        backbone = None
        flow = Flow(n_params, args.hidden_dim, args.n_layers, args.n_blocks, n_modules, backbone_type=args.backbone_type,
                    freeze_backbone=(not args.update_backbone), backbone=backbone, ignore_time=args.ignore_time).to(device, dtype=dtype)
        flow_path = os.path.join(args.flow_path, args.exp_tag)
        model_type = 'gen' if args.is_gen else 'recon'
        flow_path = os.path.join(flow_path, f'checkpoints/best_{model_type}.pth')
        flow_dict = torch.load(flow_path, map_location='cuda')
        flow.load_state_dict(flow_dict['flow'], strict=True)
        flow.eval()
        
    #######################################################
    # Load dataloader
    train_transforms = fe_metadata['train_transform']
    test_transforms = fe_metadata['test_transform']
    # support_dataset_test = AggregatedDataset(args.base_sources, 'test', args.n_tasks, fix_seed=True, transform=test_transforms)
    # dataloader = DataLoader(support_dataset_test, batch_size=1, shuffle=False, pin_memory=False)
    # train_eval_loader = get_data_loader(train_eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=False)
    # valid_eval_loader = get_data_loader(valid_eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=False)
    
    for i, dataset in enumerate(args.datasets):
        data_path = f"/common_datasets/METAFLOW_DATASETS/caml_universal_eval_datasets/{dataset}/test"
        episode_path = f"/common_datasets/METAFLOW_DATASETS/test_episodes/{dataset}.pth"
        eval_loader = dataloaders.meta_test_dataloader(
            data_path=data_path,
            episode_path=episode_path,
            way=5,
            shot=5,
            pre=False,
            transform_type=test_transforms,
            query_shot=16,
            trial=args.n_tasks
            # trial=2
            )
        print(f"{dataset} dataset loaded")    
        
        print("Start evaluation!")
        accs, losses = evaluate(args.eval_version, model, eval_loader, args.euler_steps, p_init, dt, model_dict)
        
        save_path = os.path.join(args.save_dir, args.eval_version)
        if args.eval_version == 'flow':
            save_path = os.path.join(save_path, f"{model_type}_{args.exp_tag}")
        save_path = os.path.join(save_path, dataset)
        
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"acc.npy"), accs)
        np.save(os.path.join(save_path, f"loss.npy"), losses)