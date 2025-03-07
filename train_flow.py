import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import os
import argparse
import shutil
import gc
import random
import numpy as np

from tqdm import tqdm
# from easydict import EasyDict
from copy import deepcopy
from glob import glob
from timm.scheduler import create_scheduler
from einops import rearrange
# import wandb


from meta_flow import Flow, FlowDataset, train_collate_fn

from src.datasets.dataloaders import FineTuneDataset, AggregatedDataset, EpisodicDataset
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


def get_data_loader(dataset, batch_size, shuffle, num_workers, persistent_workers=False, pin_memory=True, seed=42, collate_fn=None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=num_workers, persistent_workers=persistent_workers,
                      pin_memory=pin_memory, worker_init_fn=seed_worker, generator=generator)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def get_data_iterator(data_loader, device):
    '''
    Iterator wrapper for dataloader
    '''
    def get_batch():
        while True:
            for batch in data_loader:
                yield to_device(batch, device)
    return get_batch()

    
@torch.no_grad()
def evaluate(flow, model, eval_loader, euler_steps, dt, n_valid_steps, device, logger=None, tag='', global_step=0, gaudi_trigger=None):
    assert eval_loader.batch_size == 1, 'Batch size of eval_loader must be 1'
    dataset_names = eval_loader.dataset.datasets
    recon_losses = {euler_step: {d: [] for d in dataset_names} for euler_step in euler_steps}
    recon_task_accs = {euler_step: {d: [] for d in dataset_names} for euler_step in euler_steps}
    recon_task_losses = {euler_step: {d: [] for d in dataset_names} for euler_step in euler_steps}
    gen_task_accs = {euler_step: {d: [] for d in dataset_names} for euler_step in euler_steps}
    gen_task_losses = {euler_step: {d: [] for d in dataset_names} for euler_step in euler_steps}
    if n_valid_steps < 0:
        n_valid_steps = len(eval_loader)
    
    
    n_blocks = len(model.feature_extractor.blocks)
    p_vanilla = torch.cat([model.feature_extractor.blocks[b*2].attn.qkv_bias.q_bias.data for b in range(n_blocks//2)]).unsqueeze(0)
    flow.eval()
    pbar = tqdm(total=n_valid_steps, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}", leave=False)
    pbar.set_description('performing evaulation')
    for valid_step, valid_batch in enumerate(eval_loader):
        valid_batch = to_device(valid_batch, device)
        
        dataset_name = dataset_names[valid_step % len(dataset_names)]
        
        p_init, p_target, x_s, y_s, x_q, y_q = valid_batch
        x_s, y_s, x_q, y_q = x_s.to(device=device, dtype=dtype), y_s.to(device=device), x_q.to(device=device, dtype=dtype), y_q.to(device=device)
        
        p_traj_recon = flow.inference(p_init, x_s, y=y_s, euler_step=max(euler_steps), dt=dt, get_traj=True)
        p_traj_gen = flow.inference(p_vanilla, x_s, y=y_s, euler_step=max(euler_steps), dt=dt, get_traj=True)
        
        for euler_step in euler_steps:
            # evaluate reconstruction (from simulated init) and generation (from meta-trained init) loss
            if euler_step > 0:
                p_recon = p_traj_recon[euler_step - 1]
                p_gen = p_traj_gen[euler_step - 1]
            else:
                p_recon = p_init
                p_gen = p_vanilla
            
            # evaluate reconstruction loss
            recon_loss = F.mse_loss(p_recon, p_target)
            if gaudi_trigger is not None:
                gaudi_trigger()
            recon_losses[euler_step][dataset_name].append(recon_loss.item())

            # evaluate reconstruction task accuracy and loss
            for b in range(p_vanilla.shape[1]):
                model.feature_extractor.blocks[b*2].attn.qkv_bias.q_bias.data = p_recon[:, b]
                # model.transformer_encoder.encoder.layers[b*2].self_attention.in_proj_bitfit_bias.q_bias.data = p_recon[:, b]
            inp = torch.cat([x_s, x_q], dim=1)
            n_s = y_s.size(1)
            y_q = y_q.flatten()
            bias_idx = torch.arange(1)
            y_q_pred = model(inp, y_s, n_s, bias_idx)
            if gaudi_trigger is not None:
                gaudi_trigger()
            
            recon_task_acc = (y_q_pred.argmax(-1) == y_q).float().mean()
            recon_task_loss = F.cross_entropy(y_q_pred, y_q)
            recon_task_accs[euler_step][dataset_name].append(recon_task_acc.item())
            recon_task_losses[euler_step][dataset_name].append(recon_task_loss.item())
            
            # evaluate generation task accuracy and loss
            for b in range(p_vanilla.shape[1]):
                model.feature_extractor.blocks[b*2].attn.qkv_bias.q_bias.data = p_gen[:, b]
                # model.transformer_encoder.encoder.layers[b*2].self_attention.in_proj_bitfit_bias.q_bias.data = p_gen[:, b]
            y_q_pred = model(inp, y_s, n_s, bias_idx)
            if gaudi_trigger is not None:
                gaudi_trigger()
            gen_task_acc = (y_q_pred.argmax(-1) == y_q).float().mean()
            gen_task_loss = F.cross_entropy(y_q_pred, y_q)
            gen_task_accs[euler_step][dataset_name].append(gen_task_acc.item())
            gen_task_losses[euler_step][dataset_name].append(gen_task_loss.item())

        pbar.update()
        del valid_batch
        gc.collect()
        if valid_step + 1 == n_valid_steps:
            break
    pbar.close()
    flow.train()

    # restore model state
    for b in range(p_vanilla.shape[1]):
        model.feature_extractor.blocks[b*2].attn.qkv_bias.q_bias.data = p_recon[:, b]
        # model.transformer_encoder.encoder.layers[b*2].self_attention.in_proj_bitfit_bias.q_bias.data = p_vanilla[:, b]

    for euler_step in euler_steps:
        for dataset_name in dataset_names:
            recon_losses[euler_step][dataset_name] = np.mean(recon_losses[euler_step][dataset_name])
            recon_task_accs[euler_step][dataset_name] = np.mean(recon_task_accs[euler_step][dataset_name])
            recon_task_losses[euler_step][dataset_name] = np.mean(recon_task_losses[euler_step][dataset_name])
            gen_task_accs[euler_step][dataset_name] = np.mean(gen_task_accs[euler_step][dataset_name])
            gen_task_losses[euler_step][dataset_name] = np.mean(gen_task_losses[euler_step][dataset_name])
    
    recon_accs = torch.tensor([np.mean(list(recon_task_accs[euler_step].values())) for euler_step in euler_steps])
    best_recon_acc, recon_euler_idx = torch.max(recon_accs, dim=0)
    gen_accs = torch.tensor([np.mean(list(gen_task_accs[euler_step].values())) for euler_step in euler_steps])
    best_gen_acc, gen_euler_idx = torch.max(gen_accs, dim=0)

    if logger is not None:
        log_dict = {}
        for euler_step in euler_steps:
            log_dict[f'_avg_recon_loss{tag}/{euler_step}step'] = np.mean(list(recon_losses[euler_step].values()))
            log_dict[f'_avg_recon_task_acc{tag}/{euler_step}step'] = np.mean(list(recon_task_accs[euler_step].values()))
            log_dict[f'_avg_recon_task_loss{tag}/{euler_step}step'] = np.mean(list(recon_task_losses[euler_step].values()))
            log_dict[f'_avg_gen_task_acc{tag}/{euler_step}step'] = np.mean(list(gen_task_accs[euler_step].values()))
            log_dict[f'_avg_gen_task_loss{tag}/{euler_step}step'] = np.mean(list(gen_task_losses[euler_step].values()))
            for dataset_name in dataset_names:
                log_dict[f'{dataset_name}/recon_loss{tag}_{euler_step}step'] = recon_losses[euler_step][dataset_name]
                log_dict[f'{dataset_name}/recon_task_acc{tag}_{euler_step}step'] = recon_task_accs[euler_step][dataset_name]
                log_dict[f'{dataset_name}/recon_task_loss{tag}_{euler_step}step'] = recon_task_losses[euler_step][dataset_name]
                log_dict[f'{dataset_name}/gen_task_acc{tag}_{euler_step}step'] = gen_task_accs[euler_step][dataset_name]
                log_dict[f'{dataset_name}/gen_task_loss{tag}_{euler_step}step'] = gen_task_losses[euler_step][dataset_name]
        log_dict[f'_avg_recon_task_acc{tag}/best'] = best_recon_acc.item()
        log_dict[f'_avg_gen_task_acc{tag}/best'] = best_gen_acc.item()
        
        for k, v in log_dict.items():
            logger.add_scalar(k, v, global_step=global_step)
        # wandb.log(log_dict)

    return best_gen_acc, best_recon_acc, euler_steps[gen_euler_idx], euler_steps[recon_euler_idx]
    

def train_flow(flow, model, train_loader, train_eval_loader, valid_eval_loader, opt, lr_scheduler, logger, device,
               n_train_steps, euler_steps, dt, valid_period, n_valid_steps, save_period, save_dir, start_step, best_gen_acc=0, best_recon_acc=0, best_gen_steps=1, best_recon_steps=1, gaudi_trigger=None):
    # create logging variables
    pbar = tqdm(total=n_train_steps, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")

    # initial validation
    if start_step == 0 and not args.skip_first_eval:
        # evaluate(flow, model, train_eval_loader, euler_steps, dt, n_valid_steps, device=device,
        #         logger=logger, tag='_train', global_step=0)
        gen_acc, recon_acc, gen_steps, recon_steps = evaluate(flow, model, valid_eval_loader, euler_steps, dt, n_valid_steps, device=device,
                                      logger=logger, tag='_valid', global_step=0)
        best_gen_acc = gen_acc
        best_recon_acc = recon_acc
        best_gen_steps = gen_steps
        best_recon_steps = recon_steps
    else:
        pbar.update(start_step)

    patience = 0

    # training loop
    train_iterator = get_data_iterator(train_loader, device)
    for train_step in range(start_step, n_train_steps):
        # training step
        train_batch = next(train_iterator)
        p_t, t, v, x, m = train_batch
        p_t, t, v, x = p_t.to(dtype), t.to(dtype), v.to(dtype), x.to(dtype)
        v_pred = flow(p_t, t, x, m)
        train_loss = F.mse_loss(v_pred, v)
        train_loss = train_loss.to(dtype)

        opt.zero_grad()
        train_loss.backward()
        if gaudi_trigger is not None:
            gaudi_trigger()
        opt.step()
        if gaudi_trigger is not None:
            gaudi_trigger()
        if lr_scheduler is not None:
            lr_scheduler.step(train_step)

        logger.add_scalar('_train_loss/loss', train_loss.item(), global_step=train_step)
        logger.add_scalar('_train_loss/lr', opt.param_groups[0]['lr'], global_step=train_step)
        pbar.update()

        # log losses
        pbar.set_description(f'loss={train_loss.item():.8f}')
        
        # validation step
        if (train_step + 1) % valid_period == 0:
            evaluate(flow, model, train_eval_loader, euler_steps, dt, n_valid_steps, device=device,
                     logger=logger, tag='_train', global_step=train_step, gaudi_trigger=gaudi_trigger)
            gen_acc, recon_acc, gen_steps, recon_steps = evaluate(flow, model, valid_eval_loader, euler_steps, dt, n_valid_steps, device=device,
                                          logger=logger, tag='_valid', global_step=train_step, gaudi_trigger=gaudi_trigger)
            
            # save best model
            patience += 1
            
            if gen_acc > best_gen_acc:
                best_gen_acc = gen_acc.item()
                best_gen_steps = gen_steps
                patience = 0
                best_state_dict = {'flow': flow.state_dict(), 'train_step': train_step, 'best_acc': best_gen_acc, 'best_stepsize': best_gen_steps}
                torch.save(best_state_dict, os.path.join(save_dir, 'best_gen.pth'))

            if recon_acc > best_recon_acc:
                best_recon_acc = recon_acc.item()
                best_recon_steps = recon_steps
                patience = 0
                best_state_dict = {'flow': flow.state_dict(), 'train_step': train_step, 'best_acc': best_recon_acc, 'best_stepsize': best_recon_steps}
                torch.save(best_state_dict, os.path.join(save_dir, 'best_recon.pth'))

            if patience > 10:
                print('Early stopping')
                break

        # save checkpoints
        if (train_step + 1) % save_period == 0:
            save_path = os.path.join(save_dir, f'step_{train_step:06d}.pth')
            last_path = os.path.join(save_dir, 'last.pth')
            ckpt = {
                'flow': flow.state_dict(),
                'opt': opt.state_dict(),
                'train_step': train_step,
                'best_gen_acc': best_gen_acc,
                'best_gen_steps': best_gen_steps,
                'best_recon_acc': best_recon_acc,
                'best_recon_steps': best_recon_steps
            }
            torch.save(ckpt, save_path)
            torch.save(ckpt, last_path)

        del train_batch
        gc.collect()

    pbar.close()
    logger.close()

    return best_gen_acc, best_gen_steps



########################################################################################################################
########################################################################################################################
########################################################################################################################



def parse_args(shell_script=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--time_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', '-nw', type=int, default=3)
    parser.add_argument('--base_sources', nargs="+", default=['fungi', 'mscoco', 'wikiart_artist'])
    parser.add_argument('--n_seeds', type=int, default=None)
    parser.add_argument('--n_timesteps', type=int, default=None)
    parser.add_argument('--skip_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=1536)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--n_train_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--euler_steps', type=int, nargs='+', default=[0, 1, 5, 10, 20, 100])
    parser.add_argument('--valid_period', type=int, default=100)
    parser.add_argument('--n_valid_steps', type=int, default=80)
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--exp_root', type=str, default='outputs_metaflow')
    parser.add_argument('--exp_name', type=str, default='metaflow')
    parser.add_argument('--name_postfix', '-ptf', type=str, default='')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--backbone_type', type=str, default='resnet18')
    parser.add_argument('--update_backbone', '-ub', action='store_true', default=False)
    parser.add_argument('--gaudi', action='store_true', default=False)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--piecewise_linear', action='store_true', default=False)
    parser.add_argument('--cubic', action='store_true', default=False)
    parser.add_argument('--use_model', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--ignore_time', action='store_true', default=False)
    parser.add_argument('--regression_mode', action='store_true', default=False)
    parser.add_argument('--dynamic_tasks', action='store_true', default=False)
    parser.add_argument('--skip_first_eval', '-sfe', action='store_true', default=False)
    parser.add_argument('--stochastic', action='store_true', default=False)
    parser.add_argument('--random_couplings', action='store_true', default=False)
    parser.add_argument('--random_segment', action='store_true', default=False)
    parser.add_argument('--random_cubic', action='store_true', default=False)
    parser.add_argument('--params_tag', type=str, default='huge_fe_even_q_bias_train_40steps_0.1lr_1seeds_0.2noise')
    
    # parser.add_argument('--model_path', type=str, default='/common_datasets/METAFLOW_DATASETS/caml_pretrained_models/CAML_Laion2b')
    # parser.add_argument('--fe_type', type=str, default="cache:timm:vit_huge_patch14_clip_224.laion2b:1280")
    
    parser.add_argument('--fe_dtype', type=str, default='bfloat16')
    parser.add_argument('--n_tasks_train', type=int, default=500)
    parser.add_argument('--n_tasks_valid', type=int, default=10)
    
    parser.add_argument('--model_size', type=str, default='huge')
    parser.add_argument('--model', type=str, default='CAML')
    parser.add_argument('--use_cache', action='store_true', default=False)

    if shell_script is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shell_script.split())

    return args

if __name__ == '__main__':
    args = parse_args()

    
    ############################################################
    # setting device, dtype and seed
    device = torch.device('cuda')
    dtype = torch.bfloat16
    set_seed(42)
    if args.gaudi:
        args.device = 'hpu'
        import habana_frameworks.torch.core as htcore # type: ignore
        gaudi_trigger = htcore.mark_step
    else:
        gaudi_trigger = None
    print(f"device: {device}, dtype: {dtype}")
    

    
    ############################################################
    # load params
    def load_params(split, root='outputs'):
        params_folder_name = f"ft_trajs_{args.params_tag}"
        if split=='val':
            params_folder_name = params_folder_name.replace("train", "val")
        params_folder_path = os.path.join(root, params_folder_name)
        params = []
        for domain in args.base_sources:
            params_domain_path = os.path.join(params_folder_path, domain)
            n_tasks = args.n_tasks_train if split=='train' else args.n_tasks_valid
            for task_idx in range(n_tasks):
                file_name = f'ckpt_{task_idx:06d}.pth'
                param = torch.load(os.path.join(params_domain_path, file_name))
                params.append(param)
        params = torch.cat(params)
        params = params.permute(0, 3, 1, 2, 4)
        
        return params

    # (nT, nS, T+1, nB, d)
    params_train = load_params('train')
    params_valid = load_params('val')
    
    # args.stochastic = not args.deterministic
    # if args.stochastic:
    #     params_train_dir = os.path.join(args.model_path, f'ft_trajs_bias_qkv_train_{args.params_tag}_20w_10s')
    #     params_valid_dir = os.path.join(args.model_path, f'ft_trajs_bias_qkv_valid_{args.params_tag}_20w_10s')
    # else:
    #     params_tag = '_'.join(args.params_tag.split('_')[:2] + ['1seed'])
    #     params_train_dir = os.path.join(args.model_path, f'ft_trajs_bias_qkv_train_{params_tag}_20w_10s')
    #     params_valid_dir = os.path.join(args.model_path, f'ft_trajs_bias_qkv_valid_{params_tag}_20w_10s')
    
    # if args.dynamic_tasks:
    #     params_train_dir += '_dynamic'
    #     params_valid_dir += '_dynamic'

    # if args.piecewise_linear or args.cubic or args.random_segment or args.random_cubic:
    #     params_train_name = 'merged_500tasks.pth'
    #     params_valid_name = 'merged_10tasks.pth'
    # else:
    #     params_train_name = 'merged_500tasks_endonly.pth'
    #     params_valid_name = 'merged_10tasks_endonly.pth'

    # params_train = torch.load(os.path.join(params_train_dir, params_train_name)) 
    # params_valid = torch.load(os.path.join(params_valid_dir, params_valid_name))
    
    with torch.no_grad():
        if args.n_seeds is not None:
            params_train = params_train[:, :args.n_seeds]
            params_valid = params_valid[:, :args.n_seeds]
        if args.n_timesteps is not None:
            params_train = params_train[:, :, :args.n_timesteps+1]
            params_valid = params_valid[:, :, :args.n_timesteps+1]
        if args.skip_steps is not None:
            params_train = params_train[:, :, ::args.skip_steps]
            params_valid = params_valid[:, :, ::args.skip_steps]
    gc.collect()
    
    _, _, n_timesteps, n_modules, n_params = params_train.shape
    print(f'Loaded train params with shape {params_train.shape}')
    print(f'Loaded valid params with shape {params_valid.shape}')
    
    exit()
    
    if args.regression_mode:
        args.ignore_time = True
        args.euler_steps = [0, 1]
        dt = 1 / (n_timesteps - 1)
    else:
        n_pseudo_timesteps = 51
        args.euler_steps = [i for i in range(0, n_pseudo_timesteps, 10)] + [1, 5]
        dt = 1 / (n_pseudo_timesteps - 1)



    ############################################################
    # create exp directory
    args.exp_name = args.exp_name + args.name_postfix + '_' + args.params_tag + ':' + str(args.hidden_dim)
    os.makedirs(args.exp_root, exist_ok=True)
    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"exp directory: {exp_dir}")
    
    

    ############################################################
    # model_size
    if args.model_size == 'huge':
        args.fe_type = 'timm:vit_huge_patch14_clip_224.laion2b:1280'
        args.model_path = '/common_datasets/METAFLOW_DATASETS/caml_pretrained_models/CAML_Laion2b'
        encoder_size = 'laion'
    elif args.model_size == 'base':
        args.fe_type = 'timm:vit_base_patch16_clip_224.openai:768'
        args.model_path = '/common_datasets/METAFLOW_DATASETS/caml_pretrained_models/CAML_CLIP'
        encoder_size = 'large'
    
    
    
    ############################################################
    # create model
    print(f"Creating model: CAML with {args.fe_type}")
    
    fe_metadata = get_fe_metadata(args)
    
    def load_model(model_size, fe_metadata):
        if model_size == 'huge':
            patch_size = 14
            embed_dim = 1280
            depth = 32
            num_heads = 16
        elif model_size == 'base':
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
        
        feature_extractor = VisionTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            pre_norm=True,
            norm_layer=nn.LayerNorm,
            num_classes=0,
            n_bias=1,
            weight_init='skip'
            ).to(device=device, dtype=dtype)
        
        fe_dict = fe_metadata['fe'].state_dict()
        for b in range(depth):
            for i, c in enumerate(['q', 'k', 'v']):
                fe_dict[f'blocks.{b}.attn.qkv_bias.{c}_bias'] = fe_dict[f'blocks.{b}.attn.qkv.bias'][i*embed_dim:(i+1)*embed_dim].unsqueeze(0).repeat_interleave(1, dim=0)
            del fe_dict[f'blocks.{b}.attn.qkv.bias']
        feature_extractor.load_state_dict(fe_dict, strict=True)
        
        model = CAML(
            feature_extractor=feature_extractor,
            fe_dim=fe_metadata['fe_dim'],
            fe_dtype=dtype,
            train_fe=False, # whether to update the feature encoder weights during meta-training
            encoder_size=encoder_size,
            num_bias=1,
            dropout=0.0,
            label_elmes=True,
            device=device,
            set_transformer=False
        )
        
        model_path = os.path.join(args.model_path, 'model.pth')
        model_dict = torch.load(model_path, map_location='cuda')
        for b in range(24):
            for i, c in enumerate(['q', 'k', 'v']):
                model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{b}.self_attention.in_proj_bitfit_bias.{c}_bias'] = \
                    model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{b}.self_attention.in_proj_bias'][i*(embed_dim+256):(i+1)*(embed_dim+256)].unsqueeze(0).repeat_interleave(1, dim=0)
        model_dict["transformer_encoder.elmes_scale"] = model_dict['transformer_encoder.etf_scale']
        model_dict["transformer_encoder.label_elmes"] = model_dict["transformer_encoder.label_etf"]
        del model_dict['transformer_encoder.label_etf']
        del model_dict['transformer_encoder.etf_scale']
        model.load_state_dict(model_dict, strict=False)
        model.to(device=device, dtype=dtype)
        model.eval()
        print(f"checkpoint loaded: {model_path}")
        
        return model
    
    model = load_model(args.model_size, fe_metadata)
    
    # load checkpoint
    # model_path = os.path.join(args.model_path, 'model.pth')
    # model_dict = torch.load(model_path, map_location='cuda')
    # for i in range(24):
    #     bias = model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias'].unsqueeze(0)
    #     d = bias.shape[1] // 3
    #     model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.q_bias'] = bias[:,:d]
    #     model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.k_bias'] = bias[:,d:2*d]
    #     model_dict[f'transformer_encoder.encoder.layers.encoder_layer_{i}.self_attention.in_proj_bitfit_bias.v_bias'] = bias[:,2*d:]
    # model_dict["transformer_encoder.elmes_scale"] = model_dict['transformer_encoder.etf_scale']
    # model_dict["transformer_encoder.label_elmes"] = model_dict["transformer_encoder.label_etf"]
    # del model_dict['transformer_encoder.label_etf']
    # del model_dict['transformer_encoder.etf_scale']
    # model.load_state_dict(model_dict, strict=True)
    # model.to(device=device, dtype=dtype)
    # model.eval()
    # print(f"checkpoint loaded: {model_path}")
    
    # create flow model
    backbone = None
    if args.use_model:
        if args.update_backbone:
            backbone = deepcopy(model.backbone)
        else:
            backbone = model.backbone
    flow = Flow(n_params, args.hidden_dim, args.n_layers, args.n_blocks, n_modules, use_cache=args.use_cache, backbone_type=args.backbone_type,
                freeze_backbone=(not args.update_backbone), backbone=backbone, ignore_time=args.ignore_time, gaudi=args.gaudi).to(device, dtype=dtype)

    
    
    ############################################################
    # create support dataset
    train_transform = fe_metadata['train_transform']
    support_dataset_train = AggregatedDataset(args.base_sources, 'train', args.n_tasks_train, transform=train_transform)
    support_dataset_train_eval = AggregatedDataset(args.base_sources, 'train', args.n_tasks_train, transform=train_transform, fix_seed=True)
    support_dataset_valid_eval = AggregatedDataset(args.base_sources, 'val', args.n_tasks_valid, transform=train_transform, fix_seed=True)
    
    train_eval_dataset = FlowDataset(support_dataset_train_eval, params_train, args.n_tasks_train, eval_mode=True, stochastic=args.stochastic)
    valid_eval_dataset = FlowDataset(support_dataset_valid_eval, params_valid, args.n_tasks_valid, eval_mode=True, stochastic=args.stochastic)
    train_dataset = FlowDataset(support_dataset_train, params_train, args.n_tasks_train, time_batch_size=args.time_batch_size,
                                noise=args.noise, piecewise_linear=args.piecewise_linear, cubic=args.cubic,
                                stochastic=args.stochastic, regression_mode=args.regression_mode, random_couplings=args.random_couplings,
                                random_segment=args.random_segment, random_cubic=args.random_cubic)
    
    train_loader = get_data_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, collate_fn=train_collate_fn)
    train_eval_loader = get_data_loader(train_eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=False)
    valid_eval_loader = get_data_loader(valid_eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=False)
    
    print("Dataset loaded")
    
    

    ############################################################
    # optimizer, scheduler
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    # cfg = EasyDict({'sched': 'cosine', 'epochs': args.n_train_steps, 'warmup_epochs': 100})
    # lr_scheduler, _ = create_scheduler(cfg, opt)
    lr_scheduler = None

    

    ############################################################
    # Training Part
    # resume training
    start_step = 0
    best_gen_acc = 0
    best_gen_steps = 1
    best_recon_acc = 0
    best_recon_steps = 1
    if args.resume:
        last_path = os.path.join(exp_dir, 'checkpoints', 'last.pth')
        ckpt = torch.load(last_path, map_location='cpu')
        flow.load_state_dict(ckpt['flow'])
        opt.load_state_dict(ckpt['opt'])
        start_step = ckpt['train_step'] + 1
        best_gen_acc = ckpt['best_gen_acc']
        best_gen_steps = ckpt['best_gen_steps']
        best_recon_acc = ckpt['best_recon_acc']
        best_recon_steps = ckpt['best_recon_steps']
        print_log = f'Resuming from {last_path} at step {start_step}' + \
                    f', best gen acc {best_gen_acc:.2f} with {best_gen_steps} steps' + \
                    f' and best recon acc {best_recon_acc:.2f} with {best_recon_steps} steps.'
        print(print_log)
    
    # create logger
    log_dir = os.path.join(exp_dir, 'logs')
    if not args.resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    
    # create checkpoints directory
    save_dir = os.path.join(exp_dir, 'checkpoints')
    if not args.resume:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    # set seed for training
    set_seed(42)

    # main loop
    best_acc = train_flow(
        flow, model, train_loader, train_eval_loader, valid_eval_loader, opt, lr_scheduler, logger, device,
        args.n_train_steps, args.euler_steps, dt, args.valid_period, args.n_valid_steps, args.save_period, save_dir,
        start_step, best_gen_acc, best_recon_acc, best_gen_steps, best_recon_steps, gaudi_trigger=gaudi_trigger
    )

    # wandb.log({"best_acc": best_acc})
    print(f"best_acc: {best_acc}")
