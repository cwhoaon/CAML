import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import math
from pmf_models.vision_transformer import CrossBlock, Block
from pmf_models import vision_transformer as vit


def unnormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return x*std + mean


def train_collate_fn(batch):
    p_t, t, v, x, m = zip(*batch)
    p_t = torch.stack(p_t, dim=0)
    t = torch.stack(t, dim=0)
    v = torch.stack(v, dim=0)
    
    x = torch.stack(x)
    m = torch.stack(m)
    # max_ways = max([m_i.shape[0] for m_i in m])
    # max_shot = max([m_i.shape[1] for m_i in m])
    # x = torch.stack([F.pad(x_i, (0, 0, 0, 0, 0, 0, 0, max_shot - x_i.shape[1], 0, max_ways - x_i.shape[0]), value=0) for x_i in x], dim=0)
    # m = torch.stack([F.pad(m_i, (0, max_shot - m_i.shape[1], 0, max_ways - m_i.shape[0]), value=False) for m_i in m], dim=0)

    return p_t, t, v, x, m


class FlowDataset(Dataset):
    def __init__(self, support_dataset, params, n_tasks=500,
                 max_ways=10, max_shot=5, time_batch_size=8, eval_mode=False, noise=0,
                 piecewise_linear=False, cubic=False, split=None, stochastic=False, 
                 regression_mode=False, random_couplings=False, random_segment=False, random_cubic=False):
        self.support_dataset = support_dataset
        self.datasets = self.support_dataset.domains
        self.params = params
        self.piecewise_linear = piecewise_linear
        self.cubic = cubic
        self.max_ways = max_ways
        self.max_shot = max_shot
        self.time_batch_size = time_batch_size
        self.eval_mode = eval_mode
        self.noise = noise
        self.stochastic = stochastic
        self.regression_mode = regression_mode
        self.random_couplings = random_couplings
        self.random_segment = random_segment
        self.random_cubic = random_cubic
        if self.random_couplings:
            assert self.stochastic, "random_couplings should be used with stochastic=True"

        assert self.params.ndim == 5, f"params should be 5D tensor (n_tasks, n_seeds, n_timesteps, n_blocks, dim), given shape is {self.params.shape}"

        self.split = split
        if split is None:
            self.total_tasks_per_dataset = self.n_tasks_per_dataset = n_tasks
        else:
            valid_ratio = 0.1
            self.total_tasks_per_dataset = n_tasks
            self.train_tasks_per_dataset = int(n_tasks * (1 - valid_ratio))
            self.valid_tasks_per_dataset = n_tasks - self.train_tasks_per_dataset
            if self.split == 'train':
                self.n_tasks_per_dataset = self.train_tasks_per_dataset
            else:
                self.n_tasks_per_dataset = self.valid_tasks_per_dataset
        self.n_tasks = self.n_tasks_per_dataset * len(self.datasets)
        self.n_support_tasks_per_dataset = self.support_dataset.n_tasks // len(self.datasets)

    def __len__(self):
        return self.n_tasks
    
    def parse_index(self, idx):
        # parse index
        if self.eval_mode and self.split is None:
            dataset_idx = idx % len(self.datasets) # uniformly iterate datasets
            task_idx = idx // len(self.datasets)
        else:
            dataset_idx = idx // self.n_tasks_per_dataset
            task_idx = idx % self.n_tasks_per_dataset
            if self.split == 'valid':
                task_idx += self.train_tasks_per_dataset # adjust task index for validation set

        # task index for params
        p_idx = task_idx + dataset_idx*self.total_tasks_per_dataset

        # task index for data
        d_idx = task_idx + dataset_idx*self.n_support_tasks_per_dataset

        return p_idx, d_idx

    def __getitem__(self, idx):
        # parse index
        p_idx, d_idx = self.parse_index(idx)

        # sample parameters
        if self.stochastic and not self.eval_mode:
            seed = torch.randint(0, self.params.shape[1], (1,)).item()
        else:
            seed = 0

        p = self.params[p_idx, seed]
        
        if self.eval_mode:
            p_init = p[0]
            p_target = p[-1]
            x_s, y_s, x_q, y_q, *_ = self.support_dataset.sample_episode(d_idx)

            return p_init, p_target, x_s, y_s, x_q, y_q
        
        else:        
            # sample support data
            x, m = self.support_dataset.sample_support(d_idx)
            
            # sample timesteps
            if self.regression_mode:
                t = torch.zeros((self.time_batch_size, 1, 1), dtype=x.dtype)
            else:
                t = torch.rand((self.time_batch_size, 1, 1), dtype=x.dtype) # (B, 1, 1), range [0, 1)

            # sample parameters
            if self.random_couplings:
                seed2 = torch.randint(0, self.params.shape[1], (1,)).item()
                p2 = self.params[p_idx][seed2]
            else:
                p2 = None
            
            if self.cubic:
                p_t, v = self.get_cubic_targets(p, t)
            elif self.random_cubic:
                p_t, v, t = self.get_random_cubic_targets(p, t)
            elif self.random_segment:
                p_t, v, t = self.get_random_segment_targets(p, t, p2=p2)
            else:
                p_t, v = self.get_linear_targets(p, t, p2=p2)
                
            return p_t, t, v, x, m
    
    def get_linear_targets(self, p, t, p2=None):
        if p2 is None:
            p2 = p

        n_timesteps = len(p) # == T + 1
        t_init = (t[:, 0, 0]*(n_timesteps - 1)).floor().long() # (B), range [0, T-1]
        p_init = p[t_init] # (B, m, d)
        p_target = p2[t_init + 1] # (B, m, d)

        p_t = t * p_target + (1 - t) * p_init # (B, m, d)
        v = p_target - p_init
        v = v * (n_timesteps - 1) # (B, m, d)

        return p_t, v
    
    def get_random_segment_targets(self, p, t, p2=None):
        n_timesteps = len(p) # == T + 1
        start = torch.randint(0, len(p) - 1, (1,)).item()
        end = torch.randint(start + 1, len(p), (1,)).item()
        p = torch.cat([p[start:start+1], p[end:end+1]], dim=0) # (2, m, d)
        if p2 is not None:
            p2 = torch.cat([p2[start:start+1], p2[end:end+1]], dim=0)
        p_t, v = self.get_linear_targets(p, t, p2=p2)
        v = v * (n_timesteps - 1) / (end - start)
        t = (start + t * (end - start)) / (n_timesteps - 1)
        return p_t, v, t
    
    def get_cubic_targets(self, p, t):
        n_timesteps = len(p) # == T + 1
        t_init = (t[:, 0, 0]*(n_timesteps - 1)).floor().long() # (B), range [0, T-1]
        p = torch.cat([p[:1], p, p[-1:]], dim=0) # (T+3, m, d)

        p0 = p[t_init]
        p1 = p[t_init + 1] # (B, m, d)
        p2 = p[t_init + 2] # (B, m, d)
        p3 = p[t_init + 3]
        t1 = t * (n_timesteps - 1) - t_init[:, None, None].to(dtype=t.dtype) # (B, m, d), range [0, 1)

        t2 = t1 * t1
        t3 = t2 * t1
        a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
        a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
        a2 = -0.5 * p0 + 0.5 * p2
        a3 = p1

        p_t = a0 * t3 + a1 * t2 + a2 * t1 + a3
        v = 3 * a0 * t2 + 2 * a1 * t1 + a2
        v = v * (n_timesteps - 1) # (B, m, d)

        return p_t, v
    
    def get_random_cubic_targets(self, p, t):
        n_timesteps = len(p)
        sample_indices = sorted(torch.randperm(n_timesteps)[:4].numpy())

        t_samples = torch.tensor(sample_indices, dtype=t.dtype) / (n_timesteps - 1) # (4)
        p_samples = p[sample_indices] # (4, m, d)
        p0, p1, p2, p3 = p_samples

        t1 = t
        t2 = t1 * t1
        t3 = t2 * t1
        a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
        a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
        a2 = -0.5 * p0 + 0.5 * p2
        a3 = p1

        p_t = a0 * t3 + a1 * t2 + a2 * t1 + a3
        v = 3 * a0 * t2 + 2 * a1 * t1 + a2
        v = v * (n_timesteps - 1) / (sample_indices[-1] - sample_indices[0])

        t = t_samples[0] + t * (t_samples[1] - t_samples[0])
        return p_t, v, t



class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
    

class SupportEncoder(nn.Module):
    def __init__(self, hidden_dim=384, n_blocks=2, n_heads=8, use_cache=False, backbone_type='resnet18', freeze_backbone=True, backbone=None, gaudi=False):
        super().__init__()
        self.use_cache = use_cache
        
        if self.use_cache:
            feature_dim = 1280
        else:
            if backbone is not None:
                self.backbone = backbone
                feature_dim = 384
            else:
                if backbone_type == 'resnet18':
                    self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                    self.backbone.fc = nn.Identity()
                    feature_dim = 512
                elif backbone_type == 'resnet50':
                    self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                    self.backbone.fc = nn.Identity()
                    feature_dim = 2048
                elif backbone_type == 'dino_small_patch16':
                    self.backbone = vit.__dict__['vit_small'](patch_size=16, num_classes=0, gaudi=gaudi)
                    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                    self.backbone.load_state_dict(state_dict, strict=True)
                    feature_dim = 384
                else:
                    raise ValueError(f"backbone_type {backbone_type} is not supported")
            self.freeze_backbone = freeze_backbone
        
        
        self.task_projector = nn.Linear(feature_dim, hidden_dim)

        if n_blocks == 0:
            self.task_encoder = Identity()
        else:
            self.task_prompts = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.task_encoder = nn.ModuleList([Block(hidden_dim, n_heads) for _ in range(n_blocks)])

    def learnable_parameters(self):
        if self.freeze_backbone:
            for p in self.task_projector.parameters():
                yield p
            for p in self.task_encoder.parameters():
                yield p
        else:
            return self.parameters()

    def forward(self, x, m=None, y=None):
        '''
        get
            x: (B, N, K, C, H, W), torch.float32
                - N-way K-shot support images
                - B: batch_size, N: ways, K: shots C: n_channels
            m: (B, N, K), torch.bool
                - validity mask for x
        or
            x: (B, M, C, H, W), torch.float32
                - unknwon way/shot support images
                - B: batch_size, M: support size C: n_channels
            y: (B, N), torch.long
                - support labels
        return 
            z: (B, d), torch.float32
                - support embedding
                - d: hidden_dim
        '''
        if self.use_cache:
            assert x.ndim == 3 or x.ndim == 4, f"x should be 5D or 6D tensor, given shape is {x.shape}"
            if x.ndim == 3:
                assert y is not None
                B, M = x.shape[:2]
                assert B == 1
                num_classes = y.max() + 1 # NOTE: assume B==1
                m = None
            elif x.ndim == 4:
                y = None
            z = x
            # average for shots
            if y is not None:
                y_onehot = F.one_hot(y, num_classes).to(dtype=z.dtype).transpose(1, 2) # B, N, M
                z = torch.bmm(y_onehot, z) # B, N, d
                z = z / y_onehot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
                m_ways = None
            else:
                if m is not None:
                    m_shot = m[..., None].to(dtype=z.dtype) # (B, N, K, 1)
                z = (z*m_shot).sum(dim=2) / torch.where(m[:, :, :1], m_shot.sum(dim=2), torch.ones_like(m[:, :, :1])) # B, N, d
                if m is not None:
                    m_ways = m[:, :, 0] # B, N
        else:
            assert x.ndim == 5 or x.ndim == 6, f"x should be 5D or 6D tensor, given shape is {x.shape}"
            if x.ndim == 5:
                assert y is not None
                B, M = x.shape[:2]
                assert B == 1
                num_classes = y.max() + 1 # NOTE: assume B==1
                x = rearrange(x, 'B M C H W -> (B M) C H W')
                m = None
            else:
                B, N, K = x.shape[:3]
                x = rearrange(x, 'B N K ... -> (B N K) ...')
                y = None
                
            # per-image encoding
            if self.freeze_backbone:
                self.backbone.eval()
                with torch.no_grad():
                    z = self.backbone(x)
            else:
                z = self.backbone(x)
                
            # average for shots
            if y is not None:
                z = rearrange(z, '(B M) d -> B M d', B=B, M=M)
                y_onehot = F.one_hot(y, num_classes).to(dtype=z.dtype).transpose(1, 2) # B, N, M
                z = torch.bmm(y_onehot, z) # B, N, d
                z = z / y_onehot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
                m_ways = None
            else:
                if m is not None:
                    m_shot = m[..., None].to(dtype=z.dtype) # (B, N, K, 1)
                z = rearrange(z, '(B N K) d -> B N K d', B=B, N=N, K=K)
                z = (z*m_shot).sum(dim=2) / torch.where(m[:, :, :1], m_shot.sum(dim=2), torch.ones_like(m[:, :, :1])) # B, N, d
                if m is not None:
                    m_ways = m[:, :, 0] # B, N

        # task encoding
        z = self.task_projector(z)
        z = torch.cat([self.task_prompts.repeat(z.size(0), 1, 1), z], dim=1)
        if m is not None:
            m_ways = F.pad(m_ways, (1, 0), value=True)

        for blk in self.task_encoder:
            z = blk(z, mask=m_ways)
        z = z[:, 0]

        return z


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        nn.init.ones_(self.weight)

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = x * torch.rsqrt(var)
        rmsnorm = self.weight * input_norm
        return rmsnorm


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.silu(gate)


class Drift(nn.Module):
    def __init__(self, n_params=1152, hidden_dim=384, n_layers=3, n_modules=12, ignore_time=False):
        super().__init__()
        self.ignore_time = ignore_time
        self.n_modules = n_modules
        self.support_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.params_embedding = nn.Linear(n_params*n_modules, hidden_dim)
        if not self.ignore_time:
            self.time_embedding = nn.Linear(1, hidden_dim)

        residual_blocks = []
        for _ in range(n_layers):
            residual_blocks.append(nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                SwiGLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.output_layer = nn.Linear(hidden_dim, n_params*n_modules)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

    def forward(self, p, z, t, loss=None):
        '''
        get
            p: (B, m, n)
                - input parameters
                - B: batch_size, m: n_modules, n: n_params
            z: (B, d)
                - support embedding of target task
                - d: hidden_dim
            t: (B, 1, 1)
                - time variable
        return
            d: (B, m, n)
                - drift direction to target task at time t
        '''
        z = self.support_embedding(z)
        p = self.params_embedding(rearrange(p, 'B m n -> B (m n)'))
        x = p + z
        if not self.ignore_time:
            t = self.time_embedding(t.squeeze(1) - 0.5)
            x = x + t

        for residual_block in self.residual_blocks:
            x = x + residual_block(x)
        x = self.output_layer(x)

        x = rearrange(x, 'B (m n) -> B m n', m=self.n_modules)

        return x
        

class Flow(nn.Module):
    def __init__(self, n_params=1152, hidden_dim=384, n_layers=4, n_blocks=2, n_modules=12,
                 use_cache=False, backbone_type='resnet18', freeze_backbone=True, backbone=None, ignore_time=False, gaudi=False):
        super().__init__()
        self.drift = Drift(n_params, hidden_dim, n_layers, n_modules=n_modules, ignore_time=ignore_time)
        self.support_encoder = SupportEncoder(hidden_dim, n_blocks, use_cache=use_cache, backbone_type=backbone_type,
                                              freeze_backbone=freeze_backbone, backbone=backbone, gaudi=gaudi)

    def forward(self, p_t, t, x, m=None, y=None, loss=None):
        # encode support images
        z = self.support_encoder(x, m, y)
    
        # merge batch-dimension and time-batch dimension
        if p_t.ndim == 4:
            T = p_t.shape[1]
            p_t = rearrange(p_t, 'B T m n -> (B T) m n')
            t = rearrange(t, 'B T 1 1 -> (B T) 1 1')
            z = repeat(z, 'B n -> (B T) n', T=T)
            if loss is not None:
                loss = rearrange(loss, 'B T 1 1 -> (B T) 1 1')
        else:
            T = 0

        # encode drift
        v_pred = self.drift(p_t, z, t, loss)

        if T > 0:
            v_pred = rearrange(v_pred, '(B T) m n -> B T m n', T=T)

        return v_pred

    def inference(self, p_init, x, m=None, y=None, euler_step=100, get_traj=False, dt=None, method='euler'):
        dt = 1./euler_step if dt is None else dt
        p = p_init
    
        if get_traj:
            traj = []
        z = self.support_encoder(x, m, y)
        for i in range(euler_step):
            t = torch.ones((len(p_init), 1, 1), dtype=p_init.dtype, device=p_init.device) * i / euler_step

            if method == 'euler':
                v_pred = self.drift(p, z, t)
                p = p + v_pred * dt
            elif method == 'rk4':
                k1 = self.drift(p, z, t)
                k2 = self.drift(p + dt/2 * k1, z, t + dt/2)
                k3 = self.drift(p + dt/2 * k2, z, t + dt/2)
                k4 = self.drift(p + dt * k3, z, t + dt)
                p = p + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            elif method == 'rk2':
                k1 = self.drift(p, z, t)
                k2 = self.drift(p + dt/2 * k1, z, t + dt/2)
                p = p + dt * k2
            elif method == 'heun':
                k1 = self.drift(p, z, t)
                p_predict = p + dt * k1
                k2 = self.drift(p_predict, z, t + dt)
                p = p + dt/2 * (k1 + k2)
            else:
                raise ValueError(f"method {method} is not supported")

            if get_traj:
                traj.append(p)
        
        if get_traj:
            traj = torch.stack(traj)
            return traj
        else:
            return p