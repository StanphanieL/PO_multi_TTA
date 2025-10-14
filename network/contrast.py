import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d


class POContrast(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, proj_dim: int = 128, arch: str = 'MinkUNet34C'):
        super().__init__()
        self.backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch=arch)
        # projection head: C -> 128 -> proj_dim
        self.proj = nn.Sequential(
            nn.Linear(out_channels, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim, bias=True),
        )
        self.global_pool = ME.MinkowskiGlobalAvgPooling()

    def forward_embed(self, feat_voxel, xyz_voxel):
        # build sparse tensor and extract features
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f'cuda:{cuda_cur_device}')
        voxel_feat = self.backbone(inputs)  # SparseTensor [#vox, C]
        pooled = self.global_pool(voxel_feat)  # SparseTensor [B, C]
        x = pooled.F  # [B, C]
        z = self.proj(x)
        z = F.normalize(z, dim=1)
        return z


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss from: https://arxiv.org/abs/2004.11362
    Input:
        features: [B, V, D] where V is number of views
        labels: [B] with class indices
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B, V, D = features.shape
        feats = F.normalize(features.view(B * V, D), dim=1)  # [B*V, D]
        labels = labels.view(-1, 1)  # [B, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
        mask = mask.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1)  # [B*V, B*V]
        logits = torch.div(torch.matmul(feats, feats.T), self.temperature)  # [B*V, B*V]
        # mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(B * V, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        # mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - mean_log_prob_pos
        loss = loss.view(B, V).mean()
        return loss


class Prototypes(nn.Module):
    def __init__(self, num_classes: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.m = momentum
        self.register_buffer('proto', torch.zeros(num_classes, dim))
        self.register_buffer('counts', torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor):
        # z: [N, D], y: [N]
        if z.numel() == 0:
            return
        y = y.view(-1)
        for cls in y.unique():
            cls = cls.item()
            idx = (y == cls)
            if idx.sum() == 0:
                continue
            z_mean = F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1)
            old = self.proto[int(cls):int(cls)+1]
            new = F.normalize(self.m * old + (1 - self.m) * z_mean, dim=1)
            self.proto[int(cls):int(cls)+1] = new
            self.counts[int(cls)] = self.counts[int(cls)] + idx.sum()


class ProtoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # z: [N, D] normalized, prototypes: [K, D] normalized
        logits = torch.matmul(z, prototypes.T) / self.temperature  # [N, K]
        loss = F.cross_entropy(logits, y)
        return loss
