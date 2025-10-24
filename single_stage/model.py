import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d


class SingleStageNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int = 0, class_embed_dim: int = 32, conditional_mode: str = 'film', proj_dim: int = 128):
        super().__init__()
        self.backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch='MinkUNet34C')
        # conditional offset head (same as PONet)
        self.num_classes = num_classes
        self.conditional_mode = conditional_mode
        self.class_embed_dim = class_embed_dim if (num_classes and class_embed_dim) else 0
        if self.class_embed_dim > 0:
            self.class_embed = nn.Embedding(self.num_classes, self.class_embed_dim)
            self.film = nn.Linear(self.class_embed_dim, out_channels * 2) if self.conditional_mode == 'film' else None
        else:
            self.class_embed, self.film = None, None
        feat_in = out_channels if (self.film is not None) else (out_channels + (self.class_embed_dim if self.class_embed is not None else 0))
        self.linear_offset = nn.Sequential(
            nn.Linear(feat_in, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 3, bias=True)
        )
        # projection head (like POContrast)
        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.proj = nn.Sequential(
            nn.Linear(out_channels, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim, bias=True),
        )

    def forward_offset(self, feat_voxel, xyz_voxel, v2p_v1, batch_count=None, category_ids=None):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f'cuda:{cuda_cur_device}')
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        if self.class_embed is not None and batch_count is not None and category_ids is not None:
            B = category_ids.shape[0]
            cond_list = []
            cat_emb = self.class_embed(category_ids.cuda())
            for b in range(B):
                start = int(batch_count[b].item()); end = int(batch_count[b+1].item()); n = end - start
                cond_list.append(cat_emb[b:b+1].repeat(n, 1))
            cond = torch.cat(cond_list, dim=0)
            if self.film is not None and self.conditional_mode == 'film':
                gamma_beta = self.film(cond)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                point_feat = point_feat * (1 + gamma) + beta
            else:
                point_feat = torch.cat([point_feat, cond], dim=1)
        return self.linear_offset(point_feat)

    def forward_embed(self, feat_voxel, xyz_voxel):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device=f'cuda:{cuda_cur_device}')
        voxel_feat = self.backbone(inputs)
        pooled = self.global_pool(voxel_feat)
        x = pooled.F
        z = self.proj(x)
        return F.normalize(z, dim=1)
