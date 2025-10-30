import torch
import torch.nn as nn
import MinkowskiEngine as ME
from network.Mink import Mink_unet as unet3d
import numpy as np
import open3d as o3d


class PONet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes: int = 0, class_embed_dim: int = 0, conditional_mode: str = 'concat'):
        super(PONet, self).__init__()
        self.backbone = unet3d(in_channels=in_channels, out_channels=out_channels, arch='MinkUNet34C')
        self.num_classes = num_classes
        self.conditional_mode = conditional_mode #类别信息与特征结合的方式
        self.class_embed_dim = class_embed_dim if (num_classes is not None and num_classes > 0 and class_embed_dim is not None and class_embed_dim > 0) else 0
        if self.class_embed_dim > 0:
            self.class_embed = nn.Embedding(self.num_classes, self.class_embed_dim) #初始化类别嵌入层，将类别ID映射为嵌入向量
            if self.conditional_mode == 'film':
                self.film = nn.Linear(self.class_embed_dim, out_channels * 2) #初始化FiLM，用于生成类别条件的缩放和偏移
            else:
                self.film = None
        else:
            self.class_embed = None
            self.film = None
        feat_in = out_channels if (self.film is not None) else (out_channels + (self.class_embed_dim if self.class_embed is not None else 0)) #根据是否使用FiLM，确定输入特征的维度
        self.linear_offset = nn.Sequential(
            nn.Linear(feat_in, out_channels, bias=False),                                                                                                                                                            
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 3, bias=True)
        )

        self.weight_initialization()
    
    def weight_initialization(self): # 针对MinkowskiEngine库中的特殊层类型进行了定制化的初始化，确保网络在训练开始时有一个良好的起点
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self,  feat_voxel, xyz_voxel, v2p_v1, batch_count=None, category_ids=None):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        if self.class_embed is not None and batch_count is not None and category_ids is not None: #类别条件融合
            # build per-point conditional embedding
            B = category_ids.shape[0]
            cond_list = []
            cat_emb = self.class_embed(category_ids.cuda())  # [B, D] 将类别ID转换为嵌入向量
            for b in range(B):
                start = int(batch_count[b].item())
                end = int(batch_count[b+1].item())
                n = end - start
                cond_list.append(cat_emb[b:b+1].repeat(n, 1)) #为每个样本的所有点重复其类别嵌入向量
            cond = torch.cat(cond_list, dim=0)
            if self.film is not None and self.conditional_mode == 'film': #条件特征融合
                gamma_beta = self.film(cond)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                point_feat = point_feat * (1 + gamma) + beta
            else:
                point_feat = torch.cat([point_feat, cond], dim=1) #concat模式拼接
        pred_offset = self.linear_offset(point_feat)

        return pred_offset

    def test_inference(self, feat_voxel, xyz_voxel, v2p_v1, batch_count=None, category_ids=None):
        cuda_cur_device = torch.cuda.current_device()
        inputs = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        if self.class_embed is not None and batch_count is not None and category_ids is not None:
            B = category_ids.shape[0]
            cond_list = []
            cat_emb = self.class_embed(category_ids.cuda())
            for b in range(B):
                start = int(batch_count[b].item())
                end = int(batch_count[b+1].item())
                n = end - start
                cond_list.append(cat_emb[b:b+1].repeat(n, 1))
            cond = torch.cat(cond_list, dim=0)
            if self.film is not None and self.conditional_mode == 'film':
                gamma_beta = self.film(cond)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                point_feat = point_feat * (1 + gamma) + beta
            else:
                point_feat = torch.cat([point_feat, cond], dim=1)
        pred_offset = self.linear_offset(point_feat)

        return pred_offset

def model_fn(batch, model, cfg):
    batch_size = cfg.batch_size
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']
    batch_count = batch['batch_count']
    category_ids = batch.get('category_id', None)
    pred_offset = model(feat_voxel, xyz_voxel,  v2p_index, batch_count=batch_count, category_ids=category_ids.cuda() if category_ids is not None else None)

    gt_offsets = batch['batch_offset'].cuda()
    pt_diff = pred_offset - gt_offsets  # [N, 3] float32  :l1 distance between gt and pred
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # [N]    float32  :sum l1
    valid = torch.ones(pt_dist.shape[0]).cuda()  # # get valid num
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)  # # avg

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # [N]    float32  :norm
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    pt_offsets_norm = torch.norm(pred_offset, p=2, dim=1)  # [N]    float32  :norm
    pt_offsets = pred_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    direction_diff = - (gt_offsets_ * pt_offsets).sum(-1)  # [N]    float32  :direction diff (cos)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)  # # avg
    loss = offset_norm_loss + offset_dir_loss

    with torch.no_grad():
        pred = {}

        visual_dict = {}
        visual_dict['loss'] = loss.item()

        meter_dict = {}
        meter_dict['loss'] = (loss.item(), pred_offset.shape[0])

    return loss, pred, visual_dict, meter_dict

def eval_fn(batch, model, category_ids=None, quantile=0.95, score_method='quantile'):
    """
    计算样本级异常得分，支持多种计算方法
    
    参数:
        batch: 输入批次数据
        model: 模型
        category_ids: 类别ID
        quantile: 用于计算异常得分的分位数，默认为0.95，表示使用95%分位数
        score_method: 样本级异常得分计算方法，可选'mean'（均值）、'max'（最大值）或'quantile'（分位数）
        
    返回:
        sample_score: 样本级异常得分
        pred_offset: 预测的偏移量
    """
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']
    batch_count = batch.get('batch_count', None)

    with torch.no_grad():
        pred_offset = model.test_inference(feat_voxel, xyz_voxel, v2p_index, batch_count=batch_count, category_ids=category_ids)
    
    # 计算每个点的异常得分（L1范数）
    point_scores = torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1)
    
    # 根据选择的计算方法计算样本级异常得分
    if score_method == 'mean':
        # 使用均值方法（原始方法）
        sample_score = torch.mean(point_scores)
    elif score_method == 'max':
        # 使用最大值方法
        sample_score = torch.max(point_scores)
    else:  # score_method == 'quantile'
        # 使用分位数方法
        sample_score = torch.quantile(point_scores, quantile)
    
    return sample_score, pred_offset