import os
import sys
import time
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# 获取当前文件所在目录的父目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import get_parser
from model import SingleStageNet
from network.contrast import SupConLoss, Prototypes, ProtoNCELoss
from tools import log as log_tools

torch.multiprocessing.set_start_method('spawn', force=True)

def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + np.cos(np.pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_schedule(epoch, total, warmup_pct, decay_pct, w_con_min):
    t = epoch / max(1, total)
    if t < warmup_pct:
        w_off = 0.0
        w_con = 1.0
    elif t < (warmup_pct + decay_pct):
        r = (t - warmup_pct) / max(1e-6, decay_pct)
        w_off = r
        w_con = 1.0 - (1.0 - w_con_min) * r
    else:
        w_off = 1.0
        w_con = w_con_min
    return float(w_con), float(w_off)


def get_datasets(cfg):
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
    else:
        raise RuntimeError('Unsupported dataset')
    ds = Dataset(cfg)
    ds.contrastiveLoader()  # for contrastive views
    ds.trainLoader()        # for offset regression
    return ds


def train_one_epoch(ds, model, optimizer, proto_mod, supcon_crit, proto_crit, cfg, epoch, logger, writer):
    model.train()
    am = log_tools.AverageMeter()
    start, end = time.time(), time.time()

    # build two independent loaders to avoid being overwritten
    ds.contrastiveLoader()
    loader_con = ds.train_data_loader
    ds.trainLoader()
    loader_off = ds.train_data_loader

    con_iter = iter(loader_con)
    off_iter = iter(loader_off)
    steps = max(len(loader_con), len(loader_off))

    for it in range(steps):
        try:
            con_batch = next(con_iter)
        except StopIteration:
            ds.contrastiveLoader(); loader_con = ds.train_data_loader; con_iter = iter(loader_con); con_batch = next(con_iter)
        try:
            off_batch = next(off_iter)
        except StopIteration:
            ds.trainLoader(); loader_off = ds.train_data_loader; off_iter = iter(loader_off); off_batch = next(off_iter)

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs)
        w_con, w_off = weight_schedule(epoch-1, cfg.epochs, cfg.warmup_pct, cfg.decay_pct, cfg.w_con_min)

        loss = 0.0
        # contrastive branch
        if w_con > 0:
            z1 = model.forward_embed(con_batch['feat_voxel_view1'], con_batch['xyz_voxel_view1'])
            z2 = model.forward_embed(con_batch['feat_voxel_view2'], con_batch['xyz_voxel_view2'])
            labels = con_batch['labels'].cuda(non_blocking=True)
            feats = torch.stack([z1, z2], dim=1)
            l_con = supcon_crit(feats, labels)
            if proto_mod is not None and cfg.proto_loss_weight > 0:
                with torch.no_grad():
                    proto_mod.update(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0))
                proto = F.normalize(proto_mod.proto, dim=1)
                l_proto = proto_crit(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0), proto)
                l_con = l_con + cfg.proto_loss_weight * l_proto
            loss = loss + w_con * l_con
        # offset branch
        if w_off > 0:
            xyz_voxel = off_batch['xyz_voxel']; feat_voxel = off_batch['feat_voxel']; v2p_index = off_batch['v2p_index']
            batch_count = off_batch['batch_count']; category_ids = off_batch.get('category_id', None)
            gt_offsets = off_batch['batch_offset'].cuda()
            pred_offset = model.forward_offset(feat_voxel, xyz_voxel, v2p_index, batch_count=batch_count, category_ids=category_ids.cuda() if category_ids is not None else None)
            pt_diff = pred_offset - gt_offsets
            pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
            offset_norm_loss = pt_dist.mean()
            gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)
            gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
            pt_offsets_norm = torch.norm(pred_offset, p=2, dim=1)
            pt_offsets = pred_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
            direction_diff = - (gt_offsets_ * pt_offsets).sum(-1)
            offset_dir_loss = direction_diff.mean()
            l_off = offset_norm_loss + cfg.lambda_dir * offset_dir_loss
            loss = loss + w_off * l_off

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        am.update(loss.item(), 1)
        if (it % 10) == 0:
            logger.info(f"[SS] epoch {epoch} it {it}/{steps} loss {am.avg:.4f} w_con={w_con:.2f} w_off={w_off:.2f}")

    writer.add_scalar('single_stage/loss', am.avg, epoch)
    return float(am.avg)


def main():
    cfg = get_parser() ; cfg = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    random.seed(cfg.manual_seed); np.random.seed(cfg.manual_seed); torch.manual_seed(cfg.manual_seed); torch.cuda.manual_seed(cfg.manual_seed)

    logger = log_tools.get_logger(cfg)
    writer = SummaryWriter(cfg.logpath)

    ds = get_datasets(cfg)
    num_classes = getattr(ds, 'num_classes', 0)

    model = SingleStageNet(cfg.in_channels, cfg.out_channels, num_classes=num_classes, class_embed_dim=cfg.class_embed_dim, conditional_mode=cfg.conditional_mode, proj_dim=cfg.proj_dim).cuda()
    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay)

    supcon_crit = SupConLoss(temperature=cfg.temperature).cuda()
    proto_mod = Prototypes(num_classes=num_classes, dim=cfg.proj_dim, momentum=cfg.proto_m).cuda() if num_classes > 0 else None
    proto_crit = ProtoNCELoss(temperature=cfg.temperature).cuda()

    start_epoch, pretrain_file = log_tools.checkpoint_restore(model, optimizer, cfg.logpath, pretrain_file=cfg.pretrain)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0 else 'Start from epoch {}'.format(start_epoch))

    best = 1e9
    for epoch in range(start_epoch, cfg.epochs):
        loss_epoch = train_one_epoch(ds, model, optimizer, proto_mod, supcon_crit, proto_crit, cfg, epoch, logger, writer)
        # save latest
        log_tools.checkpoint_save_newest(model, optimizer, cfg.logpath, epoch, cfg.save_freq)
        if loss_epoch < best:
            best = loss_epoch
            best_file = os.path.join(cfg.logpath, 'best.pth')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, best_file)
            logger.info(f'[SS-Best] epoch={epoch} loss={best:.6f} -> {best_file}')


if __name__ == '__main__':
    main()
