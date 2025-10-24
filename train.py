import os, sys
import time
import random
import torch
import numpy as np
import open3d as o3d
import torch.optim as optim
import torch.nn.functional as F
from math import cos, pi
from tensorboardX import SummaryWriter

import tools.log as log
from config.config_train import get_parser

# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(train_loader, model, model_fn, optimizer, epoch, max_batch_iter):
    model.train()

    # #for log the run time and remain time
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()  # initialization
    am_dict = {}

     # log first round epoch and lr
    try:
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f"[Stage2] start epoch={epoch} lr={cur_lr}")
    except Exception:
        pass

    # #start train
    for i, batch in enumerate(train_loader):
        batch_time.update(time.time() - end_time)  # update time

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)  # adjust lr
        loss, _, visual_dict, meter_dict = model_fn(batch, model, cfg)

        # # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #average batch loss, time for print
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = log.AverageMeter()
            am_dict[k].update(v[0], v[1])

        current_iter = (epoch-1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        sys.stdout.write("epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f})  data_time: {:.2f}({:.2f}) "
                         "iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                         .format(epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val,
                                 am_dict['loss'].avg,
                                 batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
                                 remain_time=remain_time))
        if (i == len(train_loader) - 1): print()

    epoch_loss = am_dict['loss'].avg if 'loss' in am_dict else None
    logger.info("epoch: {}/{}, train loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, epoch_loss,
                                                                      time.time() - start_time))
    # #write tensorboardX
    lr = optimizer.param_groups[0]['lr']
    if epoch_loss is not None:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/learning_rate', lr, epoch)

    # save pretrained model
    pretrain_file = log.checkpoint_save_newest(model, optimizer, cfg.logpath, epoch, cfg.save_freq)
    logger.info('Saving {}'.format(pretrain_file))

    # return epoch loss for best selection
    return epoch_loss
    # return epoch metrics for best selection
    return am_loss.avg, am_acc.avg
    pass


def train_epoch_contrastive(train_loader, model, supcon_criterion, proto_module, proto_criterion, optimizer, epoch):
    model.train()
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()
    am_loss = log.AverageMeter()
    am_acc = log.AverageMeter()

    # log first round epoch and lr
    try:
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f"[Con] start epoch={epoch} lr={cur_lr}")
    except Exception:
        pass

    # collect embeddings across many batches for better visualization
    viz_z_list = []
    viz_y_list = []

    for i, batch in enumerate(train_loader):
        batch_time.update(time.time() - end_time)
        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)

        # forward two views
        z1 = model.forward_embed(batch['feat_voxel_view1'], batch['xyz_voxel_view1']) # (B,proj_dim)
        z2 = model.forward_embed(batch['feat_voxel_view2'], batch['xyz_voxel_view2']) # (B,proj_dim)
        labels = batch['labels'].cuda(non_blocking=True)
        z1 = z1.cuda()
        z2 = z2.cuda()
        feats = torch.stack([z1, z2], dim=1)  # [B, 2, D]

        loss = supcon_criterion(feats, labels)
        # optional prototype nce
        if proto_module is not None and proto_criterion is not None and cfg.proto_loss_weight > 0:
            with torch.no_grad():
                proto_module.update(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0))
            proto = F.normalize(proto_module.proto, dim=1)
            proto_loss = proto_criterion(torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0), proto)
            loss = loss + cfg.proto_loss_weight * proto_loss
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics: nearest-prototype accuracy and intra/inter stats
        with torch.no_grad():
            if proto_module is not None:
                proto = F.normalize(proto_module.proto, dim=1)
                z_mean = F.normalize(0.5 * (z1 + z2), dim=1)
                logits = torch.matmul(z_mean, proto.T) 
                preds = torch.argmax(logits, dim=1) #通过比较样本特征与所有原型的相似度预测类别样本
                acc = (preds == labels).float().mean().item() #计算最近原型准确率
                # intra-class variance (distance to own prototype)
                own_proto = proto[labels]
                intra = (1 - (z_mean * own_proto).sum(dim=1)).mean().item() #计算类内方差（样本与其所属类别原型之间的平均距离）
                # inter-class average distance between prototypes
                if proto.shape[0] > 1:
                    pp = torch.matmul(proto, proto.T)
                    mask = ~torch.eye(proto.shape[0], dtype=torch.bool, device=proto.device)
                    inter = (1 - pp[mask]).mean().item() #计算类间距离：不同原型之间的平均距离
                else:
                    inter = 0.0
            else:
                acc, intra, inter = 0.0, 0.0, 0.0

        am_loss.update(loss.item(), labels.shape[0])
        am_acc.update(acc, labels.shape[0])

        # 收集部分样本的嵌入向量和标签，用于可视化
        try:
            if cfg.emb_save_samples > 0 and (sum([arr.shape[0] for arr in viz_z_list]) < cfg.emb_save_samples):
                take = min(labels.shape[0], cfg.emb_save_samples - sum([arr.shape[0] for arr in viz_z_list]))
                if take > 0:
                    viz_z_list.append(z_mean.detach().cpu().numpy()[:take])
                    viz_y_list.append(labels.detach().cpu().numpy()[:take])
        except Exception:
            pass

        current_iter = (epoch-1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        # sys.stdout.write("[Con] epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) acc: {:.3f} intra: {:.3f} inter: {:.3f}  data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
        #                  .format(epoch, cfg.epochs, i + 1, len(train_loader), am_loss.val,
        #                          am_loss.avg, acc, intra, inter,
        #                          batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
        #                          remain_time=remain_time))
        line = "[Con] epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) acc: {:.3f} intra: {:.3f} inter: {:.3f}  data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {}\n".format(
            epoch, cfg.epochs, i + 1, len(train_loader), am_loss.val,
            am_loss.avg, acc, intra, inter,
            batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
            remain_time)
        sys.stdout.write(line)
        try:
            logger.info(line.strip())
        except Exception:
            pass
        if (i == len(train_loader) - 1): print()

    logger.info("[Con] epoch: {}/{}, train loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_loss.avg,
                                                                           time.time() - start_time))
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('contrastive/train_loss', am_loss.avg, epoch)
    writer.add_scalar('contrastive/train_acc', am_acc.avg, epoch)
    writer.add_scalar('train/learning_rate', lr, epoch)

    # save embedding snapshots for visualization
    try:
        embed_dir = os.path.join(cfg.logpath, 'embeds')
        os.makedirs(embed_dir, exist_ok=True)
        # save multi-batch snapshot if scheduled
        if (epoch % max(1, getattr(cfg, 'emb_save_every', 1))) == 0 and len(viz_z_list) > 0:
            z_cat = np.concatenate(viz_z_list, axis=0)
            y_cat = np.concatenate(viz_y_list, axis=0)
            out_multi = os.path.join(embed_dir, f'epoch_{epoch:04d}_multi.npz')
            np.savez(out_multi, z=z_cat, y=y_cat)
            try:
                logger.info(f'[Embeds] epoch={epoch} saved multi snapshot: {z_cat.shape[0]} samples -> {out_multi}')
            except Exception:
                pass
        # also save last batch snapshot for quick check
        # out_last = os.path.join(embed_dir, f'epoch_{epoch:04d}.npz')
        # np.savez(out_last, z=z_mean.detach().cpu().numpy(), y=labels.detach().cpu().numpy())
    except Exception as e:
        logger.info(f'Embedding save failed: {e}')

    # save model with prototypes
    # save_contrastive_checkpoint(model, optimizer, proto_module, cfg.logpath, epoch, cfg.save_freq)

    # save model(latest only for contrastive)
    save_contrastive_checkpoint_latest(model, optimizer, proto_module, cfg.logpath, epoch)

    # return epoch metrics for best selection
    return am_loss.avg, am_acc.avg

def save_contrastive_checkpoint(model, optimizer, proto_module, logpath, epoch, save_freq=1):
    pretrain_file = os.path.join(logpath + '%09d' % epoch + '.pth')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'prototypes': proto_module.state_dict() if proto_module is not None else None,
        'epoch': epoch,
    }, pretrain_file)
    # remove very old checkpoints (keep last ~10 epochs)
    old_epoch = epoch - 10
    fd = os.path.join(logpath + '%09d' % old_epoch + '.pth')
    if os.path.isfile(fd):
        try:
            os.remove(fd)
        except Exception:
            pass
    return pretrain_file


def save_contrastive_checkpoint_latest(model, optimizer, proto_module, logpath, epoch):
    """Save only the latest contrastive checkpoint (overwrite)."""
    latest_file = os.path.join(logpath, 'latest.pth')
    os.makedirs(logpath, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'prototypes': proto_module.state_dict() if proto_module is not None else None,
        'epoch': epoch,
    }, latest_file)
    try:
        logger.info(f'[Latest] epoch={epoch} saved -> {latest_file}')
    except Exception:
        pass
    return latest_file


def restore_contrastive_checkpoint(model, optimizer, proto_module, logpath, pretrain_file=''):
    epoch = 0
    # if not pretrain_file:
    #     files = []
    #     import glob
    #     files = sorted(glob.glob(os.path.join(logpath + '00*.pth')))
    #     if len(files) > 0:
    #         pretrain_file = files[-1]
    #         try:
    #             epoch = int(pretrain_file[len(logpath) + 2: -4])
    #         except Exception:
    #             epoch = 0

    if not pretrain_file:
        import os, glob
        # prefer latest.pth, then best.pth, then numbered checkpoints
        cand = os.path.join(logpath, 'latest.pth')
        if os.path.isfile(cand):
            pretrain_file = cand
        else:
            cand = os.path.join(logpath, 'best.pth')
            if os.path.isfile(cand):
                pretrain_file = cand
            else:
                files = sorted(glob.glob(os.path.join(logpath + '00*.pth')))
                if len(files) > 0:
                    pretrain_file = files[-1]
                    try:
                        epoch = int(pretrain_file[len(logpath) + 2: -4])
                    except Exception:
                        epoch = 0

    if len(pretrain_file) > 0 and os.path.isfile(pretrain_file):
        ckpt = torch.load(pretrain_file, map_location='cuda')
        model.load_state_dict(ckpt['model'], strict=False)
        # prefer saved epoch in ckpt if available
        if 'epoch' in ckpt:
            try:
                epoch = int(ckpt['epoch'])
            except Exception:
                pass
        if optimizer is not None and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            for state in optimizer.state.values():
                if state is None: continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if proto_module is not None and 'prototypes' in ckpt and ckpt['prototypes'] is not None:
            proto_module.load_state_dict(ckpt['prototypes'], strict=False)
    return epoch + 1, pretrain_file


def SingleCard_training(cfgs):
    global cfg
    cfg = cfgs
    # logger and summary write
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # # summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    use_cuda = torch.cuda.is_available()
    assert use_cuda

    # load dataset
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)

    if cfg.contrastive or cfg.task == 'contrastive':
        logger.info('=> creating contrastive model ...')
        from network.contrast import POContrast, SupConLoss, Prototypes, ProtoNCELoss
        model = POContrast(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        # optimizer
        if cfg.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
        elif cfg.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                                  momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                    weight_decay=cfg.weight_decay)
        # dataloader for contrastive
        dataset.contrastiveLoader()
        logger.info('Contrastive training samples: {}'.format(len(dataset.train_file_list)))
        # losses
        supcon_criterion = SupConLoss(temperature=cfg.temperature).cuda()
        proto_module = Prototypes(num_classes=dataset.num_classes, dim=cfg.proj_dim, momentum=cfg.proto_m).cuda()
        proto_criterion = ProtoNCELoss(temperature=cfg.temperature).cuda()
        # restore
        # cfg.pretrain = '' #设置为空 自动查找最新检查点
        start_epoch, pretrain_file = restore_contrastive_checkpoint(model, optimizer, proto_module, cfg.logpath, pretrain_file=cfg.pretrain)
        logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                    else 'Start from epoch {}'.format(start_epoch))
        best_metric = -1e9 if cfg.contrastive_best_metric == 'acc' else 1e9
        best_file = None
        for epoch in range(start_epoch, cfg.epochs):
            epoch_loss, epoch_acc = train_epoch_contrastive(dataset.train_data_loader, model, supcon_criterion, proto_module, proto_criterion, optimizer, epoch)
            cur_metric = epoch_acc if cfg.contrastive_best_metric == 'acc' else epoch_loss
            improved = (cur_metric > best_metric) if cfg.contrastive_best_metric == 'acc' else (cur_metric < best_metric)
            if improved:
                best_metric = cur_metric
                # save best checkpoint as a dedicated file
                best_file = os.path.join(cfg.logpath, 'best.pth')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'prototypes': proto_module.state_dict(), 'epoch': epoch}, best_file)
                logger.info(f'[Best] epoch={epoch} save best (metric={cfg.contrastive_best_metric}, value={best_metric:.6f}) -> {best_file}')
        return

    # default: offset regression training
    # dataset first to get num_classes
    dataset.trainLoader()
    logger.info('Training samples: {}'.format(len(dataset.train_file_list)))

    logger.info('=> creating model ...')
    from network.PO3AD import PONet as net
    from network.PO3AD import model_fn
    num_classes = getattr(dataset, 'num_classes', 0)
    model = net(cfg.in_channels, cfg.out_channels, num_classes=num_classes, class_embed_dim=cfg.class_embed_dim, conditional_mode=cfg.conditional_mode)
    model = model.cuda()

    # optionally load contrastive backbone weights 仅把第一阶段MInkUNet部分拷贝到第二阶段的PONet.backbone上，其他的部分保持随机初始化
    if getattr(cfg, 'contrastive_backbone', ''): 
        try:
            ckpt = torch.load(cfg.contrastive_backbone, map_location='cuda')
            state = ckpt['model'] if 'model' in ckpt else ckpt
            model_dict = model.state_dict()
            mapped = {k: v for k, v in state.items() if k.startswith('backbone.') and k in model_dict}
            model_dict.update(mapped)
            model.load_state_dict(model_dict, strict=False)
            logger.info(f'Loaded backbone from contrastive ckpt: {cfg.contrastive_backbone} ({len(mapped)} params)')
        except Exception as e:
            logger.info(f'Load contrastive backbone failed: {e}')

    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)

    max_batch_iter = len(dataset.train_file_list) // cfg.batch_size

    # #train
    # cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, optimizer, cfg.logpath, pretrain_file=cfg.pretrain)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))

    best_metric = 1e9  # for loss, lower is better
    best_file = None
    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss = train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch, max_batch_iter)
        if epoch_loss is not None:
            cur_metric = epoch_loss  # only loss for stage-2
            if cur_metric < best_metric:
                best_metric = cur_metric
                best_file = os.path.join(cfg.logpath, 'best.pth')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, best_file)
                logger.info(f'[Stage2 Best] epoch={epoch} save best (loss={best_metric:.6f}) -> {best_file}')
    pass




if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    # fix seed for debug
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)

    # # Determine whether it is distributed training
    SingleCard_training(cfg)