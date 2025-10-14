import os, sys
import time
import random
import torch
import numpy as np
import open3d as o3d
import torch.optim as optim
from math import cos, pi
from tensorboardX import SummaryWriter

import tools.log as log
from config.config_eval import get_parser
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score


def load_checkpoint(model, pretrain_file, gpu=0):
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    checkpoint = torch.load(pretrain_file, map_location=map_location)
    model_dict = checkpoint['model']
    for k, v in model_dict.items():
        if 'module.' in k:
            model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
        break
    model.load_state_dict(model_dict, strict=False)


def load_contrastive_checkpoint(model, pretrain_file):
    ckpt = torch.load(pretrain_file, map_location='cuda')
    model.load_state_dict(ckpt['model'], strict=False)
    prototypes = None
    if 'prototypes' in ckpt and ckpt['prototypes'] is not None:
        prototypes = ckpt['prototypes']['proto'] if 'proto' in ckpt['prototypes'] else None
    return prototypes


def eval_cluster(cfg):
    # cluster evaluation using contrastive model and prototypes
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')
        return

    from network.contrast import POContrast
    dataset = Dataset(cfg)
    # ensure using multi categories if specified
    dataset.testLoader()
    print(f'Cluster eval - test samples: {len(dataset.test_file_list)}')

    model = POContrast(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
    proto = load_contrastive_checkpoint(model, cfg.logpath + cfg.checkpoint_name)
    assert proto is not None, 'Prototypes not found in contrastive checkpoint.'
    import torch.nn.functional as F
    proto = F.normalize(torch.from_numpy(proto).float().cuda() if isinstance(proto, np.ndarray) else proto, dim=1)

    model.eval()
    correct = 0
    total = 0

    per_class_total = {}
    per_class_correct = {}

    for i, batch in enumerate(dataset.test_data_loader):
        # infer embedding
        with torch.no_grad():
            z = model.forward_embed(batch['feat_voxel'], batch['xyz_voxel'])
            z = F.normalize(z, dim=1)
            logits = torch.matmul(z, proto.T)
            pred = torch.argmax(logits, dim=1).item()
        # get true class id from path
        fn_path = batch['fn'][0]
        # category folder is -3 segment
        cat = fn_path.split('/')[-3]
        if hasattr(dataset, 'cat2id') and cat in dataset.cat2id:
            true_id = dataset.cat2id[cat]
        else:
            # fallback: cannot map
            continue
        total += 1
        correct += int(pred == true_id)
        per_class_total[true_id] = per_class_total.get(true_id, 0) + 1
        per_class_correct[true_id] = per_class_correct.get(true_id, 0) + int(pred == true_id)

    overall_acc = correct / max(total, 1)
    print(f'Cluster overall accuracy: {overall_acc:.4f} ({correct}/{total})')
    # per-class
    for cid in sorted(per_class_total.keys()):
        acc = per_class_correct.get(cid, 0) / max(per_class_total[cid], 1)
        print(f'  class_id={cid} acc={acc:.4f} ({per_class_correct.get(cid, 0)}/{per_class_total[cid]})')



def eval(cfgs):
    global cfg
    cfg = cfgs

    if getattr(cfg, 'contrastive_eval', False) or cfg.task == 'contrastive_eval':
        eval_cluster(cfg)
        return

    from network.PO3AD import PONet as net
    from network.PO3AD import eval_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda

    # build dataset first to decide num_classes for conditional head
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')
        return
    dataset = Dataset(cfg)
    num_classes = getattr(dataset, 'num_classes', 0)

    model = net(cfg.in_channels, cfg.out_channels, num_classes=num_classes, class_embed_dim=cfg.class_embed_dim, conditional_mode=cfg.conditional_mode)
    model = model.cuda()
    load_checkpoint(model, cfg.logpath + cfg.checkpoint_name)

    # optional cluster assigner for normalization
    assigner = None
    proto = None
    if getattr(cfg, 'cluster_norm', False):
        from network.contrast import POContrast
        # resolve ckpt path
        ckpt_path = cfg.contrastive_ckpt if getattr(cfg, 'contrastive_ckpt', '') else (cfg.logpath + cfg.checkpoint_name)
        try:
            assigner = POContrast(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
            proto = load_contrastive_checkpoint(assigner, ckpt_path)
            import torch.nn.functional as F
            proto = F.normalize(torch.from_numpy(proto).float().cuda() if isinstance(proto, np.ndarray) else proto, dim=1)
            assigner.eval()
        except Exception as e:
            print(f'cluster_norm setup failed: {e}')
            assigner = None
            proto = None

    # decide normal tag for test set
    if cfg.dataset == 'AnomalyShapeNet':
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        tag = 'good'
    else:
        tag = ''

    dataset.testLoader()
    print(f'Test samples: {len(dataset.test_file_list)}')

    def safe_auc_roc(y_true, y_score):
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        if len(np.unique(y_true)) < 2:
            return np.nan
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            return np.nan

    def safe_ap(y_true, y_score):
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        if len(np.unique(y_true)) < 2:
            return np.nan
        try:
            return average_precision_score(y_true, y_score)
        except Exception:
            return np.nan

    model.eval()
    label_score = []
    gt_masks = []
    pred_masks = []
    fns = []
    pred_clusters = []
    true_cats = []
    sample_point_ap = []

    for i, batch in enumerate(dataset.test_data_loader):
        sample_path = batch['fn'][0]
        fns.append(sample_path)
        # true category from path
        true_cat = sample_path.split('/')[-3]
        if hasattr(dataset, 'cat2id') and true_cat in dataset.cat2id:
            true_id = dataset.cat2id[true_cat]
        else:
            true_id = -1
        true_cats.append(true_id)

        sample_name = sample_path.split('/')[-1].split('.')[0]

        if tag in sample_name:
            gt_this = np.zeros(batch['xyz_original'].shape[0])
            gt_masks.append(gt_this)
        else:
            if cfg.dataset == 'AnomalyShapeNet':
                gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{true_cat}/GT/'
                gt_this = np.loadtxt(gt_mask_path + sample_name + '.txt', delimiter=',')[:, 3:].squeeze(1)
            elif cfg.dataset == 'Real3D':
                gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{true_cat}/gt/'
                gt_this = np.loadtxt(gt_mask_path + sample_name + '.txt')[:, 3:].squeeze(1)
            gt_masks.append(gt_this)

        # cluster prediction (before scoring) to decide conditioning id
        if assigner is not None and proto is not None:
            with torch.no_grad():
                import torch.nn.functional as F
                z = assigner.forward_embed(batch['feat_voxel'], batch['xyz_voxel'])
                z = F.normalize(z, dim=1)
                logits = torch.matmul(z, proto.T)
                cid = torch.argmax(logits, dim=1).item()
        else:
            cid = -1
        pred_clusters.append(cid)

        # conditioning category: predicted cluster if available else true category
        cond_cid = cid if cid >= 0 else true_id
        cond_ids = None if cond_cid < 0 else torch.tensor([cond_cid], dtype=torch.long).cuda()
        score, pred_mask = eval_fn(batch, model, category_ids=cond_ids)
        pred_mask = pred_mask.detach().cpu().abs().sum(dim=-1).numpy()
        # optional kNN smoothing of point scores
        if getattr(cfg, 'smooth_knn', 0) and cfg.smooth_knn > 0:
            try:
                from sklearn.neighbors import NearestNeighbors
                xyz = batch['xyz_original'].numpy()
                k = min(cfg.smooth_knn, xyz.shape[0])
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz)
                inds = nbrs.kneighbors(xyz, return_distance=False)
                pred_mask = pred_mask[inds].mean(axis=1)
            except Exception as e:
                print(f'[smooth_knn] failed: {e}')
        pred_masks.append(pred_mask)
        label_score += list(zip(batch['labels'].numpy().tolist(), [score.item()]))

        # per-sample macro AP
        if getattr(cfg, 'point_macro_ap', False):
            pts = pred_mask.copy()
            if getattr(cfg, 'sample_norm', False):
                # reuse normalization function definition below after it's defined
                pass  # will normalize later outside when function is defined
            # store tuple for later normalization depending on flag
            sample_point_ap.append(('PENDING', pts, gt_this))

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    # define normalization function once for reuse
    def normalize_array(x, method):
        x = np.asarray(x)
        if method == 'minmax':
            d = (np.max(x) - np.min(x)) + 1e-12
            return (x - np.min(x)) / d
        elif method == 'zscore':
            mu = np.mean(x)
            sd = np.std(x) + 1e-12
            return (x - mu) / sd
        elif method == 'mad':
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-12
            return (x - med) / mad
        else:
            d = (np.max(x) - np.min(x)) + 1e-12
            return (x - np.min(x)) / d

    # global metrics (no cluster norm)
    auc_roc = safe_auc_roc(labels, scores)
    auc_pr = safe_ap(labels, scores)
    point_pred_all = np.concatenate(pred_masks, axis=0)
    # global min-max
    denom = (np.max(point_pred_all) - np.min(point_pred_all)) + 1e-12
    point_pred_all = (point_pred_all - np.min(point_pred_all)) / denom
    gt_all = np.concatenate(gt_masks, axis=0)
    point_auc_roc = safe_auc_roc(gt_all, point_pred_all)
    point_auc_pr = safe_ap(gt_all, point_pred_all)
    if getattr(cfg, 'print_pos_rate', False):
        pos_rate_global = float(np.mean(gt_all))
        print(f'[Global] pos_rate={pos_rate_global:.6f}')
    print(f'[Global] object AUC-ROC: {auc_roc}, point AUC-ROC: {point_auc_roc}, object AUCP-PR: {auc_pr}, point AUCP-PR: {point_auc_pr}')

    # compute per-sample macro AP if requested
    if getattr(cfg, 'point_macro_ap', False) and len(sample_point_ap) == len(pred_masks):
        macro_list = []
        norm_type = getattr(cfg, 'cluster_norm_type', 'minmax')
        for idx, (_, pts, gt) in enumerate(sample_point_ap):
            pts_n = normalize_array(pts, norm_type) if getattr(cfg, 'sample_norm', False) else pts
            ap = safe_ap(gt, pts_n)
            if not np.isnan(ap):
                macro_list.append(ap)
        if len(macro_list) > 0:
            macro_ap = float(np.nanmean(macro_list))
            print(f'[Sample-macro] point AP (macro over samples) = {macro_ap}')

    # cluster-based normalization and per-cluster/per-category metrics
    if assigner is not None and proto is not None:
        from collections import defaultdict
        import csv

        norm_type = getattr(cfg, 'cluster_norm_type', 'minmax')
        # group by predicted cluster
        groups = defaultdict(list)
        for idx, cid in enumerate(pred_clusters):
            groups[cid].append(idx)

        print(f"\n[Per-Cluster metrics with cluster-wise {norm_type} normalization]")
        cluster_stats = {}
        for cid, idxs in sorted(groups.items(), key=lambda x: x[0]):
            if cid < 0:
                continue
            l = labels[idxs]
            s = scores[idxs]
            # object-level group normalization
            s_n = normalize_array(s, norm_type)
            auc_obj = safe_auc_roc(l, s_n)
            ap_obj = safe_ap(l, s_n)
            # point-level: concat and cluster-wise normalization
            pts = np.concatenate([pred_masks[i] for i in idxs], axis=0)
            gts = np.concatenate([gt_masks[i] for i in idxs], axis=0)
            pts_n = normalize_array(pts, norm_type)
            auc_pt = safe_auc_roc(gts, pts_n)
            ap_pt = safe_ap(gts, pts_n)
            pos_rate = float(np.mean(gts)) if getattr(cfg, 'print_pos_rate', False) else None
            cluster_stats[cid] = (auc_obj, auc_pt, ap_obj, ap_pt, len(idxs), pos_rate)
            if getattr(cfg, 'print_pos_rate', False):
                print(f'  [cluster {cid}] N={len(idxs)} objAUC={auc_obj} ptAUC={auc_pt} objAP={ap_obj} ptAP={ap_pt} pos_rate={pos_rate:.6f}')
            else:
                print(f'  [cluster {cid}] N={len(idxs)} objAUC={auc_obj} ptAUC={auc_pt} objAP={ap_obj} ptAP={ap_pt}')
        # macro average over clusters with valid metrics
        macro_cluster = None
        if len(cluster_stats) > 0:
            valid = [v for k, v in cluster_stats.items() if not (np.isnan(v[0]) or np.isnan(v[1]))]
            if len(valid) > 0:
                mean_obj = float(np.nanmean([v[0] for v in valid]))
                mean_pt = float(np.nanmean([v[1] for v in valid]))
                mean_obj_ap = float(np.nanmean([v[2] for v in valid]))
                mean_pt_ap = float(np.nanmean([v[3] for v in valid]))
                macro_cluster = (mean_obj, mean_pt, mean_obj_ap, mean_pt_ap)
                print(f'  [cluster-macro] objAUC={mean_obj} ptAUC={mean_pt} objAP={mean_obj_ap} ptAP={mean_pt_ap}')

        # per-true-category metrics (using true category parsed from path)
        cat_groups = defaultdict(list)
        for idx, tc in enumerate(true_cats):
            cat_groups[tc].append(idx)
        print(f"\n[Per-Category metrics (by true category) with category-wise {norm_type} normalization]")
        cat_stats = {}
        for tc, idxs in sorted(cat_groups.items(), key=lambda x: x[0]):
            if tc < 0:
                continue
            l = labels[idxs]
            s = scores[idxs]
            s_n = normalize_array(s, norm_type)
            auc_obj = safe_auc_roc(l, s_n)
            ap_obj = safe_ap(l, s_n)
            pts = np.concatenate([pred_masks[i] for i in idxs], axis=0)
            gts = np.concatenate([gt_masks[i] for i in idxs], axis=0)
            pts_n = normalize_array(pts, norm_type)
            auc_pt = safe_auc_roc(gts, pts_n)
            ap_pt = safe_ap(gts, pts_n)
            pos_rate = float(np.mean(gts)) if getattr(cfg, 'print_pos_rate', False) else None
            cat_stats[tc] = (auc_obj, auc_pt, ap_obj, ap_pt, len(idxs), pos_rate)
            if getattr(cfg, 'print_pos_rate', False):
                print(f'  [cat {tc}] N={len(idxs)} objAUC={auc_obj} ptAUC={auc_pt} objAP={ap_obj} ptAP={ap_pt} pos_rate={pos_rate:.6f}')
            else:
                print(f'  [cat {tc}] N={len(idxs)} objAUC={auc_obj} ptAUC={auc_pt} objAP={ap_obj} ptAP={ap_pt}')
        macro_cat = None
        if len(cat_stats) > 0:
            valid = [v for k, v in cat_stats.items() if not (np.isnan(v[0]) or np.isnan(v[1]))]
            if len(valid) > 0:
                mean_obj = float(np.nanmean([v[0] for v in valid]))
                mean_pt = float(np.nanmean([v[1] for v in valid]))
                mean_obj_ap = float(np.nanmean([v[2] for v in valid]))
                mean_pt_ap = float(np.nanmean([v[3] for v in valid]))
                macro_cat = (mean_obj, mean_pt, mean_obj_ap, mean_pt_ap)
                print(f'  [category-macro] objAUC={mean_obj} ptAUC={mean_pt} objAP={mean_obj_ap} ptAP={mean_pt_ap}')

        # optional CSV export
        csv_path = getattr(cfg, 'metrics_csv', '')
        if not csv_path:
            # default path under logpath
            csv_path = os.path.join(cfg.logpath, 'eval_metrics.csv')
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['section','id','N','objAUC','ptAUC','objAP','ptAP','norm','pos_rate'])
                # global
                pos_rate_global = float(np.mean(gt_all)) if getattr(cfg, 'print_pos_rate', False) else ''
                writer.writerow(['global','all',len(labels),auc_roc,point_auc_roc,auc_pr,point_auc_pr,'global-minmax',pos_rate_global])
                # clusters
                for cid, v in sorted(cluster_stats.items(), key=lambda x: x[0]):
                    writer.writerow(['cluster',cid,v[4],v[0],v[1],v[2],v[3],norm_type,'' if v[5] is None else v[5]])
                if macro_cluster is not None:
                    writer.writerow(['cluster-macro','avg','',macro_cluster[0],macro_cluster[1],macro_cluster[2],macro_cluster[3],norm_type,''])
                # categories
                for tc, v in sorted(cat_stats.items(), key=lambda x: x[0]):
                    writer.writerow(['category',tc,v[4],v[0],v[1],v[2],v[3],norm_type,'' if v[5] is None else v[5]])
                if macro_cat is not None:
                    writer.writerow(['category-macro','avg','',macro_cat[0],macro_cat[1],macro_cat[2],macro_cat[3],norm_type,''])
                # per-sample macro AP
                if getattr(cfg, 'point_macro_ap', False) and len(sample_point_ap) == len(pred_masks):
                    writer.writerow(['sample-macro','ptAP',len(pred_masks),'','','',macro_ap,'', ''])
            print(f'[Metrics CSV] saved to {csv_path}')
        except Exception as e:
            print(f'[Metrics CSV] failed to save: {e}')



if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    eval(cfg)
