import os, sys
import time
import random
import torch
import numpy as np
import open3d as o3d
import torch.optim as optim
from pathlib import Path
from math import cos, pi
from tensorboardX import SummaryWriter

import tools.log as log
from config.config_eval import get_parser
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score


def load_checkpoint(model, pretrain_file, gpu=0):
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    checkpoint = torch.load(pretrain_file, map_location=map_location)
    model_dict = checkpoint['model']
    for k, v in model_dict.items(): #移除module前缀，PyTorch会自动在每个参数名前添加'module.'前缀。但在单GPU加载时，模型不需要此前缀
        if 'module.' in k:
            model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
        break
    model.load_state_dict(model_dict, strict=False)


def load_contrastive_checkpoint(model, pretrain_file): #加载对比学习checkpoint并提取原型向量
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
    proto = load_contrastive_checkpoint(model, cfg.contrastive_ckpt)
    assert proto is not None, 'Prototypes not found in contrastive checkpoint.'
    import torch.nn.functional as F
    proto = F.normalize(torch.from_numpy(proto).float().cuda() if isinstance(proto, np.ndarray) else proto, dim=1)

    model.eval()
    correct = 0 #预测正确的数量
    total = 0 #总样本数

    per_class_total = {}
    per_class_correct = {}

    for i, batch in enumerate(dataset.test_data_loader):
        # infer embedding
        with torch.no_grad():
            z = model.forward_embed(batch['feat_voxel'], batch['xyz_voxel'])
            logits = torch.matmul(z, proto.T) #计算嵌入向量与所有原型向量的点积
            pred = torch.argmax(logits, dim=1).item() # 取相似度最大的作为预测类别
        # get true class id from path
        fn_path = batch['fn'][0]
        # category folder is -3 segment
        # cat = fn_path.split('/')[-3]
        parts = Path(fn_path).parts
        cat = parts[-3] if len(parts) >= 3 else ''
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

    # optional: tee stdout to a markdown buffer for saving later
    orig_stdout = sys.stdout
    md_capture = None
    md_path = getattr(cfg, 'metrics_md', '')
    if md_path:
        from io import StringIO
        md_capture = StringIO()
        class _Tee:
            def __init__(self, a, b):
                self.a = a; self.b = b
            def write(self, data):
                self.a.write(data) #写入到第一个输出流
                self.b.write(data) #写入到第二个输出流
            def flush(self): #刷新方式
                try:
                    self.a.flush()
                except Exception:
                    pass
                try:
                    self.b.flush()
                except Exception:
                    pass
        sys.stdout = _Tee(orig_stdout, md_capture) #输出分流

    if getattr(cfg, 'contrastive_eval', False) or cfg.task == 'contrastive_eval':
        eval_cluster(cfg)
        # save md if requested
        if md_capture is not None:
            sys.stdout = orig_stdout
            try:
                os.makedirs(os.path.dirname(md_path), exist_ok=True)
            except Exception:
                pass
            try:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_capture.getvalue())
                print(f'[Metrics MD] saved to {md_path}')
            except Exception as e:
                print(f'[Metrics MD] failed to save: {e}')
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

    # optional: restrict evaluation to a single category while keeping unified model
    if getattr(cfg, 'eval_category_only', ''):
        only_cat = cfg.eval_category_only
        if hasattr(dataset, 'test_file_list'):
            before = len(dataset.test_file_list)
            filtered = []
            for p in dataset.test_file_list:
                # robust category extraction
                parts = p.replace('\\', '/').split('/')
                cat = parts[-3] if len(parts) >= 3 else ''
                if cat == only_cat:
                    filtered.append(p)
            dataset.test_file_list = filtered
            after = len(dataset.test_file_list)
            print(f'Evaluating category-only subset: "{only_cat}" | samples: {after} (was {before})')

    model = net(cfg.in_channels, cfg.out_channels, num_classes=num_classes, class_embed_dim=cfg.class_embed_dim, conditional_mode=cfg.conditional_mode)
    model = model.cuda()
    load_checkpoint(model, cfg.logpath + cfg.checkpoint_name)

    # optional cluster assigner for normalization
    assigner = None #聚类分配器
    proto = None
    # decide if we need cluster predictions (for conditioning or per-cluster metrics)
    cond_src_eval = str(getattr(cfg, 'conditioning_source', 'auto')).lower()
    need_cluster_preds_for_cond = (cond_src_eval in ['cluster', 'auto'])
    need_assigner = bool(getattr(cfg, 'cluster_norm', False) or need_cluster_preds_for_cond)
    if need_assigner:
        from network.contrast import POContrast
        # resolve ckpt path
        ckpt_path = cfg.contrastive_ckpt if getattr(cfg, 'contrastive_ckpt', '') else (cfg.logpath + cfg.checkpoint_name)
        try:
            assigner = POContrast(cfg.in_channels, cfg.out_channels, proj_dim=cfg.proj_dim).cuda()
            proto = load_contrastive_checkpoint(assigner, ckpt_path) #(40,128)
            import torch.nn.functional as F
            proto = F.normalize(torch.from_numpy(proto).float().cuda() if isinstance(proto, np.ndarray) else proto, dim=1)
            assigner.eval()
        except Exception as e:
            print(f'cluster_norm setup failed: {e}')
            assigner = None
            proto = None

    # conditioning source option
    cond_src = cond_src_eval
    if cond_src not in ['auto', 'true', 'cluster', 'none']:
        cond_src = 'auto'

    # report cluster assigner / conditioning status
    cluster_assigner_ready = (assigner is not None and proto is not None)
    print(f"[Conditioning] source={cond_src}")
    if getattr(cfg, 'cluster_norm', False):
        if cluster_assigner_ready:
            print('[Cluster] cluster_norm=True -> Predicted clusters available for per-cluster metrics.')
        else:
            print('[Cluster] cluster_norm=True but assigner unavailable -> Per-cluster metrics disabled.')
    else:
        if need_cluster_preds_for_cond and cluster_assigner_ready:
            print('[Cluster] using predicted clusters for conditioning only (no per-cluster metrics).')
        else:
            print('[Cluster] predicted clusters disabled; using category metrics only.')

    # decide normal tag for test set
    if cfg.dataset == 'AnomalyShapeNet':
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        tag = 'good'
    # else:
    #     tag = ''

    dataset.testLoader()
    print(f'Test samples: {len(dataset.test_file_list)}')

    
    # BN-TTA: refresh BN running stats using a few test samples
    if getattr(cfg, 'bn_tta', False) and int(getattr(cfg, 'bn_tta_samples', 0)) > 0: #?
        try:
            n_bntta = int(min(getattr(cfg, 'bn_tta_samples', 16), len(dataset.test_file_list)))
            print(f'[BN-TTA] refreshing BN stats with {n_bntta} samples ...')
            t_start_bn = time.time()
            model.train()
            with torch.no_grad():
                it = 0
                for batch in dataset.test_data_loader:
                    _ = model.test_inference(batch['feat_voxel'], batch['xyz_voxel'], batch['v2p_index'], batch_count=batch.get('batch_count', None), category_ids=None)
                    it += 1
                    if it >= n_bntta:
                        break
            model.eval()
            bn_tta_time_total = time.time() - t_start_bn
        except Exception as e:
            print(f'[BN-TTA] failed: {e}')

    # timing accumulators
    pre_infer_time_sum = 0.0           # baseline forward per sample
    post_infer_time_sum = 0.0          # extra TTA forward time (geom views)
    ttt_time_sum = 0.0                 # TTT adaptation time
    bn_tta_time_total = 0.0            # BN refresh time (global)
    sample_count = 0
    

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
    # pre/post containers for global comparison
    label_score_pre = []
    label_score_post = []
    pred_masks_pre = []
    pred_masks_post = []

    gt_masks = []
    pred_masks = []
    fns = []
    pred_clusters = []
    true_cats = []
    xyz_list = []

    # counters for conditioning decisions
    used_pred_cluster = 0
    used_true_cond = 0
    used_unconditional = 0

    for i, batch in enumerate(dataset.test_data_loader):
        sample_count += 1
        sample_path = batch['fn'][0]
        fns.append(sample_path)
        # true category from path
        # true_cat = sample_path.split('/')[-3]
        true_cat = Path(sample_path).parts[-3] if len(Path(sample_path).parts) >= 3 else ''
        if hasattr(dataset, 'cat2id') and true_cat in dataset.cat2id:
            true_id = dataset.cat2id[true_cat]
        else:
            true_id = -1
        true_cats.append(true_id)

        sample_name = sample_path.split('/')[-1].split('.')[0]
        # sample_name = Path(sample_path).stem

        if tag in sample_name:
            gt_this = np.zeros(batch['xyz_original'].shape[0])
            gt_masks.append(gt_this)
        else:
            if cfg.dataset == 'AnomalyShapeNet':
                gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{true_cat}/GT/'
                gt_this = np.loadtxt(gt_mask_path + sample_name + '.txt', delimiter=',')[:, 3:].squeeze(1)
            elif cfg.dataset == 'Real3D':
                # gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{true_cat}/gt/'
                # gt_this = np.loadtxt(gt_mask_path + sample_name + '.txt')[:, 3:].squeeze(1)
                gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{true_cat}/gt/'
                gt_file = gt_mask_path + sample_name + '.txt'
                try:
                    arr = np.loadtxt(gt_file)
                except Exception:
                    arr = np.loadtxt(gt_file, delimiter=',')
                # robust selection: if has >=4 columns, use last; else use last available
                if arr.ndim == 1:
                    gt_this = np.array(arr[-1:]).reshape(-1)
                else:
                    gt_this = arr[:, -1].reshape(-1)
            gt_masks.append(gt_this)

        # cluster prediction (before scoring) to decide conditioning id
        if assigner is not None and proto is not None and (need_cluster_preds_for_cond or getattr(cfg, 'cluster_norm', False)):
            with torch.no_grad():
                import torch.nn.functional as F
                z = assigner.forward_embed(batch['feat_voxel'], batch['xyz_voxel']) #(1,128) (B,proj_dim)
                # z = F.normalize(z, dim=1)
                logits = torch.matmul(z, proto.T)
                cid = torch.argmax(logits, dim=1).item()
                # optional Prototype-EMA update with confidence gating
                if getattr(cfg, 'proto_ema', False):
                    try:
                        # 使用logits最大值代替softmax概率
                        # prob = torch.softmax(logits, dim=1)[0] # 40
                        # conf, cmax = float(prob.max().item()), int(prob.argmax().item())
                        logits_value = logits[0]  # 获取logits值
                        conf, cmax = float(logits_value.max().item()), int(logits_value.argmax().item())
                        if conf >= float(getattr(cfg, 'proto_ema_tau', 0.8)):
                            m = float(getattr(cfg, 'proto_ema_m', 0.99))
                            old = proto[cmax:cmax+1]
                            new = F.normalize(m * old + (1.0 - m) * z, dim=1)
                            proto[cmax:cmax+1] = new
                    except Exception as _:
                        pass
        else:
            cid = -1
        pred_clusters.append(cid)

        # conditioning category: follow user option
        if cond_src == 'true':
            cond_cid = true_id
        elif cond_src == 'cluster':
            cond_cid = cid if cid >= 0 else -1
        elif cond_src == 'none':
            cond_cid = -1
        else:  # auto: cluster then true
            cond_cid = cid if cid >= 0 else true_id
        cond_ids = None if cond_cid < 0 else torch.tensor([cond_cid], dtype=torch.long).cuda()

        # logging for usage
        if cond_src == 'true':
            used_true_cond += 1
        elif cond_src == 'cluster':
            if cond_cid >= 0:
                used_pred_cluster += 1
            else:
                used_unconditional += 1
                print(f"[Cond][UNCOND] sample='{sample_name}' (cluster chosen but unavailable)")
        elif cond_src == 'none':
            used_unconditional += 1
        else:  # auto
            if cid >= 0:
                used_pred_cluster += 1
            else:
                if true_id >= 0:
                    used_true_cond += 1
                    print(f"[Cond][Auto->TRUE] sample='{sample_name}' true_cat_id={true_id} (predicted cluster unavailable)")
                else:
                    used_unconditional += 1
                    print(f"[Cond][Auto->UNCOND] sample='{sample_name}' (no predicted cluster and true category unknown)")

        # baseline forward (pre-TTA)
        t_pre0 = time.time()
        score, pred_mask_tensor = eval_fn(batch, model, category_ids=cond_ids, 
                                          quantile=getattr(cfg, 'score_quantile', 0.95),
                                          score_method=getattr(cfg, 'score_method', 'quantile'))
        pre_infer_time_sum += (time.time() - t_pre0)
        pred_mask_base = pred_mask_tensor.detach().cpu().abs().sum(dim=-1).numpy()
        xyz_np = batch['xyz_original'].numpy()
        xyz_list.append(xyz_np)

        # Prepare optional smoothing indices once
        inds = None
        if getattr(cfg, 'smooth_knn', 0) and cfg.smooth_knn > 0:
            try:
                from sklearn.neighbors import NearestNeighbors
                k = min(cfg.smooth_knn, xyz_np.shape[0])
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz_np)
                inds = nbrs.kneighbors(xyz_np, return_distance=False)  # (point_num,k)
            except Exception as e:
                print(f'[smooth_knn] failed to build index: {e}')
                inds = None

        # pre-TTA mask (raw) and optional smoothed copy for metrics only
        pred_mask_pre_raw = pred_mask_base  # do not smooth the raw mask used for TTA fusion
        pred_mask_pre = pred_mask_pre_raw
        if inds is not None:
            try:
                pred_mask_pre = pred_mask_pre_raw[inds].mean(axis=1)
            except Exception as e:
                print(f'[smooth_knn] apply(pre) failed: {e}')

        # optional TTT adaptation before final inference / TTA fusion
        if getattr(cfg, 'ttt_enable', False) and int(getattr(cfg, 'ttt_steps', 0)) > 0:
            t_ttt0 = time.time()
            try: # 选择最后一个head的参数
                steps = int(getattr(cfg, 'ttt_steps', 1))
                lr = float(getattr(cfg, 'ttt_lr', 1e-4))
                # select parameters from the last head only
                params = []
                for n, p in model.named_parameters():
                    if n.startswith('linear_offset'):
                        p.requires_grad = True
                        params.append(p)
                    else:
                        p.requires_grad = False
                opt = torch.optim.SGD(params, lr=lr, momentum=0.0)
                for _ in range(steps):
                    opt.zero_grad()
                    # base forward
                    score0, pred0 = eval_fn(batch, model, category_ids=cond_ids,
                                           quantile=getattr(cfg, 'score_quantile', 0.95),
                                           score_method=getattr(cfg, 'score_method', 'quantile'))
                    obj0 = pred0.abs().sum(dim=-1).mean()
                    # weak view: small rotate + jitter
                    base_xyz = batch['xyz_original'].numpy()
                    import math, MinkowskiEngine as ME  #生成弱增强视图
                    deg = float(getattr(cfg, 'ttt_weak_rotate_deg', 2.0))
                    ax, ay, az = np.deg2rad(np.random.uniform(-deg, deg, size=3))
                    Rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]], dtype=np.float32)
                    Ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]], dtype=np.float32)
                    Rz = np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]], dtype=np.float32)
                    R = Rz @ Ry @ Rx
                    xyz_w = base_xyz @ R.T
                    sj = float(getattr(cfg, 'ttt_weak_jitter', 0.001))
                    if sj > 0:
                        xyz_w = xyz_w + np.random.normal(scale=sj, size=xyz_w.shape).astype(np.float32)
                    q, f, index, inv = ME.utils.sparse_quantize(xyz_w.astype(np.float32), xyz_w.astype(np.float32), quantization_size=cfg.voxel_size, return_index=True, return_inverse=True)
                    xyz_voxel_t, feat_voxel_t = ME.utils.sparse_collate([q],[f])
                    if isinstance(inv, np.ndarray): v2p_t = torch.from_numpy(inv).long()
                    elif torch.is_tensor(inv): v2p_t = inv.long().cpu()
                    else: v2p_t = torch.as_tensor(inv, dtype=torch.long)
                    batch_count_t = torch.tensor([0, xyz_w.shape[0]], dtype=torch.int64)
                    tta_batch = {'xyz_voxel': xyz_voxel_t, 'feat_voxel': feat_voxel_t, 'v2p_index': v2p_t, 'batch_count': batch_count_t}
                    _, pred_w = eval_fn(tta_batch, model, category_ids=cond_ids,
                                       quantile=getattr(cfg, 'score_quantile', 0.95),
                                       score_method=getattr(cfg, 'score_method', 'quantile'))
                    objw = pred_w.abs().sum(dim=-1).mean()
                    # losses
                    loss_cons = (objw - obj0).pow(2)
                    loss_reg = 0.0
                    lam_reg = float(getattr(cfg, 'ttt_reg', 1e-3))
                    if lam_reg > 0:
                        for p in params:
                            loss_reg = loss_reg + p.pow(2).sum()
                    loss = float(getattr(cfg, 'ttt_consistency', 1.0)) * loss_cons + lam_reg * loss_reg
                    loss.backward()
                    opt.step()
            except Exception as e:
                print(f'[TTT] failed: {e}')
            ttt_time_sum += (time.time() - t_ttt0)

        # TTA multi-view geometric augmentations -> post mask (start from raw)
        pred_mask_post = pred_mask_pre_raw.copy()
        if getattr(cfg, 'tta_views', 0) and cfg.tta_views > 0:
            import math
            import MinkowskiEngine as ME
            base_xyz = batch['xyz_original'].numpy()
            tta_masks = []
            tta_time_local = 0.0
            for _ in range(int(cfg.tta_views)):
                xyz = base_xyz.copy()
                deg = float(cfg.tta_rotate_deg)
                ax, ay, az = np.deg2rad(np.random.uniform(-deg, deg, size=3))
                Rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]], dtype=np.float32)
                Ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]], dtype=np.float32)
                Rz = np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]], dtype=np.float32)
                R = Rz @ Ry @ Rx
                xyz = xyz @ R.T
                s = 1.0 + np.random.uniform(-float(cfg.tta_scale), float(cfg.tta_scale))
                xyz = xyz * s
                if float(cfg.tta_jitter) > 0:
                    xyz = xyz + np.random.normal(scale=float(cfg.tta_jitter), size=xyz.shape).astype(np.float32)
                q, f, index, inv = ME.utils.sparse_quantize(xyz.astype(np.float32), xyz.astype(np.float32),
                                                            quantization_size=cfg.voxel_size,
                                                            return_index=True, return_inverse=True)
                xyz_voxel_t, feat_voxel_t = ME.utils.sparse_collate([q],[f])
                if isinstance(inv, np.ndarray):
                    v2p_t = torch.from_numpy(inv).long()
                elif torch.is_tensor(inv):
                    v2p_t = inv.long().cpu()
                else:
                    v2p_t = torch.as_tensor(inv, dtype=torch.long)
                batch_count_t = torch.tensor([0, xyz.shape[0]], dtype=torch.int64)
                tta_batch = {'xyz_voxel': xyz_voxel_t, 'feat_voxel': feat_voxel_t, 'v2p_index': v2p_t, 'batch_count': batch_count_t}
                t1 = time.time()
                _, pred_offset_t = eval_fn(tta_batch, model, category_ids=cond_ids,
                                          quantile=getattr(cfg, 'score_quantile', 0.95),
                                          score_method=getattr(cfg, 'score_method', 'quantile'))
                tta_time_local += (time.time() - t1)
                tta_mask = pred_offset_t.detach().cpu().abs().sum(dim=-1).numpy()
                tta_masks.append(tta_mask)
            if len(tta_masks) > 0:
                post_infer_time_sum += tta_time_local
                if getattr(cfg, 'tta_reduce', 'mean').lower() == 'max':
                    pred_mask_post = np.maximum.reduce([pred_mask_pre_raw] + tta_masks)
                else:
                    pred_mask_post = (pred_mask_pre_raw + np.sum(np.stack(tta_masks, axis=0), axis=0)) / (len(tta_masks) + 1.0)

        # post smoothing (single pass)
        if inds is not None:
            try:
                pred_mask_post = pred_mask_post[inds].mean(axis=1)
            except Exception as e:
                print(f'[smooth_knn] apply(post) failed: {e}')

        # object-level scores
        if getattr(cfg, 'score_method', 'mean') == 'mean':
            obj_score_pre = float(np.mean(pred_mask_pre))
            fused_obj_score = float(np.mean(pred_mask_post))
        elif getattr(cfg, 'score_method', 'mean') == 'max':
            obj_score_pre = float(np.max(pred_mask_pre))
            fused_obj_score = float(np.max(pred_mask_post))
        elif getattr(cfg, 'score_method', 'mean') == 'quantile':
            quantile = getattr(cfg, 'score_quantile', 0.95)
            obj_score_pre = float(np.quantile(pred_mask_pre, quantile))
            fused_obj_score = float(np.quantile(pred_mask_post, quantile))
        else:
            # 默认使用mean方式
            obj_score_pre = float(np.mean(pred_mask_pre))
            fused_obj_score = float(np.mean(pred_mask_post))

        # collect (post used by main metrics)
        pred_masks_pre.append(pred_mask_pre)
        pred_masks_post.append(pred_mask_post)
        pred_masks.append(pred_mask_post)
        y_list = batch['labels'].numpy().tolist()
        label_score_pre += list(zip(y_list, [obj_score_pre]))
        label_score_post += list(zip(y_list, [fused_obj_score]))
        label_score += list(zip(y_list, [fused_obj_score]))

        # optional: save .npz per sample for visualization
        if getattr(cfg, 'save_npz', False):
            try:
                os.makedirs(getattr(cfg, 'npz_dir', './results/vis'), exist_ok=True)
                out_npz = os.path.join(getattr(cfg, 'npz_dir', './results/vis'), f'{sample_name}.npz')
                np.savez(out_npz,
                         xyz=xyz_np.astype(np.float32),
                         score=pred_mask_post.astype(np.float32),
                         score_pre=pred_mask_pre.astype(np.float32),
                         gt=gt_this.astype(np.float32) if 'gt_this' in locals() else None,
                         label=np.array(y_list, dtype=np.int64))
            except Exception as e:
                print(f'[NPZ] save failed: {e}')



    # summary of conditioning decisions
    total_samples = len(fns)
    print(f"[Conditioning][Summary] used_cluster={used_pred_cluster}/{total_samples}, used_true={used_true_cond}, used_unconditional={used_unconditional}")

    # Cluster-Class confusion / consistency summary 
    if 'cluster_assigner_ready' in locals() and cluster_assigner_ready:
        valid_pairs = [(t, p) for t, p in zip(true_cats, pred_clusters) if (t is not None and p is not None and t >= 0 and p >= 0)]
        if len(valid_pairs) > 0:
            try:
                n_cls = int(num_classes) if (num_classes is not None and num_classes > 0) else (max(t for t, _ in valid_pairs) + 1)
            except Exception:
                n_cls = (max(t for t, _ in valid_pairs) + 1)
            try:
                n_clu = int(proto.shape[0]) if (proto is not None and hasattr(proto, 'shape')) else (max(p for _, p in valid_pairs) + 1)
            except Exception:
                n_clu = (max(p for _, p in valid_pairs) + 1)
            C = np.zeros((n_cls, n_clu), dtype=np.int64)
            for t, p in valid_pairs:
                if 0 <= t < n_cls and 0 <= p < n_clu:
                    C[t, p] += 1
            total_cp = int(C.sum()) #总样本数
            diag = int(np.sum([C[i, i] for i in range(min(n_cls, n_clu))])) if total_cp > 0 else 0 #对角元素和
            diag_frac = (diag / total_cp) if total_cp > 0 else float('nan') #对角元素占比
            print(f"[Cluster-Class][Confusion] valid={total_cp} classes={n_cls} clusters={n_clu}")
            print(f"[Cluster-Class][Index-consistency] diag_frac={diag_frac:.4f} (assuming cluster_id == class_id)")
            col_sums = C.sum(axis=0)
            for j in range(n_clu):
                if col_sums[j] == 0:
                    continue
                i = int(np.argmax(C[:, j]))
                hit = int(C[i, j])
                ratio = hit / int(col_sums[j])
                print(f"  cluster {j} -> class {i} hit={hit}/{int(col_sums[j])} ({ratio:.2%})")
                right_class = int(C[i, i]) #正确分类的样本数
                right_ratio = right_class / int(col_sums[j]) #正确分类的样本数占比
                print(f"cluster {j} -> class {i} right_class={right_class}/{int(col_sums[j])} ({right_ratio:.2%})")

            # optional save confusion matrix image
            if getattr(cfg, 'save_confmat', False):
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    out_path = getattr(cfg, 'confmat_out', '')
                    if not out_path:
                        out_path = os.path.join(cfg.logpath, 'confusion_matrix.png')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    plt.figure(figsize=(max(4, n_clu*0.6), max(3, n_cls*0.6)))
                    im = plt.imshow(C, cmap='Blues', aspect='auto')
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    plt.xlabel('Predicted cluster id')
                    plt.ylabel('True class id')
                    plt.title('Cluster vs Class Confusion')
                    # tick labels
                    plt.xticks(range(n_clu), range(n_clu))
                    plt.yticks(range(n_cls), range(n_cls))
                    # annotate counts (avoid clutter on large mats)
                    if n_cls * n_clu <= 400:
                        for ii in range(n_cls):
                            for jj in range(n_clu):
                                val = int(C[ii, jj])
                                if val > 0:
                                    plt.text(jj, ii, str(val), ha='center', va='center', fontsize=7, color='black')
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=200)
                    plt.close()
                    print(f"[ConfMat] saved to {out_path}")
                except Exception as e:
                    print(f"[ConfMat] save failed: {e}")
        else:
            print("[Cluster-Class][Confusion] no valid (true, predicted) pairs to summarize.")

    labels, scores = zip(*label_score)
    labels = np.array(labels) #提取真实标签
    scores = np.array(scores) #提取预测分数

    # define normalization function once for reuse
    def normalize_array(x, method):
        x = np.asarray(x)
        if method == 'minmax':
            x_min = np.min(x)
            x_max = np.max(x)
            d = (x_max - x_min) + 1e-12
            return (x - x_min) / d
        elif method == 'zscore':
            mu = np.mean(x)
            sd = np.std(x) + 1e-12
            return (x - mu) / sd
        elif method == 'mad':
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-12
            return (x - med) / mad
        else:
            x_min = np.min(x)
            x_max = np.max(x)
            d = (x_max - x_min) + 1e-12
            return (x - x_min) / d

    def compute_group_metrics(group_map):
        stats = {}
        for gid, idxs in sorted(group_map.items(), key=lambda x: x[0]):
            if gid < 0:
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
            stats[gid] = (auc_obj, auc_pt, ap_obj, ap_pt, len(idxs), pos_rate)
        macro = None
        valid = [v for v in stats.values() if not (np.isnan(v[0]) or np.isnan(v[1]))]
        if len(valid) > 0:
            mean_obj = float(np.nanmean([v[0] for v in valid]))
            mean_pt = float(np.nanmean([v[1] for v in valid]))
            mean_obj_ap = float(np.nanmean([v[2] for v in valid]))
            mean_pt_ap = float(np.nanmean([v[3] for v in valid]))
            macro = (mean_obj, mean_pt, mean_obj_ap, mean_pt_ap)
        return stats, macro

    # Print pre/post inference timing summary
    if sample_count > 0:
        methods = []
        if getattr(cfg, 'bn_tta', False): methods.append('bn-tta')
        if getattr(cfg, 'tta_views', 0) and cfg.tta_views > 0: methods.append(f'geom({int(cfg.tta_views)})')
        if getattr(cfg, 'ttt_enable', False) and int(getattr(cfg, 'ttt_steps', 0)) > 0: methods.append('ttt')
        if getattr(cfg, 'proto_ema', False): methods.append('proto-ema')
        methods_str = ','.join(methods) if methods else 'none'
        print(f"[TTA] methods={methods_str}")
        print(f"[TTA-Time] pre_infer_total={pre_infer_time_sum:.4f}s, per_sample={pre_infer_time_sum/sample_count:.4f}s")
        if post_infer_time_sum > 0:
            print(f"[TTA-Time] geom_infer_total={post_infer_time_sum:.4f}s, per_sample={post_infer_time_sum/sample_count:.4f}s")
        if ttt_time_sum > 0:
            print(f"[TTA-Time] ttt_total={ttt_time_sum:.4f}s, per_sample={ttt_time_sum/sample_count:.4f}s")
        if bn_tta_time_total > 0:
            print(f"[TTA-Time] bn_tta_refresh_total={bn_tta_time_total:.4f}s")

    # If any TTA is enabled, also report pre/post global metrics for comparison
    if (getattr(cfg, 'tta_views', 0) and cfg.tta_views > 0) or getattr(cfg, 'ttt_enable', False) or getattr(cfg, 'bn_tta', False) or getattr(cfg, 'proto_ema', False):
        def compute_global(labels_, scores_, masks_):
            auc_roc_ = safe_auc_roc(np.array(labels_), np.array(scores_))
            auc_pr_ = safe_ap(np.array(labels_), np.array(scores_))
            point_all = np.concatenate(masks_, axis=0)
            p_min = np.min(point_all)
            p_max = np.max(point_all)
            denom_ = (p_max - p_min) + 1e-12
            point_all = (point_all - p_min) / denom_
            gt_all_ = np.concatenate(gt_masks, axis=0)
            point_auc_roc_ = safe_auc_roc(gt_all_, point_all)
            point_auc_pr_ = safe_ap(gt_all_, point_all)
            return auc_roc_, auc_pr_, point_auc_roc_, point_auc_pr_
        

        if len(label_score_pre) == len(label_score_post) == len(pred_masks_pre) == len(pred_masks_post) and len(label_score_post) > 0:
            labels_pre_, scores_pre_ = zip(*label_score_pre)
            labels_post_, scores_post_ = zip(*label_score_post)
            g_obj_auc_pre, g_obj_ap_pre, g_pt_auc_pre, g_pt_ap_pre = compute_global(labels_pre_, scores_pre_, pred_masks_pre)
            g_obj_auc_post, g_obj_ap_post, g_pt_auc_post, g_pt_ap_post = compute_global(labels_post_, scores_post_, pred_masks_post)
            print(f'[Global-preTTA] object AUC-ROC: {g_obj_auc_pre}, point AUC-ROC: {g_pt_auc_pre}, object AUCP-PR: {g_obj_ap_pre}, point AUCP-PR: {g_pt_ap_pre}')
            print(f'[Global-postTTA] object AUC-ROC: {g_obj_auc_post}, point AUC-ROC: {g_pt_auc_post}, object AUCP-PR: {g_obj_ap_post}, point AUCP-PR: {g_pt_ap_post}')
            
            # Assign global metrics for CSV export
            auc_roc = g_obj_auc_post
            auc_pr = g_obj_ap_post
            point_auc_roc = g_pt_auc_post
            point_auc_pr = g_pt_ap_post
        else:
            # If TTA is not enabled, compute global metrics once
            auc_roc = safe_auc_roc(labels, scores)
            auc_pr = safe_ap(labels, scores)
            point_pred_all = np.concatenate(pred_masks, axis=0)
            # global min-max
            p_min = np.min(point_pred_all)
            p_max = np.max(point_pred_all)
            denom = (p_max - p_min) + 1e-12
            point_pred_all = (point_pred_all - p_min) / denom
            gt_all = np.concatenate(gt_masks, axis=0)
            point_auc_roc = safe_auc_roc(gt_all, point_pred_all)
            point_auc_pr = safe_ap(gt_all, point_pred_all)
            if getattr(cfg, 'print_pos_rate', False):
                pos_rate_global = float(np.mean(gt_all))
                print(f'[Global] pos_rate={pos_rate_global:.6f}') #全局异常点所占比例
            print(f'[Global] object AUC-ROC: {auc_roc}, point AUC-ROC: {point_auc_roc}, object AUCP-PR: {auc_pr}, point AUCP-PR: {point_auc_pr}')

    # Global AUPRO (optional)
    if getattr(cfg, 'compute_aupro', False):
        # normalize each sample's scores using global min-max for consistency
        global_min, global_max = np.min(np.concatenate(pred_masks)), np.max(np.concatenate(pred_masks))
        denom2 = (global_max - global_min) + 1e-12
        scores_norm_list = [ (pm - global_min) / denom2 for pm in pred_masks ]
        try:
            from tools.aupro import aupro_curve
            fpr_grid, pro_curve, aupro_area, aupro_norm = aupro_curve(
                scores_norm_list, gt_masks, xyz_list=xyz_list,
                fpr_max=getattr(cfg, 'aupro_fpr_max', 0.3),
                num_fpr=getattr(cfg, 'aupro_points', 31),
                region_knn=getattr(cfg, 'region_knn', 16),
            )
            print(f'[Global] AUPRO@FPR<= {getattr(cfg, "aupro_fpr_max", 0.3)} : area={aupro_area:.6f}, normalized={aupro_norm:.6f}')
        except Exception as e:
            print(f'[AUPRO] computation failed: {e}')



    # cluster-based normalization and per-cluster/per-category metrics
    if assigner is not None and proto is not None:
        from collections import defaultdict
        import csv

        norm_type = getattr(cfg, 'cluster_norm_type', 'minmax')
        # group by predicted cluster
        groups = defaultdict(list) #groups字典存储每个聚类簇的样本索引： key为聚类簇ID，value为该簇的样本索引列表
        for idx, cid in enumerate(pred_clusters):
            groups[cid].append(idx)

        print(f"\n[Per-Cluster metrics with cluster-wise {norm_type} normalization]")
        cluster_stats, macro_cluster = compute_group_metrics(groups)
        for cid, v in cluster_stats.items():
            if getattr(cfg, 'print_pos_rate', False):
                print(f'  [cluster {cid}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]} pos_rate={v[5]:.6f}' if v[5] is not None else f'  [cluster {cid}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}')
            else:
                print(f'  [cluster {cid}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}')
        if macro_cluster is not None:
            print(f'  [cluster-macro] objAUC={macro_cluster[0]} ptAUC={macro_cluster[1]} objAP={macro_cluster[2]} ptAP={macro_cluster[3]}')

        # per-true-category metrics (using true category parsed from path)
        cat_groups = defaultdict(list)
        for idx, tc in enumerate(true_cats):
            cat_groups[tc].append(idx)
        print(f"\n[Per-Category metrics (by true category) with category-wise {norm_type} normalization]")
        cat_stats, macro_cat = compute_group_metrics(cat_groups)
        for tc, v in cat_stats.items():
            if getattr(cfg, 'print_pos_rate', False):
                print(f'  [cat {tc}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]} pos_rate={v[5]:.6f}' if v[5] is not None else f'  [cat {tc}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}')
            else:
                print(f'  [cat {tc}] N={v[4]} objAUC={v[0]} ptAUC={v[1]} objAP={v[2]} ptAP={v[3]}')
        if macro_cat is not None:
            print(f'  [category-macro] objAUC={macro_cat[0]} ptAUC={macro_cat[1]} objAP={macro_cat[2]} ptAP={macro_cat[3]}')

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
                # Ensure variables are defined
                if 'auc_roc' not in locals():
                    auc_roc = safe_auc_roc(labels, scores)
                    print("auc_roc is not in locals, recomputing this metric")
                if 'point_auc_roc' not in locals():
                    point_pred_all = np.concatenate(pred_masks, axis=0)
                    p_min = np.min(point_pred_all)
                    p_max = np.max(point_pred_all)
                    denom = (p_max - p_min) + 1e-12
                    point_pred_all = (point_pred_all - p_min) / denom
                    gt_all = np.concatenate(gt_masks, axis=0)
                    point_auc_roc = safe_auc_roc(gt_all, point_pred_all)
                    print("point_auc_roc is not in locals, recomputing this metric")
                if 'auc_pr' not in locals():
                    auc_pr = safe_ap(labels, scores)
                    print("auc_pr is not in locals, recomputing this metric")
                if 'point_auc_pr' not in locals():
                    point_auc_pr = safe_ap(gt_all, point_pred_all)
                    print("point_auc_pr is not in locals, recomputing this metric")
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

            print(f'[Metrics CSV] saved to {csv_path}')
        except Exception as e:
            print(f'[Metrics CSV] failed to save: {e}')
    # restore stdout and persist md capture if requested (normal path)
    if md_capture is not None:
        sys.stdout = orig_stdout
        try:
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_capture.getvalue())
            print(f'[Metrics MD] saved to {md_path}')
        except Exception as e:
            print(f'[Metrics MD] failed to save: {e}')
    else:
        sys.stdout = orig_stdout


if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    eval(cfg)