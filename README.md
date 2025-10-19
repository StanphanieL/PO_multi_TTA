# PO3AD: Unified Multi-Class 3D Point-Cloud Anomaly Detection

A unified anomaly detection framework for multiple 3D categories with conditional offset heads and robust evaluation. It integrates contrastive clustering, conditional regression (concat / FiLM), per-cluster/category normalization, extensive TTA (geometric / temperature) and light adaptation (Prototype-EMA, AdaBN, TTT/CTTA, Pseudo-label).

## Features
- Unified multi-class model: one PONet for all categories.
- Conditional offset head: concat or FiLM modulation by class embedding.
- Contrastive clustering: prototypes + nearest-prototype cluster assignment at test time.
- Fair metrics: per-cluster/category normalization (minmax / zscore / mad), macro averages.
- Rich TTA/adaptation:
  - Geometric TTA (multi-view) with score fusion.
  - Temperature TTA (post-processing) on object / point scores.
  - Prototype EMA update (confidence-gated).
  - BN-TTA (AdaBN) to refresh BN running stats.
  - TTT (consistency / entropy, last head layer).
  - CTTA (class embedding light adaptation).
  - Pseudo-label adaptation (high-confidence regions).
- Reporting: global + per-cluster/category metrics, pre/post TTA comparisons, CSV export, full log to .md.

## Repo Structure (core)
- `network/PO3AD.py`: PONet (MinkUNet backbone + conditional offset head), eval forward.
- `network/contrast.py`: POContrast, projection head, prototype memory.
- `network/Mink.py`: MinkowskiEngine UNet arch.
- `datasets/AnomalyShapeNet` and `datasets/Real3D`: preprocessing, loaders (`cat2id`, train/test lists).
- `tools/log.py`: logging and checkpoints; `tools/aupro.py`: AUPRO computation.
- `config/config_train.py`, `config/config_eval.py`: training/eval configs and TTA/adaptation flags.
- `train.py`: contrastive (stage-1) and offset regression (stage-2) training.
- `eval.py`: full evaluation (cluster prediction, conditioning, TTA/adaptation, metrics, export).

## Installation (brief)
- Python 3.8+, PyTorch (CUDA), MinkowskiEngine 0.5.x.
- Install project deps; ensure datasets are placed under `datasets/AnomalyShapeNet` and `datasets/Real3D`.

## Datasets
- AnomalyShapeNet: expected GT under `datasets/AnomalyShapeNet/dataset/pcd/<cat>/GT/*.txt`
- Real3D: expected GT under `datasets/Real3D/Real3D-AD-PCD/<cat>/gt/*.txt`
- The project parses `cat` from the input path (3rd segment from the end) and maps via `dataset.cat2id`.

## Training

### Stage-1: Contrastive with Prototypes
Trains POContrast and produces class prototypes used for cluster assignment.
- Typical flags (see `config_train.py`): temperature, proto momentum, epochs, etc.
- Best/Latest checkpoints saved to the specified `logpath`.

### Stage-2: Offset Regression (Unified Model)
Trains PONet with conditional head (concat / FiLM).
- Optionally load stage-1 backbone weights to initialize (`--contrastive_backbone`).
- Configure conditional embedding via `--class_embed_dim`, `--conditional_mode`.

## Evaluation

### Core flags
- Dataset/model:
  - `--dataset {AnomalyShapeNet,Real3D}`
  - `--class_embed_dim 32` `--conditional_mode {concat,film}`
  - `--logpath LOG_DIR/` `--checkpoint_name best.pth`
- Contrastive cluster assigner:
  - `--cluster_norm` `--contrastive_ckpt PATH_TO_STAGE1/best.pth` `--proj_dim 128`
- Score normalization and export:
  - `--cluster_norm_type {minmax,zscore,mad}`
  - `--metrics_csv out.csv` (per-cluster/category + global)
  - `--point_macro_ap` `--sample_norm` `--print_pos_rate`
- Category filter:
  - `--eval_category_only <cat_name>` (subset to one category)

### TTA and Light Adaptation
- Geometric TTA:
  - `--tta_mode {none,geom,temp,geom+temp}` (geom = multi-view; temp = temperature scaling)
  - Geom: `--tta_views K --tta_rotate_deg 5 --tta_scale 0.05 --tta_jitter 0.002 --tta_reduce {mean,max}`
  - Temp: `--tta_temp_type {pow,logistic} --tta_gamma 0.9 --tta_temp 1.0 --tta_apply_scope {object,point,both}`
- Prototype EMA:
  - `--proto_ema --proto_ema_m 0.99 --proto_ema_metric {softmax,gap} --proto_ema_tau 0.8`
- BN-TTA (AdaBN):
  - `--bn_tta --bn_tta_samples 16` (refresh BN stats with a few test samples)
- TTT (last head layer):
  - `--ttt_enable --ttt_steps 1~5 --ttt_lr 1e-4 --ttt_consistency 1.0 --ttt_reg 1e-3 --ttt_entropy 0`
  - Weak aug: `--ttt_weak_rotate_deg 2 --ttt_weak_jitter 0.001`
- CTTA (class embedding light adaptation):
  - `--ctta_enable --ctta_steps 1~5 --ctta_lr 5e-4 --ctta_consistency 1.0 --ctta_reg 1e-3 --ctta_entropy 0`
  - Weak aug: `--ctta_weak_rotate_deg 2 --ctta_weak_jitter 0.001`
- Pseudo-label adaptation:
  - `--pl_enable --pl_steps 1~5 --pl_lr 5e-4 --pl_scope {point,object}`
  - `--pl_norm {sample-minmax,sample-zscore} --pl_tau_hi 0.9 --pl_tau_lo 0.1 --pl_weight 1.0 --pl_reg 1e-3`

### Example: AnomalyShapeNet with geometric TTA
```
python eval.py --dataset AnomalyShapeNet \
  --class_embed_dim 32 --conditional_mode film \
  --cluster_norm --contrastive_ckpt ./log/ashape_contrast/best.pth --proj_dim 128 \
  --cluster_norm_type mad --metrics_csv ./results/ashape_metrics.csv \
  --logpath ./log/ashape_po3ad/ --checkpoint_name best.pth \
  --tta_mode geom --tta_views 4 --smooth_knn 16 --point_macro_ap --sample_norm --print_pos_rate
```

### Example: EMA only
```
python eval.py --dataset AnomalyShapeNet \
  --class_embed_dim 32 --conditional_mode film \
  --cluster_norm --contrastive_ckpt ./log/ashape_contrast/best.pth --proj_dim 128 \
  --logpath ./log/ashape_po3ad/ --checkpoint_name best.pth \
  --proto_ema --proto_ema_m 0.99 --proto_ema_metric softmax --proto_ema_tau 0.8
```

### Example: Real3D with geom+temp + AdaBN
```
python eval.py --dataset Real3D \
  --class_embed_dim 32 --conditional_mode film \
  --cluster_norm --contrastive_ckpt ./log/real3d_contrast/best.pth --proj_dim 128 \
  --cluster_norm_type mad --metrics_csv ./results/real3d_metrics.csv \
  --logpath ./log/real3d_po3ad/ --checkpoint_name best.pth \
  --tta_mode geom+temp --tta_views 3 --tta_temp_type pow --tta_gamma 0.9 --tta_apply_scope both \
  --bn_tta --bn_tta_samples 16 --smooth_knn 16 --point_macro_ap --sample_norm --print_pos_rate
```

## Outputs
- Console:
  - Global metrics (+ pre/post TTA if geometric TTA enabled).
  - Per-Cluster / Per-Category metrics with group-wise normalization and macro.
  - Cluster-class consistency (diag_frac) and conditioning fallback summary.
  - TTA timing; optional pseudo/adaptation/EMA summaries.
- Files:
  - CSV metrics at `--metrics_csv`; full console saved to `.md` via `--metrics_md`.

## Tips
- 簇一致性较差时：启用 Prototype-EMA（或一次性簇↔类映射）、BN-TTA；必要时软条件化（可在后续版本加入）。
- 时延控制：多视图打包成单批次、K取2~3、使用median/trimmed-mean融合、弱增广幅度小。
- 指标口径：Global（全局min-max）；分簇/分类别（组内归一）。对比时注意口径差异。

## License
See `LICENSE`.
