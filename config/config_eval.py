import argparse

# config para
def get_parser():

    # # Eval config
    parser = argparse.ArgumentParser(description='3D anomaly detection')
    parser.add_argument('--task', type=str, default='eval', help='task: eval, contrastive_eval')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=1001, help='Total epoch')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=1, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str, default='./log/ashtray0/', help='path to save logs')
    parser.add_argument('--validation', type=bool, default=False, help='Whether to verify the validation set')
    parser.add_argument('--checkpoint_name', type=str, default='', help='checkpoint name')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', help='datasets')
    parser.add_argument('--category', type=str, default='', help='categories for each class (single)')
    parser.add_argument('--categories', type=str, default='', help='multi-categories for cluster eval, e.g., "all"')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for single GPU')
    parser.add_argument('--data_repeat', type=int, default=100, help='repeat the date for each epoch')
    parser.add_argument('--mask_num', type=int, default=32)

    # #Adjust learning rate
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--step_epoch', type=int, default=10, help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.5, help='Learning rate decay: lr = lr * multiplier')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for SGD')

    # #model parameter
    parser.add_argument('--voxel_size', type=float, default=0.03, help='voxel size')
    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--out_channels', type=int, default=32, help='backbone feat channels')
    parser.add_argument('--class_embed_dim', type=int, default=32, help='class embedding dim used by conditional offset head at eval; must match training')
    parser.add_argument('--conditional_mode', type=str, default='concat', help='conditional mode for offset head at eval: concat or film')

    # contrastive eval params
    parser.add_argument('--contrastive_eval', action='store_true', help='run cluster evaluation with contrastive model')
    parser.add_argument('--proj_dim', type=int, default=128, help='projection head output dim for contrastive eval')
    parser.add_argument('--cluster_norm', action='store_true', help='use contrastive cluster assigner to normalize scores per-cluster and report per-cluster/category metrics in anomaly eval')
    parser.add_argument('--contrastive_ckpt', type=str, default='', help='path to contrastive checkpoint (with prototypes) for cluster assigner')
    parser.add_argument('--cluster_norm_type', type=str, default='minmax', help='cluster/category normalization for scores: minmax, zscore, mad')
    parser.add_argument('--metrics_csv', type=str, default='', help='optional path to save per-cluster/category metrics CSV')
    parser.add_argument('--eval_category_only', type=str, default='', help='evaluate unified model on a single category (subset test to this category only)')
    parser.add_argument('--metrics_md', type=str, default='', help='optional path to save console outputs (metrics/timing) into a markdown file')

    # additional evaluation options
    parser.add_argument('--point_macro_ap', action='store_true', help='compute per-sample point-level AP and report macro average')
    parser.add_argument('--print_pos_rate', action='store_true', help='print positive rate (fraction of anomalous points) for global/cluster/category')
    parser.add_argument('--sample_norm', action='store_true', help='normalize scores within each sample (instead of cluster/category) when computing per-sample macro AP')
    parser.add_argument('--smooth_knn', type=int, default=0, help='kNN smoothing for point scores (k=0 disables)')

    # AUPRO options
    parser.add_argument('--compute_aupro', action='store_true', help='compute AUPRO (PRO area under FPR in a range)')
    parser.add_argument('--aupro_fpr_max', type=float, default=0.3, help='FPR upper bound for AUPRO (e.g., 0.3 for 0~30%)')
    parser.add_argument('--aupro_points', type=int, default=31, help='number of FPR points in [0, aupro_fpr_max]')
    parser.add_argument('--region_knn', type=int, default=16, help='kNN for region connectivity among positive points')

    # Test-time augmentation (multi-view) options
    parser.add_argument('--tta_views', type=int, default=0, help='number of TTA geometric views (0 disables)')
    parser.add_argument('--tta_rotate_deg', type=float, default=5.0, help='max absolute rotation (degrees) around each axis for TTA')
    parser.add_argument('--tta_scale', type=float, default=0.05, help='uniform scale jitter range [+/- tta_scale] for TTA')
    parser.add_argument('--tta_jitter', type=float, default=0.002, help='Gaussian jitter sigma for TTA (applied per point)')
    parser.add_argument('--tta_reduce', type=str, default='mean', help='reduce fused TTA scores: mean or max')

    args = parser.parse_args()
    return args
