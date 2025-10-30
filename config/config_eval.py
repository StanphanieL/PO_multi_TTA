import argparse

# config para
def get_parser():

    # # Eval config
    parser = argparse.ArgumentParser(description='3D anomaly detection')
    parser.add_argument('--task', type=str, default='eval', help='task: eval, contrastive_eval')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed to produce')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset (DataLoader workers)')
    parser.add_argument('--logpath', type=str, default='./log/po3ad_cond_film_ashape_all_test2/best.pth', help='path to save logs')
    # parser.add_argument('--validation', type=bool, default=False, help='Whether to verify the validation set')
    parser.add_argument('--checkpoint_name', type=str, default='', help='checkpoint name')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--metrics_md', type=str, default='', help='optional path to save console outputs (metrics/timing) into a markdown file')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', help='datasets')
    parser.add_argument('--category', type=str, default='', help='categories for each class (single)')
    parser.add_argument('--categories', type=str, default='', help='multi-categories for cluster eval, e.g., "all"')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for single GPU')
    parser.add_argument('--mask_num', type=int, default=32)
    parser.add_argument('--data_repeat', type=int, default=10, help='repeat the date for each epoch')
    parser.add_argument('--eval_category_only', type=str, default='', help='evaluate unified model on a single category (subset test to this category only)')

    # DataLoader performance options
    parser.add_argument('--pin_memory', action='store_true', help='enable DataLoader pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch_factor (workers>0)')

    # I/O cache options
    parser.add_argument('--cache_io', action='store_true', help='cache heavy I/O (meshes, pcd/gt coords) to .npz for faster loading')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='directory to store cached arrays')

    # #model parameter
    parser.add_argument('--voxel_size', type=float, default=0.03, help='voxel size')
    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--out_channels', type=int, default=32, help='backbone feat channels')
    parser.add_argument('--class_embed_dim', type=int, default=32, help='class embedding dim used by conditional offset head at eval; must match training')
    parser.add_argument('--conditional_mode', type=str, default='film', help='conditional mode for offset head at eval: concat or film')
    parser.add_argument('--conditioning_source', type=str, default='auto', help='conditioning source at eval: auto (cluster then true), true (always true category), cluster (predicted cluster), none (no conditioning)')

    # contrastive eval params
    parser.add_argument('--contrastive_eval', action='store_true', help='run cluster evaluation with contrastive model')
    parser.add_argument('--proj_dim', type=int, default=128, help='projection head output dim for contrastive eval')
    parser.add_argument('--cluster_norm', action='store_true', help='use contrastive cluster assigner to normalize scores per-cluster and report per-cluster/category metrics in anomaly eval')
    parser.add_argument('--contrastive_ckpt', type=str, default='./log/contrast_ashape_all_test2/best.pth', help='path to contrastive checkpoint (with prototypes) for cluster assigner')
    parser.add_argument('--cluster_norm_type', type=str, default='minmax', help='cluster/category normalization for scores: minmax, zscore, mad')
    parser.add_argument('--metrics_csv', type=str, default='./result/metrics_debug.csv', help='optional path to save per-cluster/category metrics CSV')
   
    
    
    # confusion matrix visualization
    parser.add_argument('--save_confmat', action='store_true', help='save cluster-vs-class confusion matrix image when cluster assigner is used')
    parser.add_argument('--confmat_out', type=str, default='./results/confmat.png', help='output path of the confusion matrix image (png)')

    # additional evaluation options
    parser.add_argument('--print_pos_rate', action='store_true', help='print positive rate (fraction of anomalous points) for global/cluster/category')
    parser.add_argument('--smooth_knn', type=int, default=16, help='kNN smoothing for point scores (k=0 disables)')

    # AUPRO options
    parser.add_argument('--compute_aupro', action='store_true', help='compute AUPRO (PRO area under FPR in a range)')
    parser.add_argument('--aupro_fpr_max', type=float, default=0.3, help='FPR upper bound for AUPRO (e.g., 0.3 for 0~30%)')
    parser.add_argument('--aupro_points', type=int, default=31, help='number of FPR points in [0, aupro_fpr_max]')
    parser.add_argument('--region_knn', type=int, default=16, help='kNN for region connectivity among positive points')

    # Visualization saving
    parser.add_argument('--save_npz', action='store_true', help='save per-sample xyz and scores (.npz) for visualization')
    parser.add_argument('--npz_dir', type=str, default='./results/vis', help='directory to save .npz files')

    # Test-time augmentation (multi-view) options
    parser.add_argument('--tta_views', type=int, default=0, help='number of TTA geometric views (0 disables)')
    parser.add_argument('--tta_rotate_deg', type=float, default=5.0, help='max absolute rotation (degrees) around each axis for TTA')
    parser.add_argument('--tta_scale', type=float, default=0.05, help='uniform scale jitter range [+/- tta_scale] for TTA')
    parser.add_argument('--tta_jitter', type=float, default=0.002, help='Gaussian jitter sigma for TTA (applied per point)')
    parser.add_argument('--tta_reduce', type=str, default='mean', help='reduce fused TTA scores: mean or max')

    # BN-TTA (AdaBN)
    parser.add_argument('--bn_tta', action='store_true', help='enable BN-TTA (AdaBN) to refresh BN stats with a few test samples')
    parser.add_argument('--bn_tta_samples', type=int, default=16, help='number of samples to run through the model to refresh BN stats')

    # Prototype EMA during eval (cluster assigner required)
    parser.add_argument('--proto_ema', action='store_true', help='enable prototype EMA update during evaluation')
    parser.add_argument('--proto_ema_m', type=float, default=0.99, help='prototype EMA momentum (higher=slower update)')
    parser.add_argument('--proto_ema_tau', type=float, default=0.8, help='confidence threshold on predicted cluster prob to update prototype')

    # Test-Time Training (light adaptation on last head)
    parser.add_argument('--ttt_enable', action='store_true', help='enable light test-time training (adapt last head)')
    parser.add_argument('--ttt_steps', type=int, default=0, help='number of ttt optimization steps per sample (0 disables)')
    parser.add_argument('--ttt_lr', type=float, default=1e-4, help='learning rate for ttt optimizer')
    parser.add_argument('--ttt_consistency', type=float, default=1.0, help='weight of consistency loss between weak view and original')
    parser.add_argument('--ttt_reg', type=float, default=1e-3, help='L2 weight decay on adapted params during ttt')
    parser.add_argument('--ttt_entropy', type=float, default=0.0, help='optional entropy minimization weight (on score distribution)')
    parser.add_argument('--ttt_weak_rotate_deg', type=float, default=2.0, help='weak rotation (deg) for ttt view')
    parser.add_argument('--ttt_weak_jitter', type=float, default=0.001, help='weak jitter sigma for ttt view')

    # Sample-level anomaly score calculation method
    parser.add_argument('--score_quantile', type=float, default=0.95, help='quantile for sample-level anomaly score calculation (0.0-1.0), e.g., 0.95 means using 95th percentile')
    parser.add_argument('--score_method', type=str, default='quantile', choices=['mean', 'max', 'quantile'], help='method for sample-level anomaly score calculation: mean, max, or quantile')

    args = parser.parse_args()
    return args