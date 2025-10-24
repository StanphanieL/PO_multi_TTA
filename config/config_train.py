import argparse

# config para
def get_parser():

    # # Train config
    parser = argparse.ArgumentParser(description='3D anomaly detection')
    parser.add_argument('--task', type=str, default='train', help='task: train, eval or contrastive')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=1001, help='Total epoch')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset (DataLoader workers)')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=1, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str, default='./log/ashtray0/', help='path to save logs')
    parser.add_argument('--validation', type=bool, default=False, help='Whether to verify the validation set')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', help='datasets')
    parser.add_argument('--category', type=str, default='ashtray0', help='category for single-class training')
    parser.add_argument('--categories', type=str, default='all', help='multi-categories training, e.g., "ashtray0,bottle0" or "all"')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for single GPU')
    parser.add_argument('--data_repeat', type=int, default=10, help='repeat the date for each epoch')
    parser.add_argument('--mask_num', type=int, default=64)

    # DataLoader performance options
    parser.add_argument('--pin_memory', action='store_true', help='enable DataLoader pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch_factor (workers>0)')

    # I/O cache options
    parser.add_argument('--cache_io', action='store_true', help='cache heavy I/O (meshes, pcd/gt coords) to .npz for faster loading')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='directory to store cached arrays')

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

    # class-conditional offset head (stage 2)
    parser.add_argument('--class_embed_dim', type=int, default=32, help='class embedding dim for conditional offset head; set 0 to disable')
    parser.add_argument('--conditional_mode', type=str, default='film', help='conditional mode for offset head: concat or film') #将类别信息整合到偏移头中的方式
    # warm start from contrastive backbone (stage 1)
    parser.add_argument('--contrastive_backbone', type=str, default='',help='path to contrastive ckpt to init backbone for stage 2') #若不指定，则第二阶段backbone不做预加载，直接随机初始化
    

    # #contrastive learning params (stage 1)
    parser.add_argument('--contrastive', action='store_true', help='enable contrastive training pipeline (stage 1)')
    parser.add_argument('--proj_dim', type=int, default=128, help='projection head output dim for contrastive') #对比学习的投影头输出维度
    parser.add_argument('--temperature', type=float, default=0.05, help='temperature for SupCon/InfoNCE') #较小的温度会使模型更关注难样本
    parser.add_argument('--proto_m', type=float, default=0.9, help='prototype momentum for moving average updates')
    parser.add_argument('--proto_loss_weight', type=float, default=0.2, help='weight for prototype NCE loss')

    # embedding visualization saving (stage 1)
    parser.add_argument('--emb_save_samples', type=int, default=2048, help='max number of embeddings to save per epoch for visualization')
    parser.add_argument('--emb_save_every', type=int, default=100, help='save embedding snapshot every N epochs (stage 1)')

    # contrastive best checkpoint saving
    parser.add_argument('--contrastive_best_metric', type=str, default='loss', help='metric to save best contrastive ckpt: acc or loss')

    # stage-2 best checkpoint saving
    parser.add_argument('--stage2_best_metric', type=str, default='loss', help='metric to save best stage-2 ckpt (loss only)')

    args = parser.parse_args()
    return args
