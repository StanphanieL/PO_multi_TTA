import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Single-stage training for PO3AD (contrastive + offset)')
    # base
    parser.add_argument('--task', type=str, default='single_stage_train', help='single_stage_train or single_stage_eval')
    parser.add_argument('--manual_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--logpath', type=str, default='./log/single_stage/')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--pretrain', type=str, default='')

    # dataset
    parser.add_argument('--dataset', type=str, default='AnomalyShapeNet', help='{AnomalyShapeNet,Real3D}')
    parser.add_argument('--category', type=str, default='ashtray0')
    parser.add_argument('--categories', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_repeat', type=int, default=10)
    parser.add_argument('--mask_num', type=int, default=64)
    parser.add_argument('--voxel_size', type=float, default=0.03)

    # model
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--class_embed_dim', type=int, default=32)
    parser.add_argument('--conditional_mode', type=str, default='film', help='concat|film')
    parser.add_argument('--proj_dim', type=int, default=128)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_epoch', type=int, default=10)
    parser.add_argument('--multiplier', type=float, default=0.5)

    # losses (course scheduling)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--proto_m', type=float, default=0.95)
    parser.add_argument('--proto_loss_weight', type=float, default=0.2)
    parser.add_argument('--lambda_dir', type=float, default=1.0, help='direction cosine loss weight in offset')
    parser.add_argument('--warmup_pct', type=float, default=0.2, help='first pct epochs emphasis on contrastive (w_off rises from 0)')
    parser.add_argument('--decay_pct', type=float, default=0.6, help='next pct epochs shift weight to offset')
    parser.add_argument('--w_con_min', type=float, default=0.1)

    # dataloader perf
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--cache_io', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='./cache')

    return parser
