import os
import argparse
import numpy as np
import open3d as o3d
from matplotlib import cm


def main():
    parser = argparse.ArgumentParser(description='Visualize anomaly .npz (xyz, score) as colored point cloud')
    parser.add_argument('--npz', type=str, required=True, help='path to .npz saved by eval (--save_npz)')
    parser.add_argument('--out_ply', type=str, default='', help='optional path to save colored PLY')
    parser.add_argument('--out_gt_ply', type=str, default='', help='optional path to save GT-colored PLY if gt exists')
    parser.add_argument('--colormap', type=str, default='jet', help='matplotlib cmap name: jet, turbo, viridis, etc.')
    parser.add_argument('--field', type=str, default='score', help='score field: score (post-TTA) or score_pre')
    parser.add_argument('--normalize', type=str, default='minmax', help='minmax or zscore normalization before coloring')
    parser.add_argument('--show', action='store_true', help='open interactive viewer')
    parser.add_argument('--show_gt', action='store_true', help='also show GT mask if available')
    args = parser.parse_args()

    data = np.load(args.npz)
    xyz = data['xyz']
    s = data[args.field] if args.field in data.files else data['score']

    # normalize
    s = s.astype(np.float32).reshape(-1)
    if args.normalize == 'zscore':
        mu, sd = float(np.mean(s)), float(np.std(s) + 1e-12)
        s = (s - mu) / sd
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # colorize
    cmap = cm.get_cmap(args.colormap)
    colors = np.asarray(cmap(s))[:, :3].astype(np.float32)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # save PLY
    out_ply = args.out_ply or os.path.splitext(args.npz)[0] + '.ply'
    os.makedirs(os.path.dirname(out_ply) or '.', exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pc)
    print(f'Saved colored PLY: {out_ply}  | points={xyz.shape[0]}  cmap={args.colormap} field={args.field}')

    # optional GT ply / viewer
    if 'gt' in data.files and data['gt'] is not None:
        gt = (data['gt'].reshape(-1) > 0.5).astype(np.bool_)
        colors_gt = np.tile(np.array([[0.8, 0.8, 0.8]], dtype=np.float32), (xyz.shape[0], 1))
        colors_gt[gt] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red for anomalies
        pc_gt = o3d.geometry.PointCloud()
        pc_gt.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pc_gt.colors = o3d.utility.Vector3dVector(colors_gt.astype(np.float64))
        out_gt = args.out_gt_ply or os.path.splitext(args.npz)[0] + '_gt.ply'
        o3d.io.write_point_cloud(out_gt, pc_gt)
        print(f'Saved GT PLY: {out_gt}')
        if args.show_gt and args.show:
            o3d.visualization.draw_geometries([pc_gt])

    if args.show:
        o3d.visualization.draw_geometries([pc])


if __name__ == '__main__':
    main()
