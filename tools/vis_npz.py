import os
import glob
import argparse
import numpy as np
import open3d as o3d
import matplotlib


def _load_scores_from_npz(path: str, field: str) -> np.ndarray:
    d = np.load(path)
    if field in d.files:
        return d[field].astype(np.float32).reshape(-1)
    return d['score'].astype(np.float32).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description='Visualize anomaly .npz (xyz, score) as colored point cloud')
    parser.add_argument('--npz', type=str, required=False, help='path to a single .npz saved by eval (--save_npz)')
    parser.add_argument('--npz_dir', type=str, default='', help='if set, process all .npz in this directory (non-recursive)')
    parser.add_argument('--out_ply', type=str, default='', help='optional path to save colored PLY')
    parser.add_argument('--out_gt_ply', type=str, default='', help='optional path to save GT-colored PLY if gt exists')
    parser.add_argument('--out_mask_ply', type=str, default='', help='optional path to save binary-mask PLY (thresholded)')
    parser.add_argument('--colormap', type=str, default='jet', help='matplotlib cmap name: jet, turbo, viridis, etc.')
    parser.add_argument('--field', type=str, default='score', help='score field: score (post-TTA) or score_pre')
    # normalization options
    parser.add_argument('--normalize', type=str, default='minmax', help='minmax or zscore normalization before coloring')
    parser.add_argument('--norm_scope', type=str, default='quantile', help='sample | global | quantile')
    parser.add_argument('--global_min', type=float, default=None, help='global min (used if norm_scope=global)')
    parser.add_argument('--global_max', type=float, default=None, help='global max (used if norm_scope=global)')
    parser.add_argument('--ref_dir', type=str, default='', help='optional dir of npz to compute global stats (field-based)')
    parser.add_argument('--q_low', type=float, default=0.90, help='low quantile for clipping (norm_scope=quantile)')
    parser.add_argument('--q_high', type=float, default=0.99, help='high quantile for clipping (norm_scope=quantile)')
    # thresholded mask options
    parser.add_argument('--mask_abs', type=float, default=-1.0, help='mask threshold in normalized [0,1]; <0 disables')
    parser.add_argument('--mask_topk_pct', type=float, default=-1.0, help='keep top-k percent points as anomalies; <0 disables')
    # display
    parser.add_argument('--show', action='store_true', help='open interactive viewer')
    parser.add_argument('--show_gt', action='store_true', help='also show GT mask if available')
    args = parser.parse_args()

    def process_one(npz_path: str):
        data = np.load(npz_path)
        xyz = data['xyz']
        s = data[args.field] if args.field in data.files else data['score']
        s_local = s.astype(np.float32).reshape(-1)

        # decide normalization bounds
        s_min, s_max = float(np.min(s_local)), float(np.max(s_local))
        if args.norm_scope == 'global':
            ref_dir = args.ref_dir
            if ref_dir:
                vals = []
                for f in glob.glob(os.path.join(ref_dir, '*.npz')):
                    try:
                        v = _load_scores_from_npz(f, args.field)
                        vals.append([float(np.min(v)), float(np.max(v))])
                    except Exception:
                        pass
                if len(vals) > 0:
                    s_min = min(v[0] for v in vals)
                    s_max = max(v[1] for v in vals)
            if args.global_min is not None:
                s_min = float(args.global_min)
            if args.global_max is not None:
                s_max = float(args.global_max)
        elif args.norm_scope == 'quantile':
            ref_dir = args.ref_dir
            if ref_dir:
                # compute global quantiles across dir
                all_samples = []
                for f in glob.glob(os.path.join(ref_dir, '*.npz')):
                    try:
                        v = _load_scores_from_npz(f, args.field)
                        all_samples.append(v.astype(np.float32))
                    except Exception:
                        pass
                if len(all_samples) > 0:
                    cat = np.concatenate(all_samples, axis=0)
                    s_min = float(np.quantile(cat, args.q_low))
                    s_max = float(np.quantile(cat, args.q_high))
            else:
                # per-sample quantile clipping
                s_min = float(np.quantile(s_local, args.q_low))
                s_max = float(np.quantile(s_local, args.q_high))

        # normalize
        if args.normalize == 'zscore':
            mu, sd = float(np.mean(s_local)), float(np.std(s_local) + 1e-12)
            s_norm = (s_local - mu) / sd
            # re-range to [0,1] using chosen bounds
            s_norm = (s_norm - s_min) / (s_max - s_min + 1e-12)
        else:
            s_norm = (s_local - s_min) / (s_max - s_min + 1e-12)
        s_norm = np.clip(s_norm, 0.0, 1.0)

        # colorize using modern API
        cmap = matplotlib.colormaps.get_cmap(args.colormap)
        colors = np.asarray(cmap(s_norm))[:, :3].astype(np.float32)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # save PLY
        out_ply = args.out_ply or os.path.splitext(npz_path)[0] + '.ply'
        os.makedirs(os.path.dirname(out_ply) or '.', exist_ok=True)
        o3d.io.write_point_cloud(out_ply, pc)
        print(f'Saved colored PLY: {out_ply}  | points={xyz.shape[0]}  cmap={args.colormap} field={args.field}  norm=({args.norm_scope}) bounds=[{s_min:.6f},{s_max:.6f}]')

        # optional GT ply / viewer
        if 'gt' in data.files and data['gt'] is not None:
            gt = (data['gt'].reshape(-1) > 0.5).astype(np.bool_)
            colors_gt = np.tile(np.array([[0.8, 0.8, 0.8]], dtype=np.float32), (xyz.shape[0], 1))
            colors_gt[gt] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red for anomalies
            pc_gt = o3d.geometry.PointCloud()
            pc_gt.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pc_gt.colors = o3d.utility.Vector3dVector(colors_gt.astype(np.float64))
            out_gt = args.out_gt_ply or os.path.splitext(npz_path)[0] + '_gt.ply'
            o3d.io.write_point_cloud(out_gt, pc_gt)
            print(f'Saved GT PLY: {out_gt}')
            if args.show_gt and args.show and not args.npz_dir:
                o3d.visualization.draw_geometries([pc_gt])

        # optional binary mask PLY
        if args.mask_abs >= 0 or args.mask_topk_pct > 0:
            if args.mask_abs >= 0:
                thr = float(args.mask_abs)
            else:
                k = float(args.mask_topk_pct)
                thr = float(np.quantile(s_norm, 1.0 - k / 100.0))
            mask = s_norm >= thr
            colors_m = np.tile(np.array([[0.85, 0.85, 0.85]], dtype=np.float32), (xyz.shape[0], 1))
            colors_m[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            pc_m = o3d.geometry.PointCloud()
            pc_m.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pc_m.colors = o3d.utility.Vector3dVector(colors_m.astype(np.float64))
            out_mask = args.out_mask_ply or os.path.splitext(npz_path)[0] + f'_mask.ply'
            o3d.io.write_point_cloud(out_mask, pc_m)
            print(f'Saved MASK PLY: {out_mask}  thr={thr:.4f} (normalized)  mode={"abs" if args.mask_abs>=0 else f"top{args.mask_topk_pct}%"}')

        if args.show and not args.npz_dir:
            o3d.visualization.draw_geometries([pc])

    # batch or single
    if args.npz_dir:
        if not args.ref_dir:
            args.ref_dir = args.npz_dir
        files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
        for f in files:
            process_one(f)
    else:
        if not args.npz:
            raise SystemExit('Please provide --npz or --npz_dir')
        process_one(args.npz)


if __name__ == '__main__':
    main()