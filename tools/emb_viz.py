import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings with t-SNE')
    parser.add_argument('--npz', type=str, required=True, help='path to saved embedding npz file')
    parser.add_argument('--out', type=str, default='', help='output image path (png)')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--legend_topk', type=int, default=0, help='show legend for top-K frequent classes only (0=all)')
    parser.add_argument('--annotate_centroid', action='store_true', help='annotate each class id at its 2D centroid to disambiguate colors')
    args = parser.parse_args()

    data = np.load(args.npz)
    z = data['z']
    y = data['y']

    # auto clamp perplexity to valid range (< n_samples)
    perpl = min(args.perplexity, max(5, z.shape[0] - 1))
    tsne = TSNE(n_components=2, perplexity=perpl, n_iter=args.n_iter, init='random', learning_rate='auto')
    z2 = tsne.fit_transform(z)

    plt.figure(figsize=(6, 6))
    uniq, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    uniq_sorted = uniq[order]
    counts_sorted = counts[order]
    if args.legend_topk and args.legend_topk > 0:
        legend_set = set(uniq_sorted[:min(args.legend_topk, len(uniq_sorted))])
    else:
        legend_set = set(uniq_sorted)

    for cid in uniq_sorted:
        idx = y == cid
        # show label only for classes in legend_set
        lbl = f'{int(cid)}' if cid in legend_set else None
        plt.scatter(z2[idx, 0], z2[idx, 1], s=8, label=lbl, alpha=0.7)
        # annotate centroid per class if requested
        if args.annotate_centroid and idx.sum() > 0:
            cx, cy = z2[idx, 0].mean(), z2[idx, 1].mean()
            plt.text(cx, cy, str(int(cid)), fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))
    if len(legend_set) > 0:
        plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out = args.out if args.out else os.path.splitext(args.npz)[0] + '_tsne.png'
    plt.savefig(out, dpi=200)
    print(f'Saved: {out}  |  samples={z.shape[0]}, classes={len(uniq_sorted)}, perpl={perpl}, legend_topk={args.legend_topk}')


if __name__ == '__main__':
    main()
