import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings with t-SNE')
    parser.add_argument('--npz', type=str, required=True, help='path to saved embedding npz file')
    parser.add_argument('--out', type=str, default='', help='output image path (png)')
    parser.add_argument('--perplexity', type=float, default=10.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    args = parser.parse_args()

    data = np.load(args.npz)
    z = data['z']
    y = data['y']

    tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=args.n_iter, init='random', learning_rate='auto')
    z2 = tsne.fit_transform(z)

    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(y))
    for cid in np.unique(y):
        idx = y == cid
        plt.scatter(z2[idx, 0], z2[idx, 1], s=8, label=f'{int(cid)}', alpha=0.7)
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out = args.out if args.out else os.path.splitext(args.npz)[0] + '_tsne.png'
    plt.savefig(out, dpi=200)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
