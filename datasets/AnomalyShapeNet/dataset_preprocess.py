import math
import glob
import torch
import random
import numpy as np
import open3d as o3d
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import datasets.AnomalyShapeNet.transform as aug_transform
import os
import re
import hashlib

class Dataset:
    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.dataset_workers = cfg.num_works
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num
        # perf options
        self.pin_memory = getattr(cfg, 'pin_memory', False)
        self.prefetch_factor = getattr(cfg, 'prefetch_factor', 2)
        # region-style pseudo anomaly options
        self.region_anom_enable = bool(getattr(cfg, 'region_anom_enable', False))
        self.region_anom_prob = float(getattr(cfg, 'region_anom_prob', 0.7))
        self.region_K_max = int(getattr(cfg, 'region_K_max', 3))
        self.region_area_min = float(getattr(cfg, 'region_area_min', 0.05))
        self.region_area_max = float(getattr(cfg, 'region_area_max', 0.25))
        self.region_soft_min = float(getattr(cfg, 'region_soft_min', 0.05))
        self.region_soft_max = float(getattr(cfg, 'region_soft_max', 0.2))
        self.region_amp_min = float(getattr(cfg, 'region_amp_min', 0.06))
        self.region_amp_max = float(getattr(cfg, 'region_amp_max', 0.12))
        self.region_mix_sign_prob = float(getattr(cfg, 'region_mix_sign_prob', 0.2))
        # cache options
        self.cache_io = getattr(cfg, 'cache_io', False)
        self.cache_dir = os.path.join(getattr(cfg, 'cache_dir', './cache'), 'AnomalyShapeNet')
        if self.cache_io:
            os.makedirs(self.cache_dir, exist_ok=True)

        # categories handling
        self.single_category = cfg.category
        all_categories = os.listdir('datasets/AnomalyShapeNet/dataset/pcd/')
        if hasattr(cfg, 'categories') and cfg.categories:
            if cfg.categories.strip().lower() == 'all':
                self.category_list = sorted(all_categories)
            else:
                requested = [c.strip() for c in cfg.categories.split(',') if c.strip()]
                for c in requested:
                    assert c in all_categories, f'Unknown category {c} for AnomalyShapeNet'
                self.category_list = requested
        else:
            assert self.single_category in all_categories
            self.category_list = [self.single_category]
        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.num_classes = len(self.category_list)

        # build training file list across categories (template meshes)
        self.train_file_list = []  # list of (path, cat_id)
        for c in self.category_list:
            # data_list = glob.glob(f"datasets/AnomalyShapeNet/dataset/obj/{c}/*.obj")
            pattern = f"datasets/AnomalyShapeNet/dataset/obj/{c}/*.obj"
            data_list = glob.glob(pattern)
            is_train = re.compile(r'template')
            train_files = list(filter(is_train.search, data_list))
            train_files.sort()
            # repeat per original design
            train_files = train_files * self.data_repeat
            if len(train_files) == 0:
                raise RuntimeError(f"[AnomalyShapeNet] No training templates found. Searched pattern={pattern} and filtered by 'template'. Category={c}")
            self.train_file_list += [(p, self.cat2id[c]) for p in train_files]

        # test files remain per-category, used in eval stage
        self.test_file_list = []
        for c in self.category_list:
            test_files = glob.glob(f"datasets/AnomalyShapeNet/dataset/pcd/{c}/test/*.pcd")
            test_files.sort()
            self.test_file_list += test_files

        # augmentations
        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0) #p表示应用概率
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)
        # for contrastive views (no sphere mask, no pseudo anomaly)
        self.contrast_aug = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            aug_transform.RandomScale(0.9, 1.1, p=0.5),
            aug_transform.RandomJitter(sigma=0.005, clip=0.02, p=0.5),
            self.NormalizeCoord,
        ])

        self.train_aug_compose = aug_transform.Compose([self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                        self.NormalizeCoord, self.SphereCropMask])

        self.test_aug_compose = aug_transform.Compose([self.CenterShift, self.NormalizeCoord])

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def trainLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=True, sampler=None,
                                            drop_last=True, pin_memory=self.pin_memory,
                                            worker_init_fn=self._worker_init_fn_,
                                            persistent_workers=(self.dataset_workers>0),
                                            prefetch_factor=self.prefetch_factor if self.dataset_workers>0 else None)

    def contrastiveLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.contrastiveMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=True, sampler=None,
                                            drop_last=True, pin_memory=self.pin_memory,
                                            worker_init_fn=self._worker_init_fn_,
                                            persistent_workers=(self.dataset_workers>0),
                                            prefetch_factor=self.prefetch_factor if self.dataset_workers>0 else None)

    def testLoader(self):
        test_set = list(range(len(self.test_file_list)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None,
                                           drop_last=False, pin_memory=self.pin_memory,
                                           worker_init_fn=self._worker_init_fn_,
                                           persistent_workers=(self.dataset_workers>0)
                                        #    prefetch_factor=self.prefetch_factor if self.dataset_workers>0 else None
                                           )

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center)
        movement_ratios = 1 - (distances_to_center / max_distance)
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / (np.max(movement_ratios) - np.min(movement_ratios))

        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
        movements = movement_ratios * distance_to_move * directions
        new_points = points + np.abs(normals) * movements[:, np.newaxis]

        return new_points

    def generate_region_anomaly(self, xyz: np.ndarray, normals: np.ndarray):
        """Plateau-style large-area convex/concave anomalies with soft boundary.
        Returns new_xyz, gt_offset
        """
        N = xyz.shape[0]
        if N == 0:
            return xyz.copy(), np.zeros_like(xyz)
        # unit normals magnitude safeguard
        nrm = normals
        if nrm is None or nrm.shape[0] != N:
            # fallback: approximate by zeros (no move)
            nrm = np.zeros_like(xyz, dtype=np.float32)
        else:
            nn = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8
            nrm = nrm / nn
        # choose K regions
        K = max(1, int(np.random.randint(1, max(2, self.region_K_max+1))))
        offset = np.zeros_like(xyz, dtype=np.float32)
        # precompute spatial scale
        xyz_min = xyz.min(axis=0); xyz_max = xyz.max(axis=0)
        scale = float(np.linalg.norm(xyz_max - xyz_min)) + 1e-8
        for k in range(K):
            # center index
            ci = np.random.randint(0, N)
            c = xyz[ci]
            # pick target area fraction and derive radius by percentile of distances
            alpha = float(np.random.uniform(self.region_area_min, self.region_area_max))
            # distance to center
            d = np.linalg.norm(xyz - c, axis=1)
            # radius so that roughly alpha*N points are inside
            r = np.percentile(d, min(99.0, max(1.0, alpha*100.0))) + 1e-8
            soft = float(np.random.uniform(self.region_soft_min, self.region_soft_max))
            r_hard = r * (1.0 - soft)
            # amplitude and sign
            A = float(np.random.uniform(self.region_amp_min, self.region_amp_max))
            sign = 1.0 if (np.random.rand() < 0.5) else -1.0
            # soft plateau weight
            w = np.zeros((N,), dtype=np.float32)
            inner = d <= r_hard
            w[inner] = 1.0
            band = (d > r_hard) & (d <= r)
            t = (r - d[band]) / max(1e-6, (r - r_hard))
            # cosine falloff for smooth boundary
            w[band] = 0.5 * (1.0 - np.cos(np.clip(t, 0.0, 1.0) * np.pi))
            # optional mixed-sign subregion
            if np.random.rand() < self.region_mix_sign_prob:
                # pick a subcenter and smaller radius
                cj = xyz[np.random.randint(0, N)]
                dj = np.linalg.norm(xyz - cj, axis=1)
                rj = 0.5 * r
                mask_j = dj <= rj
                # flip sign inside
                sign_map = np.ones((N,), dtype=np.float32) * sign
                sign_map[mask_j] = -sign
            else:
                sign_map = np.ones((N,), dtype=np.float32) * sign
            disp = (A * w * sign_map)[:, None] * nrm
            offset += disp.astype(np.float32)
        new_xyz = xyz + offset
        return new_xyz.astype(np.float32), offset.astype(np.float32)

    def contrastiveMerge(self, id):
        file_name = []
        labels = []
        # view1  双视图 使用相同的数据增强流 但应用的具体变换参数不同
        xyz_voxel_1 = []
        feat_voxel_1 = []
        # view2
        xyz_voxel_2 = []
        feat_voxel_2 = []
        for i, idx in enumerate(id):
            fn_path, cat_id = self.train_file_list[idx]
            file_name.append(fn_path)
            labels.append(cat_id)
            # load mesh with optional cache
            cache_key = None
            if self.cache_io:
                rel = fn_path.replace('/', '_').replace('\\', '_')
                h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
            if self.cache_io and os.path.isfile(cache_key):
                arr = np.load(cache_key)
                coord = arr['coord']
            else:
                obj = o3d.io.read_triangle_mesh(fn_path)
                obj.compute_vertex_normals()
                coord = np.asarray(obj.vertices)
                if self.cache_io:
                    try:
                        np.savez(cache_key, coord=coord.astype(np.float32), normal=np.asarray(obj.vertex_normals).astype(np.float32))
                    except Exception:
                        pass

            # two views with independent augmentations
            Point_dict_1 = {'coord': coord.copy()}
            Point_dict_1 = self.contrast_aug(Point_dict_1)
            xyz1 = Point_dict_1['coord'].astype(np.float32)
            Point_dict_2 = {'coord': coord.copy()}
            Point_dict_2 = self.contrast_aug(Point_dict_2)
            xyz2 = Point_dict_2['coord'].astype(np.float32)

            q1, f1, _, _ = ME.utils.sparse_quantize(xyz1, xyz1, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
            q2, f2, _, _ = ME.utils.sparse_quantize(xyz2, xyz2, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
            xyz_voxel_1.append(q1)
            feat_voxel_1.append(f1)
            xyz_voxel_2.append(q2)
            feat_voxel_2.append(f2)
        #将批次中所有的样本合并为一个批次
        xyz_voxel_1_batch, feat_voxel_1_batch = ME.utils.sparse_collate(xyz_voxel_1, feat_voxel_1)
        xyz_voxel_2_batch, feat_voxel_2_batch = ME.utils.sparse_collate(xyz_voxel_2, feat_voxel_2)
        labels = torch.from_numpy(np.array(labels)).long()
        return {
            'xyz_voxel_view1': xyz_voxel_1_batch,
            'feat_voxel_view1': feat_voxel_1_batch,
            'xyz_voxel_view2': xyz_voxel_2_batch,
            'feat_voxel_view2': feat_voxel_2_batch,
            'labels': labels,
            'fn': file_name,
        }

    def trainMerge(self, id):
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        xyz_shifted = []
        v2p_index_batch = []
        category_ids = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0
        gt_offset_list = []
        for i, idx in enumerate(id):
            fn_path, cat_id = self.train_file_list[idx]  # get path
            file_name.append(fn_path)
            category_ids.append(cat_id)

            # #####Load data (with optional cache)
            cache_key = None
            if self.cache_io:
                rel = fn_path.replace('/', '_').replace('\\', '_')
                h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
            if self.cache_io and os.path.isfile(cache_key):
                arr = np.load(cache_key)
                coord = arr['coord']
                vertex_normals = arr['normal']
            else:
                obj = o3d.io.read_triangle_mesh(fn_path)
                obj.compute_vertex_normals()
                coord = np.asarray(obj.vertices)
                vertex_normals = np.asarray(obj.vertex_normals)
                if self.cache_io:
                    try:
                        np.savez(cache_key, coord=coord.astype(np.float32), normal=vertex_normals.astype(np.float32))
                    except Exception:
                        pass
            mask = np.ones(coord.shape[0]) * -1

            # ####Data aug
            Point_dict = {'coord': coord, 'normal': vertex_normals, 'mask': mask}
            Point_dict, centers = self.train_aug_compose(Point_dict)

            # ####Trans to numpy
            xyz = Point_dict['coord'].astype(np.float32)
            normal = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)
            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

            # Choose between legacy Norm-AS and region-style anomaly
            use_region = self.region_anom_enable and (random.random() < self.region_anom_prob)
            if use_region:
                new_xyz, gt_offset = self.generate_region_anomaly(xyz, normal)
            else:
                num_shift = 1
                mask_range = np.arange(0, self.mask_num // 2)
                shift_index = np.random.choice(mask_range, num_shift, replace=False)
                mask[np.isin(mask, shift_index)] = -1
                shift_xyz = xyz[mask == -1].copy()
                shift_normal = normal[mask == -1].copy()
                shifted_xyz = self.generate_pseudo_anomaly(shift_xyz, shift_normal, centers[shift_index[0]], distance_to_move=np.random.uniform(0.06, 0.12))
                new_xyz = xyz.copy()
                new_xyz[mask == -1] = shifted_xyz
                gt_offset = new_xyz - xyz
            gt_offset_list.append(torch.from_numpy(gt_offset))
            xyz_shifted.append(torch.from_numpy(new_xyz))

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(new_xyz, new_xyz,
                                                                                        quantization_size=self.voxel_size,
                                                                                        return_index=True,
                                                                                        return_inverse=True)

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]

            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # -------------------------------Batch -------------------------
            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            v2p_index_batch.append(v2p_index)

        # ####numpy to torch
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        xyz_shifted = torch.cat(xyz_shifted, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        batch_count = torch.from_numpy(np.array(batch_count))
        batch_offset = torch.cat(gt_offset_list, 0).to(torch.float32)

        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'xyz_shifted': xyz_shifted, 'batch_count': batch_count, 'batch_offset': batch_offset, 'category_id': torch.tensor(category_ids, dtype=torch.long)}

    def testMerge(self, id):
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        v2p_index_batch = []
        labels = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]
        for i, idx in enumerate(id):
            fn_path = self.test_file_list[idx]  # get path
            file_name.append(self.test_file_list[idx])

            # detect category for gt path
            c = fn_path.split('/')[-3]

            if 'positive' in fn_path:
                cache_key = None
                if self.cache_io:
                    rel = fn_path.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'pcd_{h}.npz')
                if self.cache_io and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['xyz']
                else:
                    pcd = o3d.io.read_point_cloud(fn_path)
                    coord = np.asarray(pcd.points)
                    if self.cache_io:
                        try:
                            np.savez(cache_key, xyz=coord.astype(np.float32))
                        except Exception:
                            pass
            else:
                sample_name = fn_path.split('/')[-1].split('.')[0]
                gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{c}/GT/'
                gt_file = gt_mask_path + sample_name + '.txt'
                cache_key = None
                if self.cache_io:
                    rel = gt_file.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'gt_{h}.npz')
                if self.cache_io and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['xyz']
                else:
                    coord = np.loadtxt(gt_file, delimiter=',')[:, 0:3]
                    if self.cache_io:
                        try:
                            np.savez(cache_key, xyz=coord.astype(np.float32))
                        except Exception:
                            pass

            # ####Data aug
            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)

            # ####Trans to numpy
            xyz = Point_dict['coord'].astype(np.float32)

            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(xyz, xyz,
                                                                                        quantization_size=self.voxel_size,
                                                                                        return_index=True,
                                                                                        return_inverse=True)

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # -------------------------------Batch -------------------------
            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))
            # normal_original.append(torch.from_numpy(normal))
            v2p_index_batch.append(v2p_index)
            if 'positive' in fn_path:
                labels.append(0)
            else:
                labels.append(1)

        # ####numpy to torch
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))
        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'fn': file_name, 'v2p_index': v2p_index_batch, 'labels': labels, 'batch_count': batch_count}
