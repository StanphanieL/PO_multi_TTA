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
import datasets.Real3D.transform as aug_transform
import re
import os
import hashlib



def simulate_partial_view(points, normals=None, view_loss_ratio=0.5):
        """
        Simulates a partial view by removing points based on a random viewing direction.

        Args:
            points (np.ndarray): Input point cloud coordinates (N, 3).
            normals (np.ndarray, optional): Input point normals (N, 3). If provided,
                                            visibility check is based on normals.
                                            Otherwise, based on random plane cut.
            view_loss_ratio (float): Approximate ratio of points to remove.

        Returns:
            np.ndarray: Point cloud representing a partial view.
            np.ndarray: Corresponding normals if normals were input.
        """
        num_points = points.shape[0]
        if num_points == 0:
            return points, normals

        if normals is not None:
            # --- Method 1: Normal-based visibility ---
            # Randomly select a viewing direction (vector on unit sphere)
            view_dir = np.random.randn(3)
            view_dir /= np.linalg.norm(view_dir)

            # Calculate dot product between normals and view direction
            # Points whose normal faces away from the view direction are potentially occluded
            visibility_scores = np.dot(normals, view_dir)

            # Keep points whose normals are somewhat aligned with the view direction
            # We use a percentile based on the desired loss ratio
            threshold = np.percentile(visibility_scores, view_loss_ratio * 100)
            keep_indices = visibility_scores >= threshold

            # Ensure we don't remove all points
            if np.sum(keep_indices) == 0:
                keep_indices = np.ones(num_points, dtype=bool) # Keep all if threshold removes everything

            points_partial = points[keep_indices]
            normals_partial = normals[keep_indices] if normals is not None else None
            return points_partial, normals_partial

        else:
            # --- Method 2: Random Plane Cut (Simpler, if normals aren't reliably available) ---
            # Choose a random normal vector for the cutting plane
            plane_normal = np.random.randn(3)
            plane_normal /= np.linalg.norm(plane_normal)

            # Project points onto the plane normal
            projections = np.dot(points, plane_normal)

            # Determine a cutting threshold based on the desired loss ratio
            cut_threshold = np.percentile(projections, view_loss_ratio * 100)

            # Keep points on one side of the plane
            keep_indices = projections >= cut_threshold

            # Ensure we don't remove all points
            if np.sum(keep_indices) == 0:
                keep_indices = np.ones(num_points, dtype=bool)

            points_partial = points[keep_indices]
            # Normals are not modified or returned in this method
            return points_partial, None

class Dataset:
    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.dataset_workers = cfg.num_works
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num
        # train single-view simulation options
        self.partial_view_train = getattr(cfg, 'partial_view_train', False)
        self.partial_view_prob = float(getattr(cfg, 'partial_view_prob', 0.0))
        self.partial_view_ratio = float(getattr(cfg, 'partial_view_ratio', 0.4))
        self.partial_view_method = getattr(cfg, 'partial_view_method', 'plane').lower()
        # perf options
        self.pin_memory = getattr(cfg, 'pin_memory', False)
        self.prefetch_factor = getattr(cfg, 'prefetch_factor', 2)
        # cache options
        self.cache_io = getattr(cfg, 'cache_io', False)
        self.cache_dir = os.path.join(getattr(cfg, 'cache_dir', './cache'), 'Real3D')
        if self.cache_io:
            os.makedirs(self.cache_dir, exist_ok=True)

        # categories handling
        default_list = ['airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck', 'fish', 'gemstone',
                        'seahorse', 'shell', 'starfish', 'toffees']
        if hasattr(cfg, 'categories') and cfg.categories:
            if cfg.categories.strip().lower() == 'all':
                self.category_list = default_list
            else:
                requested = [c.strip() for c in cfg.categories.split(',') if c.strip()]
                for c in requested:
                    assert c in default_list, f'Unknown category {c} for Real3D'
                self.category_list = requested
        else:
            assert cfg.category in default_list
            self.category_list = [cfg.category]
        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.num_classes = len(self.category_list)

        # train files
        self.train_file_list = []  # list of (path, cat_id)
        for c in self.category_list:
            # data_list = glob.glob(f"datasets/Real3D/Real3D-AD-PLY/{c}/*.ply")
            pattern = f"datasets/Real3D/Real3D-AD-PLY/{c}/*.ply"
            data_list = glob.glob(pattern)
            is_train = re.compile(r'template')
            train_files = list(filter(is_train.search, data_list))
            train_files.sort()
            train_files = train_files * self.data_repeat
            if len(train_files) == 0:
                raise RuntimeError(f"[Real3D] No training templates found. Searched pattern={pattern} and filtered by 'template'. Category={c}")
            self.train_file_list += [(p, self.cat2id[c]) for p in train_files]

        # test files
        self.test_file_list = []
        for c in self.category_list:
            test_files = glob.glob(f"datasets/Real3D/Real3D-AD-PCD/{c}/test/*.pcd")
            test_files.sort()
            self.test_file_list += test_files

        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = aug_transform.SphereCropMask(part_num=self.mask_num)

        self.train_aug_compose = aug_transform.Compose(
            [self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
             self.NormalizeCoord, self.SphereCropMask])

        # for contrastive views
        self.contrast_aug = aug_transform.Compose([
            self.CenterShift,
            self.RandomRotate_z,
            self.RandomRotate_y,
            self.RandomRotate_x,
            aug_transform.RandomScale(0.9, 1.1, p=0.5),
            aug_transform.RandomJitter(sigma=0.005, clip=0.02, p=0.5),
            self.NormalizeCoord,
        ])

        self.test_aug_compose = aug_transform.Compose([self.CenterShift, self.NormalizeCoord])


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
                                           persistent_workers=(self.dataset_workers>0),
                                           prefetch_factor=self.prefetch_factor if self.dataset_workers>0 else None)


    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08):
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center)
        movement_ratios = 1 - (distances_to_center / max_distance)
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / (
                    np.max(movement_ratios) - np.min(movement_ratios))

        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
        movements = movement_ratios * distance_to_move * directions
        new_points = points + np.abs(normals) * movements[:, np.newaxis]

        return new_points


    # def contrastiveMerge(self, id):
    #     file_name = []
    #     labels = []
    #     xyz_voxel_1 = []
    #     feat_voxel_1 = []
    #     xyz_voxel_2 = []
    #     feat_voxel_2 = []
    #     for i, idx in enumerate(id):
    #         fn_path, cat_id = self.train_file_list[idx]
    #         file_name.append(fn_path)
    #         labels.append(cat_id)
    #         obj = o3d.io.read_triangle_mesh(fn_path)
    #         obj.compute_vertex_normals()
    #         coord = np.asarray(obj.vertices)

    #         Point_dict_1 = {'coord': coord.copy()}
    #         Point_dict_1 = self.contrast_aug(Point_dict_1)
    #         xyz1 = Point_dict_1['coord'].astype(np.float32)
    #         Point_dict_2 = {'coord': coord.copy()}
    #         Point_dict_2 = self.contrast_aug(Point_dict_2)
    #         xyz2 = Point_dict_2['coord'].astype(np.float32)

    #         q1, f1, _, _ = ME.utils.sparse_quantize(xyz1, xyz1, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
    #         q2, f2, _, _ = ME.utils.sparse_quantize(xyz2, xyz2, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
    #         xyz_voxel_1.append(q1)
    #         feat_voxel_1.append(f1)
    #         xyz_voxel_2.append(q2)
    #         feat_voxel_2.append(f2)

    #     xyz_voxel_1_batch, feat_voxel_1_batch = ME.utils.sparse_collate(xyz_voxel_1, feat_voxel_1)
    #     xyz_voxel_2_batch, feat_voxel_2_batch = ME.utils.sparse_collate(xyz_voxel_2, feat_voxel_2)
    #     labels = torch.from_numpy(np.array(labels)).long()
    #     return {
    #         'xyz_voxel_view1': xyz_voxel_1_batch,
    #         'feat_voxel_view1': feat_voxel_1_batch,
    #         'xyz_voxel_view2': xyz_voxel_2_batch,
    #         'feat_voxel_view2': feat_voxel_2_batch,
    #         'labels': labels,
    #         'fn': file_name,
    #     }

    # In datasets/Real3D/dataset_preprocess.py
    def contrastiveMerge(self, id):
        file_name = []
        labels = []
        xyz_voxel_1 = []
        feat_voxel_1 = []
        xyz_voxel_2 = []
        feat_voxel_2 = []

        # --- Probability to apply partial view simulation ---
        partial_view_prob = 0.9 # Example: Apply partial view 50% of the time
        # --- Ratio of points to remove when applying partial view ---
        partial_view_ratio = 0.4 # Example: Remove approx 40% of points

        for i, idx in enumerate(id):
            fn_path, cat_id = self.train_file_list[idx]
            file_name.append(fn_path)
            labels.append(cat_id)

            # Load mesh and get vertices and normals (with optional cache)
            try: # Use try-except for robustness, especially with normals
                cache_key = None
                if self.cache_io:
                    rel = fn_path.replace('/', '_').replace('\\', '_')
                    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                    cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
                if self.cache_io and os.path.isfile(cache_key):
                    arr = np.load(cache_key)
                    coord = arr['coord']
                    vertex_normals = arr['normal'] if 'normal' in arr.files else None
                    if vertex_normals is None:
                        # fallback compute normals once
                        obj = o3d.io.read_triangle_mesh(fn_path)
                        obj.compute_vertex_normals()
                        vertex_normals = np.asarray(obj.vertex_normals)
                else:
                    obj = o3d.io.read_triangle_mesh(fn_path)
                    obj.compute_vertex_normals()
                    coord = np.asarray(obj.vertices)
                    # --- Get normals here ---
                    vertex_normals = np.asarray(obj.vertex_normals)
                    if self.cache_io:
                        try:
                            np.savez(cache_key, coord=coord.astype(np.float32), normal=vertex_normals.astype(np.float32))
                        except Exception:
                            pass
                has_normals = True if vertex_normals is not None else False
            except Exception as e:
                print(f"Warning: Could not load mesh or compute normals for {fn_path}. Using point cloud loading. Error: {e}")
                # Fallback to loading as point cloud if mesh fails or normals are bad
                pcd = o3d.io.read_point_cloud(fn_path) # Assuming templates might also exist as PCD
                coord = np.asarray(pcd.points)
                vertex_normals = None # No reliable normals
                has_normals = False


            # --- Apply standard contrastive augmentations ---
            Point_dict_1 = {'coord': coord.copy()}
            Point_dict_1 = self.contrast_aug(Point_dict_1)
            xyz1 = Point_dict_1['coord'].astype(np.float32)
            # Apply same augmentation stream but potentially different params to get view 2
            Point_dict_2 = {'coord': coord.copy()}
            Point_dict_2 = self.contrast_aug(Point_dict_2)
            xyz2 = Point_dict_2['coord'].astype(np.float32)

            # --- Apply Partial View Simulation ---
            # We need normals corresponding to the augmented views if using normal-based method
            # For simplicity, let's use the random plane cut method here (Method 2)
            # which doesn't strictly need accurate normals after augmentation.
            # Or, apply partial view BEFORE standard augmentation (might be simpler)

            # --- Option A: Apply partial view AFTER standard augmentation (using Plane Cut) ---
            if random.random() < partial_view_prob:
                xyz1, _ = simulate_partial_view(xyz1, normals=None, view_loss_ratio=partial_view_ratio)
            if random.random() < partial_view_prob:
                xyz2, _ = simulate_partial_view(xyz2, normals=None, view_loss_ratio=partial_view_ratio)


            # --- Option B: Apply partial view BEFORE standard augmentation (More robust maybe) ---
            # coord_view1 = coord.copy()
            # normals_view1 = vertex_normals.copy() if has_normals else None
            # if random.random() < partial_view_prob:
            #      coord_view1, normals_view1 = simulate_partial_view(coord_view1, normals_view1, view_loss_ratio=partial_view_ratio)

            # coord_view2 = coord.copy()
            # normals_view2 = vertex_normals.copy() if has_normals else None
            # if random.random() < partial_view_prob:
            #      coord_view2, normals_view2 = simulate_partial_view(coord_view2, normals_view2, view_loss_ratio=partial_view_ratio)

            # # Apply standard contrastive augmentations AFTER partial view simulation
            # Point_dict_1 = {'coord': coord_view1} # Pass normals if needed by contrast_aug
            # Point_dict_1 = self.contrast_aug(Point_dict_1)
            # xyz1 = Point_dict_1['coord'].astype(np.float32)

            # Point_dict_2 = {'coord': coord_view2}
            # Point_dict_2 = self.contrast_aug(Point_dict_2)
            # xyz2 = Point_dict_2['coord'].astype(np.float32)


            # --- Quantization and Batching (No changes needed here) ---
            # Ensure xyz1 and xyz2 are not empty after simulation
            if xyz1.shape[0] > 0:
                q1, f1, _, _ = ME.utils.sparse_quantize(xyz1, xyz1, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
                xyz_voxel_1.append(q1)
                feat_voxel_1.append(f1)
            else: # Handle empty point cloud case if necessary
                # Maybe append empty tensors or skip? Needs careful handling downstream.
                # For now, let's just make sure it doesn't crash by skipping if empty
                print(f"Warning: View 1 became empty for {fn_path} after partial view simulation.")
                # Need a strategy here - perhaps skip this sample? Or use original?
                # Simplest: use original if simulation makes it empty
                if xyz1.shape[0] == 0:
                    xyz1_orig = self.contrast_aug({'coord': coord.copy()})['coord'].astype(np.float32)
                    q1, f1, _, _ = ME.utils.sparse_quantize(xyz1_orig, xyz1_orig, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
                    xyz_voxel_1.append(q1)
                    feat_voxel_1.append(f1)


            if xyz2.shape[0] > 0:
                q2, f2, _, _ = ME.utils.sparse_quantize(xyz2, xyz2, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
                xyz_voxel_2.append(q2)
                feat_voxel_2.append(f2)
            else:
                print(f"Warning: View 2 became empty for {fn_path} after partial view simulation.")
                if xyz2.shape[0] == 0:
                    xyz2_orig = self.contrast_aug({'coord': coord.copy()})['coord'].astype(np.float32)
                    q2, f2, _, _ = ME.utils.sparse_quantize(xyz2_orig, xyz2_orig, quantization_size=self.voxel_size, return_index=True, return_inverse=True)
                    xyz_voxel_2.append(q2)
                    feat_voxel_2.append(f2)


        # --- Collate batches (No changes needed) ---
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
        v2p_index_batch = []
        gt_offset_list = []
        xyz_shifted = []
        category_ids = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0
        for i, idx in enumerate(id):
            fn_path, cat_id = self.train_file_list[idx]  # get path
            file_name.append(fn_path)
            category_ids.append(cat_id)

            # load mesh with optional cache
            cache_key = None
            if self.cache_io:
                rel = fn_path.replace('/', '_').replace('\\', '_')
                h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:16]
                cache_key = os.path.join(self.cache_dir, f'mesh_{h}.npz')
            if self.cache_io and os.path.isfile(cache_key):
                arr = np.load(cache_key)
                coord = arr['coord']
                vertex_normals = arr['normal'] if 'normal' in arr.files else None
                if vertex_normals is None:
                    obj = o3d.io.read_triangle_mesh(fn_path)
                    obj.compute_vertex_normals()
                    vertex_normals = np.asarray(obj.vertex_normals)
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

            # optional single-view simulation BEFORE standard aug to mimic test partial scans
            if self.partial_view_train and random.random() < self.partial_view_prob:
                try:
                    if self.partial_view_method == 'normal' and vertex_normals is not None and vertex_normals.shape[0] == coord.shape[0]:
                        coord_sv, normals_sv = simulate_partial_view(coord, normals=vertex_normals, view_loss_ratio=self.partial_view_ratio)
                    else:
                        coord_sv, normals_sv = simulate_partial_view(coord, normals=None, view_loss_ratio=self.partial_view_ratio)
                    # fallback if too few points remain
                    if coord_sv is not None and coord_sv.shape[0] > max(128, int(0.2 * coord.shape[0])):
                        coord, vertex_normals = coord_sv, (normals_sv if normals_sv is not None else vertex_normals[:coord_sv.shape[0]] if vertex_normals is not None else None)
                except Exception:
                    pass

            mask = np.ones(coord.shape[0]) * -1

            Point_dict = {'coord': coord, 'normal': vertex_normals, 'mask': mask}
            Point_dict, centers = self.train_aug_compose(Point_dict)
            xyz = Point_dict['coord'].astype(np.float32)
            normal = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)
            mask[mask == (self.mask_num + 1)] = self.mask_num - 1

            xyz_original.append(torch.from_numpy(xyz))

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

            if 'good' in fn_path:
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
                gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{c}/gt/'
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
                    coord = np.loadtxt(gt_file)[:, 0:3]
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
            v2p_index_batch.append(v2p_index)
            if 'good' in fn_path:
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
