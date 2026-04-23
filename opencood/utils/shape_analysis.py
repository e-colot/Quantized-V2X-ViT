# Generates shapes.log for TensorRT profiling
import argparse
import json
import os

from torch.utils.data import DataLoader
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset


def _extract_input_tensors(batch_data, device='cpu'):
    cav_content = batch_data['ego']
    processed_lidar = cav_content['processed_lidar']

    return (
        processed_lidar['voxel_features'].to(device=device, dtype=torch.float32),
        processed_lidar['voxel_coords'].to(device=device, dtype=torch.int32),
        processed_lidar['voxel_num_points'].to(device=device, dtype=torch.int32),
        cav_content['record_len'].to(device=device, dtype=torch.int32),
        cav_content['spatial_correction_matrix'].to(device=device, dtype=torch.float32),
        cav_content['prior_encoding'].to(device=device, dtype=torch.float32),
    )


def _shape_componentwise(shape_list, reducer):
    dims = len(shape_list[0])
    reduced = []
    for d in range(dims):
        values = sorted(shape[d] for shape in shape_list)
        reduced.append(reducer(values))
    return reduced


def _infer_linked_dimensions(shape_dict):
    dim_series = {}
    for tensor_name, shape_list in shape_dict.items():
        rank = len(shape_list[0])
        for dim_idx in range(rank):
            dim_key = f'{tensor_name}[{dim_idx}]'
            dim_series[dim_key] = [int(shape[dim_idx]) for shape in shape_list]

    linked = []
    dim_keys = sorted(dim_series.keys())
    for i, lhs_key in enumerate(dim_keys):
        lhs_series = dim_series[lhs_key]
        for rhs_key in dim_keys[i + 1:]:
            rhs_series = dim_series[rhs_key]
            if all(lhs == rhs for lhs, rhs in zip(lhs_series, rhs_series)):
                linked.append({'type': 'equal', 'dims': [lhs_key, rhs_key]})
    return linked


class Arguments:
    def __init__(self, model_name):
        self.show_vis = False
        self.show_sequence = False
        self.save_vis = False
        self.save_npy = False
        self.global_sort_detections = False
        self.fusion_method = 'intermediate'
        if model_name == 'v2xvit':
            self.model_dir = 'opencood/logs/v2x-vit'
        elif model_name == 'ppif':
            self.model_dir = 'opencood/logs/pointPillarIntermediateFusion'
        else:
            raise ValueError(f'Unsupported model: {model_name}')

def analyze_shape(hypes):
    shape_names = [
        'voxel_features',
        'voxel_coords',
        'voxel_num_points',
        'record_len',
        'spatial_correction_matrix',
        'prior_encoding',
    ]

    model = hypes['name']
    shape_file_name = os.path.join('opencood/logs/shapes/', model + '_' + hypes['dataset'] + '.log')

    print(f"\n{'=' * 15} SHAPE ANALYSIS {'=' * 15}")
    # If shape_file_name exists, load and return its contents
    if os.path.exists(shape_file_name):
        with open(shape_file_name, 'r', encoding='utf-8') as f:
            shape_stats = json.load(f)
        print(f"Loaded existing shape log from {shape_file_name}")
        print('-' * 52)
        return shape_stats

    print(f"No existing shape log in {shape_file_name}, creating it")
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f'{len(opencood_dataset)} samples found.')
    if len(opencood_dataset) == 0:
        raise RuntimeError('Validation dataset is empty; cannot generate shapes.log')

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    all_shapes = {name: [] for name in shape_names}
    voxel_counts = []

    for batch_data in data_loader:
        cpu_inputs = _extract_input_tensors(batch_data, device='cpu')
        for name, tensor in zip(shape_names, cpu_inputs):
            all_shapes[name].append(tuple(tensor.shape))
        voxel_counts.append(int(cpu_inputs[0].shape[0]))

    shape_stats = {'ranges': {}}
    for name in shape_names:
        shape_list = all_shapes[name]
        shape_stats['ranges'][name] = {
            'min': _shape_componentwise(shape_list, lambda v: v[0]),
            'opt': _shape_componentwise(shape_list, lambda v: v[len(v) // 2]),
            'max': _shape_componentwise(shape_list, lambda v: v[-1]),
        }

    shape_stats['linked_dimensions'] = _infer_linked_dimensions(all_shapes)

    sorted_counts = sorted(voxel_counts)
    target_opt_voxels = sorted_counts[len(sorted_counts) // 2]
    opt_sample_index = min(
        range(len(voxel_counts)),
        key=lambda idx: abs(voxel_counts[idx] - target_opt_voxels),
    )
    shape_stats['opt_sample_index'] = int(opt_sample_index)

    # Save to shape_file_name
    os.makedirs(os.path.dirname(shape_file_name), exist_ok=True)
    with open(shape_file_name, 'w', encoding='utf-8') as f:
        json.dump(shape_stats, f, indent=2)

    print(f'Saved shape log to {shape_file_name}')
    print('-' * 52)
    return shape_stats
