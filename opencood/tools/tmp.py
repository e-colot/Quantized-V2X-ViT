# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch_tensorrt
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, build
from opencood.data_utils.datasets import build_dataset


def _collect_tensors(value, out_tensors):
    if torch.is_tensor(value):
        out_tensors.append(value)
        return

    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _collect_tensors(value[key], out_tensors)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensors(item, out_tensors)
        return

    raise TypeError(f'Unsupported output item type: {type(value)}')


def _flatten_concat(output):
    tensors = []
    _collect_tensors(output, tensors)

    if not tensors:
        raise RuntimeError('Output contains no tensors to compare.')

    flattened = [tensor.detach().float().reshape(-1) for tensor in tensors]
    return torch.cat(flattened, dim=0)


def _batch_mse(torch_output, trt_output):
    torch_flat = _flatten_concat(torch_output)
    trt_flat = _flatten_concat(trt_output)

    if torch_flat.numel() != trt_flat.numel():
        raise RuntimeError(
            f'Flattened output size mismatch: {torch_flat.numel()} vs {trt_flat.numel()}'
        )

    return torch.mean((torch_flat - trt_flat) ** 2).item()


def parser():
    parser = argparse.ArgumentParser(description="Model selector")
    parser.add_argument('--model', type=str,
                        required=True)
    opt = parser.parse_args()
    return str(opt.model)

class Arguments:
    def __init__(self, modelName):
        print('Default parameters used')
        self.show_vis = False
        self.show_sequence = False
        self.save_vis = False
        self.save_npy = False
        self.global_sort_detections = False
        self.fusion_method = 'intermediate'
        if modelName == "v2xvit":
            self.model_dir = 'opencood/logs/v2x-vit'
        elif modelName == "ppif":
            self.model_dir = 'opencood/logs/pointPillarIntermediateFusion'

def main():
    modelName = parser()
    valid_model_names = {
        "v2xvit",
        "ppif" # point pillar intermediate fusion
    }

    if modelName not in valid_model_names:
        raise ValueError(f"Invalid TRT_STAGE={modelName}. Use one of {sorted(valid_model_names)}")
    
    opt = Arguments(modelName)

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    
    print('Calling TensorRT builder')
    build.main('ppif')

    print('Loading TensorRT engine')
    engine_path = os.path.join(opt.model_dir, "trt.pt")
    trt_model = torch.jit.load(engine_path).cuda()


    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    mse_history = []

    progress_bar = tqdm(data_loader,
                        total=len(data_loader),
                        desc='Inference',
                        unit='batch',
                        dynamic_ncols=True)
    for i, batch_data in enumerate(progress_bar):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            cav_content = batch_data['ego']
            

            record_len = cav_content['record_len'].to(torch.int32)
            prior_encoding = cav_content['prior_encoding'].to(torch.float32)
            spatial_correction_matrix = cav_content['spatial_correction_matrix'].to(torch.float32)

            processed_lidar = cav_content['processed_lidar']
            voxel_features = processed_lidar['voxel_features'].to(torch.float32)
            voxel_coords = processed_lidar['voxel_coords'].to(torch.float32)
            voxel_num_points = processed_lidar['voxel_num_points'].to(torch.int32)

            torchOutput = model(voxel_features, voxel_coords, voxel_num_points, record_len, 
                        spatial_correction_matrix, prior_encoding)
            trtOutput = trt_model(voxel_features, voxel_coords, voxel_num_points, record_len, 
                        spatial_correction_matrix, prior_encoding)

            batch_mse = _batch_mse(torchOutput, trtOutput)
            mse_history.append(batch_mse)

            running_mse = float(np.mean(mse_history))
            progress_bar.set_postfix(
                batch_mse=f'{batch_mse:.3e}',
                running_mse=f'{running_mse:.3e}',
            )

    if not mse_history:
        raise RuntimeError('No batches were processed; MSE history is empty.')

    print(
        'MSE summary: '
        f'mean={np.mean(mse_history):.6e}, '
        f'median={np.median(mse_history):.6e}, '
        f'max={np.max(mse_history):.6e}'
    )

    plt.figure(figsize=(10, 4))
    plt.plot(mse_history, linewidth=1.5)

    plt.xlabel('Batch index')
    plt.ylabel('MSE')
    plt.title('TensorRT vs PyTorch MSE per batch')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(opt.model_dir, 'trt_vs_torch_mse.png')
    plt.savefig(fig_path, dpi=150)
    print(f'Saved MSE plot to: {fig_path}')
    plt.show()


if __name__ == '__main__':
    main()
