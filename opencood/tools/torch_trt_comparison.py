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


def _batch_relative_l2_error(torch_output, trt_output, eps=1e-12):
    torch_flat = _flatten_concat(torch_output)
    trt_flat = _flatten_concat(trt_output)

    if torch_flat.numel() != trt_flat.numel():
        raise RuntimeError(
            f'Flattened output size mismatch: {torch_flat.numel()} vs {trt_flat.numel()}'
        )

    denom = torch.linalg.norm(torch_flat)
    rel_l2 = torch.linalg.norm(torch_flat - trt_flat) / (denom + eps)
    return rel_l2.item()

def _batch_relative_snr(torch_out, trt_out):
    eps = 1e-12
    torch_flat = _flatten_concat(torch_out)
    trt_flat = _flatten_concat(trt_out)

    if torch_flat.numel() != trt_flat.numel():
        raise RuntimeError(
            f'Flattened output size mismatch: {torch_flat.numel()} vs {trt_flat.numel()}'
        )
    noise = torch_out - trt_out
    snr = 20 * torch.log10(torch.norm(torch_out) / (torch.norm(noise)+eps))
    return snr.item()


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
    build.main(modelName)

    print('Loading TensorRT engine')
    dataset_type = hypes['validate_dir'].split('/')[-1]
    engine_path = os.path.join(opt.model_dir, "trt_" + dataset_type + '.pt')
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

    rel_snr_history = []

    progress_bar = tqdm(data_loader,
                        total=len(data_loader),
                        desc='Inference',
                        unit='batch',
                        dynamic_ncols=True)
    for i, batch_data in enumerate(progress_bar):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            cav_content = batch_data['ego']
            processed_lidar = cav_content['processed_lidar']

            voxel_features = processed_lidar['voxel_features'].to(torch.float32)
            voxel_coords = processed_lidar['voxel_coords'].to(torch.int32)
            voxel_num_points = processed_lidar['voxel_num_points'].to(torch.int32)
            record_len = cav_content['record_len'].to(torch.int32)
            spatial_correction_matrix = cav_content['spatial_correction_matrix'].to(torch.float32)
            prior_encoding = cav_content['prior_encoding'].to(torch.float32)

            torchOutput = model(voxel_features, voxel_coords, voxel_num_points, record_len, 
                        spatial_correction_matrix, prior_encoding)
            trtOutput = trt_model(voxel_features, voxel_coords, voxel_num_points, record_len, 
                        spatial_correction_matrix, prior_encoding)

            batch_rel_snr = _batch_relative_snr(torchOutput, trtOutput)
            rel_snr_history.append(batch_rel_snr)

            running_rel_snr = float(np.mean(rel_snr_history))
            progress_bar.set_postfix(SNR=f'{running_rel_snr:<10.1f}')

    if not rel_snr_history:
        raise RuntimeError('No batches were processed; SNR history is empty.')

    print(f"\n{'='*15} SNR SUMMARY {'='*15}")
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 63)
    print(f"{'Mean  ':<25} | {np.mean(rel_snr_history):<10.3f} dB")
    print(f"{'Median':<25} | {np.median(rel_snr_history):<10.3f} dB")
    print(f"{'Max   ':<25} | {np.max(rel_snr_history):<10.3f} dB")
    print("-" * 63)

if __name__ == '__main__':
    main()
