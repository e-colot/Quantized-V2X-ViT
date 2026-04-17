# -*- coding: utf-8 -*-

import argparse
import importlib
import os
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


class Arguments:
    def __init__(self):
        print('Default parameters used')
        self.model_dir = 'opencood/v2x-vit'
        self.engine_path = 'tensorRT/v2xvit_fp32.ts'
        self.fusion_method = 'intermediate'
        self.num_workers = 16
        self.max_batches = -1
        self.show_vis = False
        self.save_vis = False
        self.save_npy = False
        self.global_sort_detections = False


def test_parser():
    parser = argparse.ArgumentParser(
        description='TensorRT inference for OpenCOOD (metric-compatible with torch inference)'
    )
    parser.add_argument('--model_dir', type=str,
                        help='Path to trained model directory')
    parser.add_argument('--engine_path', type=str,
                        help='Path to TensorRT TorchScript module (*.ts)')
    parser.add_argument('--fusion_method', type=str,
                        help='early or intermediate (late is not supported by this TRT adapter)')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--max_batches', type=int,
                        help='Stop early after this many batches. -1 means all batches')
    parser.add_argument('--show_vis', action='store_true',
                        help='Whether to show image visualization result')
    parser.add_argument('--save_vis', action='store_true',
                        help='Whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='Whether to save prediction and gt result in npy folder')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Whether to globally sort detections by confidence score')

    opt = parser.parse_args()
    defaults = Arguments()
    for key, value in vars(defaults).items():
        if not hasattr(opt, key) or getattr(opt, key) is None:
            setattr(opt, key, value)
    return opt


def _extract_model_inputs(cav_content):
    record_len = cav_content['record_len'].to(torch.int32)
    prior_encoding = cav_content['prior_encoding'].to(torch.float32)
    spatial_correction_matrix = cav_content['spatial_correction_matrix'].to(torch.float32)

    processed_lidar = cav_content['processed_lidar']
    voxel_features = processed_lidar['voxel_features'].to(torch.float32)
    voxel_coords = processed_lidar['voxel_coords'].to(torch.int32)
    voxel_num_points = processed_lidar['voxel_num_points'].to(torch.float32)

    return (
        voxel_features,
        voxel_coords,
        voxel_num_points,
        record_len,
        prior_encoding,
        spatial_correction_matrix,
    )


def _run_trt_engine(trt_model, model_inputs):
    outputs = trt_model(*model_inputs)

    if isinstance(outputs, dict):
        if 'psm' in outputs and 'rm' in outputs:
            return outputs['psm'], outputs['rm']
        raise RuntimeError('TRT output dict does not contain expected keys: psm/rm')

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        return outputs[0], outputs[1]

    raise RuntimeError(
        f'Unexpected TRT output type: {type(outputs)}. '
        'Expected tuple/list(psm, rm) or dict with keys psm/rm.'
    )


def _inference_with_trt(batch_data, trt_model, dataset):
    cav_content = batch_data['ego']
    model_inputs = _extract_model_inputs(cav_content)
    psm, rm = _run_trt_engine(trt_model, model_inputs)

    output_dict = OrderedDict()
    output_dict['ego'] = {'psm': psm, 'rm': rm}

    return dataset.post_process(batch_data, output_dict)


def main():
    opt = test_parser()
    assert opt.fusion_method in ['early', 'intermediate'], (
        'Only early and intermediate are supported by this TRT inference script.'
    )

    if not os.path.exists(opt.engine_path):
        raise FileNotFoundError(f'TensorRT engine module not found: {opt.engine_path}')

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f'{len(opencood_dataset)} samples found.')

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for TensorRT inference.')

    device = torch.device('cuda')

    # Register TensorRT TorchScript custom classes before torch.jit.load.
    importlib.import_module('torch_tensorrt')

    print(f'Loading TensorRT TorchScript module from: {opt.engine_path}')
    trt_model = torch.jit.load(opt.engine_path, map_location=device)
    trt_model.eval()

    # Keep the same structure as torch inference metrics.
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
    }

    progress_bar = tqdm(
        data_loader,
        total=len(data_loader),
        desc='TensorRT Inference',
        unit='batch',
        dynamic_ncols=True,
    )

    for i, batch_data in enumerate(progress_bar):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            pred_box_tensor, pred_score, gt_box_tensor = _inference_with_trt(
                batch_data,
                trt_model,
                opencood_dataset,
            )

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path,
                )

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_dir = os.path.join(opt.model_dir, 'vis_trt')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_save_path = os.path.join(vis_dir, f'{i:05d}.png')

                opencood_dataset.visualize_result(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    opt.show_vis,
                    vis_save_path,
                    dataset=opencood_dataset,
                )

        if opt.max_batches > 0 and (i + 1) >= opt.max_batches:
            print(f'Early stop at max_batches={opt.max_batches}')
            break

    eval_utils.eval_final_results(
        result_stat,
        opt.model_dir,
        opt.global_sort_detections,
    )


if __name__ == '__main__':
    main()
