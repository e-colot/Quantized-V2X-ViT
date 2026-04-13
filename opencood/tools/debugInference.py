# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils


def test_parser():
    parser = argparse.ArgumentParser(description="Inference with CPU profiler")
    parser.add_argument('--model_dir', type=str,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--profile_row_limit', type=int, default=20,
                        help='number of rows to show in profiler table')
    parser.add_argument('--profile_every_batch', action='store_true',
                        help='print profiler output for every batch')
    parser.add_argument('--profile_batch_index', type=int, default=0,
                        help='batch index to profile when profile_every_batch is not set')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='maximum number of samples to run during debug inference')
    opt = parser.parse_args()
    return opt if opt.model_dir else Arguments()


class Arguments:
    def __init__(self):
        print('Default parameters used')
        self.model_dir = 'opencood/v2x-vit'
        self.fusion_method = 'intermediate'
        self.show_vis = False
        self.show_sequence = False
        self.save_vis = False
        self.save_npy = False
        self.global_sort_detections = False
        self.profile_row_limit = 20
        self.profile_every_batch = False
        self.profile_batch_index = 0
        self.max_samples = 10


def run_inference_once(batch_data, model, opencood_dataset, fusion_method):
    if fusion_method == 'late':
        return inference_utils.inference_late_fusion(batch_data,
                                                     model,
                                                     opencood_dataset)
    if fusion_method == 'early':
        return inference_utils.inference_early_fusion(batch_data,
                                                      model,
                                                      opencood_dataset)
    if fusion_method == 'intermediate':
        return inference_utils.inference_intermediate_fusion(batch_data,
                                                             model,
                                                             opencood_dataset)
    raise NotImplementedError('Only early, late and intermediate fusion is supported.')


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    total_samples = min(len(data_loader), opt.max_samples)
    progress_bar = tqdm(data_loader,
                        total=total_samples,
                        desc='Inference',
                        unit='batch',
                        dynamic_ncols=True)
    for i, batch_data in enumerate(progress_bar):
        if i >= opt.max_samples:
            break

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                pred_box_tensor, pred_score, gt_box_tensor = run_inference_once(
                    batch_data,
                    model,
                    opencood_dataset,
                    opt.fusion_method,
                )

        should_print = opt.profile_every_batch or i == opt.profile_batch_index
        if should_print:
            print(f"\n[Profiler] Batch {i}")
            print(prof.key_averages().table(
                sort_by='cpu_time_total',
                row_limit=opt.profile_row_limit,
            ))

        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.3)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.5)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.7)

        if opt.save_npy:
            npy_save_path = os.path.join(opt.model_dir, 'npy')
            if not os.path.exists(npy_save_path):
                os.makedirs(npy_save_path)
            inference_utils.save_prediction_gt(pred_box_tensor,
                                               gt_box_tensor,
                                               batch_data['ego']['origin_lidar'][0],
                                               i,
                                               npy_save_path)

        if opt.show_vis or opt.save_vis:
            vis_save_path = ''
            if opt.save_vis:
                vis_save_path = os.path.join(opt.model_dir, 'vis')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

            opencood_dataset.visualize_result(pred_box_tensor,
                                              gt_box_tensor,
                                              batch_data['ego']['origin_lidar'],
                                              opt.show_vis,
                                              vis_save_path,
                                              dataset=opencood_dataset)

        if opt.show_sequence:
            pcd, pred_o3d_box, gt_o3d_box = \
                vis_utils.visualize_inference_sample_dataloader(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    vis_pcd,
                    mode='constant'
                )
            if i == 0:
                vis.add_geometry(pcd)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box,
                                             update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box,
                                             update_mode='add')

            vis_utils.linset_assign_list(vis,
                                         vis_aabbs_pred,
                                         pred_o3d_box)
            vis_utils.linset_assign_list(vis,
                                         vis_aabbs_gt,
                                         gt_o3d_box)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
