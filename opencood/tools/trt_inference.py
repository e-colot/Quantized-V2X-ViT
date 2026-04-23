import os
from tqdm import tqdm

import torch
import torch_tensorrt
import tensorrt as trt
from torch.utils.data import DataLoader

from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, trt_utils



def main():
    hypes, opt, parser_opt = trt_utils.load_params()

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

    print(f"\n{'='*15} ENGINE LOADING {'='*15}")
    dataset_type = hypes['validate_dir'].split('/')[-1]

    if parser_opt.type == 'torchscript':
        engine_path = os.path.join(opt.model_dir, "trt_" + dataset_type + '.pt')
        try:
            model = torch.jit.load(engine_path).cuda()
        except:
            print('[ERROR] Unable to load the TorchScript-based model')
    elif parser_opt.type == 'onnx':
        engine_path = os.path.join(opt.model_dir, "trt_" + dataset_type + '.engine')
        try:
            model = trt_utils.TRTEngineWrapper(engine_path)
        except:
            print('[ERROR] Unable to load the ONNX-based model')


    device = torch.device('cuda')

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    progress_bar = tqdm(data_loader,
                        total=len(data_loader),
                        desc='Inference',
                        unit='batch',
                        dynamic_ncols=True)
    for i, batch_data in enumerate(progress_bar):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

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

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections,
                                  hypes['validate_dir'])

if __name__ == '__main__':
    main()
