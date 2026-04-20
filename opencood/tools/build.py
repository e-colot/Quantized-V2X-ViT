# builds a model to a tensorRT engine
import torch
import torch_tensorrt
import argparse
import json
import os

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

from opencood.data_utils.datasets import build_dataset

def parser():
    parser = argparse.ArgumentParser(description="Model selector")
    parser.add_argument('--model', type=str,
                        required=True)
    opt = parser.parse_args()
    return str(opt.model)

class Arguments:
    def __init__(self, modelName):
        print('Default parameters used')
        self.model_name = modelName
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

def load_model():
    modelName = parser()
    valid_model_names = {
        "v2xvit",
        "ppif" # point pillar intermediate fusion
    }

    if modelName not in valid_model_names:
        raise ValueError(f"Invalid TRT_STAGE={modelName}. Use one of {sorted(valid_model_names)}")
    
    opt = Arguments(modelName)

    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)

    return model.eval().cuda(), hypes, opt

def build_inputs(hypes):

    min_v = 4712
    opt_v = 11769
    max_v = 18793
    max_cavs = hypes['model']['args']['max_cav']
    opt_cavs = 3
    device = 'cuda'

    inputs = (
        torch.randn((opt_v, 32, 4), dtype=torch.float32).to(device),      # voxel_features
        torch.zeros((opt_v, 4), dtype=torch.int32).to(device),            # voxel_coords
        torch.zeros((opt_v,), dtype=torch.int32).to(device),              # voxel_num_points
        torch.ones((opt_cavs,), dtype=torch.int32).to(device),                   # record_len
        torch.randn((1, 7, 4, 4), dtype=torch.float32).to(device),        # spatial_correction_matrix
        torch.randn((1, 7, 3), dtype=torch.float32).to(device)            # prior_encoding
    )

    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=[min_v, 32, 4], 
            opt_shape=[opt_v, 32, 4], 
            max_shape=[max_v, 32, 4], 
            dtype=torch.float32, name="voxel_features"
        ),
        torch_tensorrt.Input(
            min_shape=[min_v, 4], 
            opt_shape=[opt_v, 4], 
            max_shape=[max_v, 4], 
            dtype=torch.int32, name="voxel_coords"
        ),
        torch_tensorrt.Input(
            min_shape=[min_v], 
            opt_shape=[opt_v], 
            max_shape=[max_v], 
            dtype=torch.int32, name="voxel_num_points"
        ),
        torch_tensorrt.Input(shape=[1], dtype=torch.int32, name="record_len"),
        torch_tensorrt.Input(shape=[1, 7, 4, 4], dtype=torch.float32, name="spatial_correction_matrix"),
        torch_tensorrt.Input(shape=[1, 7, 3],    dtype=torch.float32, name="prior_encoding"),
    ]

    return inputs, trt_inputs

def main():
    torch.manual_seed(0)

    model, hypes, opt = load_model()
    inputs, trt_inputs = build_inputs(hypes)

    print("Tracing model to TorchScript")
    traced_model = torch.jit.trace(model, inputs)

    print("Compiling with TorchScript backend")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=trt_inputs, # trt_inputs still defines your min/opt/max ranges
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
        workspace_size=1 << 30,
        ir='torchscript'
    )

    print('Engine successfully built')

    save_path = os.path.join(opt.model_dir, "trt.pt")
    torch.jit.save(trt_model, save_path)
    print(f'Engine stored in {save_path}')

if __name__ == '__main__':
    main()
