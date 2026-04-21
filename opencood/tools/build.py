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

def load_model(opt=None):
    if opt is not None:
        modelName = opt
    else:
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

    used_dataset = hypes['validate_dir'].split('/')[-1]
    if used_dataset == 'validate':
        opt.min_v = 4707
        opt.opt_v = 11763
        opt.max_v = 18658
        opt.opt_cavs = 3
    elif used_dataset == 'test':
        opt.min_v = 3662
        opt.opt_v = 12793
        opt.max_v = 26295
        opt.opt_cavs = 3
    else:
        raise NotImplementedError

    return model.eval().cuda(), hypes, opt

def build_inputs(hypes, opt):
    max_cavs = hypes['model']['args']['max_cav']
    device = 'cuda'

    inputs = (
        torch.randn((opt.opt_v, 32, 4), dtype=torch.float32).to(device),        # voxel_features
        torch.zeros((opt.opt_v, 4), dtype=torch.float32).to(device),            # voxel_coords
        torch.zeros((opt.opt_v,), dtype=torch.float32).to(device),              # voxel_num_points
        torch.ones((1,), dtype=torch.int32).to(device),                         # record_len
        torch.randn((1, max_cavs, 4, 4), dtype=torch.float32).to(device),       # spatial_correction_matrix
        torch.randn((1, max_cavs, 3), dtype=torch.float32).to(device)           # prior_encoding
    )

    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=[opt.min_v, 32, 4], 
            opt_shape=[opt.opt_v, 32, 4], 
            max_shape=[opt.max_v, 32, 4], 
            dtype=torch.float32, name="voxel_features"
        ),
        torch_tensorrt.Input(
            min_shape=[opt.min_v, 4], 
            opt_shape=[opt.opt_v, 4], 
            max_shape=[opt.max_v, 4], 
            dtype=torch.float32, name="voxel_coords"
        ),
        torch_tensorrt.Input(
            min_shape=[opt.min_v], 
            opt_shape=[opt.opt_v], 
            max_shape=[opt.max_v], 
            dtype=torch.float32, name="voxel_num_points"
        ),
        torch_tensorrt.Input(shape=[1], dtype=torch.int32, name="record_len"),
        torch_tensorrt.Input(shape=[1, max_cavs, 4, 4], dtype=torch.float32, name="spatial_correction_matrix"),
        torch_tensorrt.Input(shape=[1, max_cavs, 3],    dtype=torch.float32, name="prior_encoding"),
    ]

    return inputs, trt_inputs

def main(opt=None):
    torch.manual_seed(0)

    model, hypes, opt = load_model(opt)
    inputs, trt_inputs = build_inputs(hypes, opt)

    print(f"{'='*15} BUILDING TRT ENGINE {'='*15}")
    print("Tracing model to TorchScript")
    traced_model = torch.jit.trace(model, inputs)
    traced_path = os.path.join(opt.model_dir, "TS_graph.log")
    with open(traced_path, "w") as f:
        f.write(str(traced_model.graph))
        print(f"Saved TorchScript graph to {traced_path}")

    print(f"{'-'*63}\n")

    print("Compiling TorchScript traced model to TensorRT engine")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=trt_inputs, # trt_inputs still defines min/opt/max ranges
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
        require_full_compilation=True,
        workspace_size=1 << 30,
        allow_shape_tensors=True,
        ir='torchscript'
    )
    trt_graph_path = os.path.join(opt.model_dir, "TRT_graph.log")
    with open(trt_graph_path, "w") as f:
        f.write(str(trt_model.graph))
        print(f"Saved TensorRT engine graph to {trt_graph_path}")

    print(f"\n{'='*15} ENGINE SUCCESSFULLY BUILT {'='*15}")

    save_path = os.path.join(opt.model_dir, "trt.pt")
    torch.jit.save(trt_model, save_path)
    print(f'Engine stored in {save_path}')

if __name__ == '__main__':
    main()
