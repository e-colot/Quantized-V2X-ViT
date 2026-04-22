# builds a model to a tensorRT engine
import torch
import torch_tensorrt
import argparse
import os

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.utils import shape_analysis

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

    # for convenient shape logs naming
    hypes['name'] = modelName

    print('Creating Model')
    model = train_utils.create_model(hypes)

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)

    return model.eval().cuda(), hypes, opt

def build_inputs(hypes):
    shapes = shape_analysis.analyze_shape(hypes)
    device = 'cuda'

    # Extract shapes for voxel_features, voxel_coords, voxel_num_points
    vf_min = shapes['ranges']['voxel_features']['min']
    vf_opt = shapes['ranges']['voxel_features']['opt']
    vf_max = shapes['ranges']['voxel_features']['max']

    vc_min = shapes['ranges']['voxel_coords']['min']
    vc_opt = shapes['ranges']['voxel_coords']['opt']
    vc_max = shapes['ranges']['voxel_coords']['max']

    vnp_min = shapes['ranges']['voxel_num_points']['min']
    vnp_opt = shapes['ranges']['voxel_num_points']['opt']
    vnp_max = shapes['ranges']['voxel_num_points']['max']

    # Get max_cavs from the shape dict (spatial_correction_matrix shape: [1, max_cavs, 4, 4])
    max_cavs = shapes['ranges']['spatial_correction_matrix']['max'][1]

    # Use opt shapes for input batch
    inputs = (
        torch.randn(tuple(vf_opt), dtype=torch.float32).to(device),         # voxel_features
        torch.zeros(tuple(vc_opt), dtype=torch.int32).to(device),           # voxel_coords
        torch.zeros(tuple(vnp_opt), dtype=torch.float32).to(device),        # voxel_num_points
        torch.ones((1,), dtype=torch.int32).to(device),                     # record_len
        torch.randn((1, max_cavs, 4, 4), dtype=torch.float32).to(device),   # spatial_correction_matrix
        torch.randn((1, max_cavs, 3), dtype=torch.float32).to(device)       # prior_encoding
    )

    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=vf_min, 
            opt_shape=vf_opt, 
            max_shape=vf_max, 
            dtype=torch.float32, name="voxel_features"
        ),
        torch_tensorrt.Input(
            min_shape=vc_min, 
            opt_shape=vc_opt, 
            max_shape=vc_max, 
            dtype=torch.int32, name="voxel_coords"
        ),
        torch_tensorrt.Input(
            min_shape=vnp_min, 
            opt_shape=vnp_opt, 
            max_shape=vnp_max, 
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
    inputs, trt_inputs = build_inputs(hypes)

    print(f"\n{'='*15} BUILDING TRT ENGINE {'='*15}")
    print("Tracing model to TorchScript")
    traced_model = torch.jit.trace(model, inputs)
    traced_path = os.path.join(opt.model_dir, "TS_graph.log")
    with open(traced_path, "w") as f:
        graph = traced_model.graph.copy()
        torch._C._jit_pass_inline(graph)
        f.write(str(graph))
        print(f"Saved TorchScript graph to {traced_path}")
    print("Remaining Long/Double nodes:")
    count = 0
    for node in traced_model.graph.nodes():
        for output in node.outputs():
            t = output.type()
            try:
                scalar_type = t.scalarType()
            except Exception:
                continue
            if scalar_type in ('Double', 'Long'):
                src = node.sourceRange() or "unknown source"
                print(f"  {scalar_type}: {node.kind()} | {output.debugName()} | {src}")
                count += 1
    print(f"Total: {count} Long/Double nodes")
    print(f"{'-'*63}")

    print("Compiling TorchScript traced model to TensorRT engine")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=trt_inputs,
        enabled_precisions={torch.float32},
        truncate_long_and_double=False,
        require_full_compilation=True,
        workspace_size=1 << 33,
        allow_shape_tensors=True,
        ir='torchscript'
    )
    print(f"\n{'='*15} ENGINE SUCCESSFULLY BUILT {'='*15}")

    dataset_type = hypes['validate_dir'].split('/')[-1]
    save_path = os.path.join(opt.model_dir, "trt_" + dataset_type + '.pt')
    torch.jit.save(trt_model, save_path)
    print(f'Engine stored in {save_path}')
    print('-' * 52)

if __name__ == '__main__':
    main()
