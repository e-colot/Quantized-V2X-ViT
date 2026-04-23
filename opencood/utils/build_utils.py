import torch
import torch_tensorrt

from opencood.utils import shape_analysis, trt_utils
from opencood.tools import train_utils


def load_model(parser_opt=None):
    hypes, opt, parser_opt = trt_utils.load_params(parser_opt)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)

    return model.eval().cuda(), hypes, opt, parser_opt

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
        torch.zeros(tuple(vnp_opt), dtype=torch.int32).to(device),          # voxel_num_points
        torch.ones((1,), dtype=torch.int32).to(device),                     # record_len
        torch.randn((1, max_cavs, 4, 4), dtype=torch.float32).to(device),   # spatial_correction_matrix
        torch.randn((1, max_cavs, 3), dtype=torch.float32).to(device)       # prior_encoding
    )

    # torchscript specific
    torchscript = {}
    torchscript['trt_inputs'] = [
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
            dtype=torch.int32, name="voxel_num_points"
        ),
        torch_tensorrt.Input(shape=[1], dtype=torch.int32, name="record_len"),
        torch_tensorrt.Input(shape=[1, max_cavs, 4, 4], dtype=torch.float32, name="spatial_correction_matrix"),
        torch_tensorrt.Input(shape=[1, max_cavs, 3],    dtype=torch.float32, name="prior_encoding"),
    ]

    # ONNX specific
    onnx = {}
    onnx['input_names'] = [
        "voxel_features", "voxel_coords", "voxel_num_points",
        "record_len", "spatial_correction_matrix", "prior_encoding",
    ]
    onnx['output_names'] = ["psm", "rm"]
    onnx['dynamic_axes'] = {
        "voxel_features":   {0: "num_pillars"},
        "voxel_coords":     {0: "num_pillars"},
        "voxel_num_points": {0: "num_pillars"},
    }

    return inputs, torchscript, onnx
