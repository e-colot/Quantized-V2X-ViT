import torch
import torch_tensorrt

from opencood.utils import shape_analysis


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
    onnx['shapes'] = shapes

    return inputs, torchscript, onnx

def build_onnx_profile(builder, shapes):

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
    max_cavs = shapes['ranges']['spatial_correction_matrix']['max'][1]

    # Optimization profile for dynamic pillar dimension
    profile = builder.create_optimization_profile()
    profile.set_shape("voxel_features",   vf_min,  vf_opt,  vf_max)
    profile.set_shape("voxel_coords",      vc_min,  vc_opt,  vc_max)
    profile.set_shape("voxel_num_points",  vnp_min, vnp_opt, vnp_max)
    # Fixed-shape inputs — min=opt=max
    profile.set_shape("record_len",               [1],              [1],              [1])
    profile.set_shape("spatial_correction_matrix",[1,max_cavs,4,4],[1,max_cavs,4,4],[1,max_cavs,4,4])
    profile.set_shape("prior_encoding",           [1,max_cavs,3],  [1,max_cavs,3],  [1,max_cavs,3])

    return profile

