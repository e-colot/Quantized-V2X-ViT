import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
import tensorrt as trt
import torch


def _parser():
    parser = argparse.ArgumentParser(description="Model selector")
    parser.add_argument('--model', type=str, default='v2xvit')
    parser.add_argument('--type', type=str, default='onnx')
    parser.add_argument('--show_vis', type=bool, default=False)
    parser.add_argument('--show_sequence', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--save_npy', type=bool, default=False)
    parser.add_argument('--global_sort_detections', type=bool, default=False)
    parser.add_argument('--fusion_method', type=str, default='intermediate')
    opt = parser.parse_args()

    if opt.model == "v2xvit":
        opt.model_dir = 'opencood/logs/v2x-vit'
    elif opt.model == "ppif":
        opt.model_dir = 'opencood/logs/pointPillarIntermediateFusion'

    assert opt.fusion_method in ['late', 'early', 'intermediate']

    return opt

def load_params():
    opt = _parser()

    valid_model_names = {
        "v2xvit",
        "ppif" # point pillar intermediate fusion
    }
    if opt.model not in valid_model_names:
        raise ValueError(f"Invalid TRT_STAGE={opt.model}. Use one of {sorted(valid_model_names)}")
    
    valid_compiler_type = {
        "torchscript",
        "onnx",
        "pytorch"
    }
    if opt.type not in valid_compiler_type:
        raise ValueError(f"Invalid TRT_STAGE={opt.type}. Use one of {sorted(valid_compiler_type)}")

    hypes = yaml_utils.load_yaml(None, opt)

    # for convenient shape logs naming
    hypes['name'] = opt.model

    hypes['dataset'] = hypes['validate_dir'].split('/')[-1]

    # convenient for eval prints
    opt.dataset = hypes['dataset']

    return hypes, opt


class TRTEngineWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.torch_stream = torch.cuda.Stream()
        self.stream = self.torch_stream.cuda_stream

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, voxel_features, voxel_coords, voxel_num_points, record_len, 
                spatial_correction_matrix, prior_encoding):
        
        feed_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len,
            'spatial_correction_matrix': spatial_correction_matrix,
            'prior_encoding': prior_encoding
        }

        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Set input address and shape
                self.context.set_input_shape(name, feed_dict[name].shape)
                self.context.set_tensor_address(name, feed_dict[name].data_ptr())
            else:
                # Get output shape and allocate buffer
                shape = self.context.get_tensor_shape(name)
                out_tensor = torch.empty(tuple(shape), device="cuda", dtype=torch.float32)
                outputs[name] = out_tensor
                self.context.set_tensor_address(name, out_tensor.data_ptr())

        with torch.cuda.stream(self.torch_stream):
            self.context.execute_async_v3(stream_handle=self.stream)
            self.torch_stream.synchronize()
        
        return outputs['psm'], outputs['rm']
