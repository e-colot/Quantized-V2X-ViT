import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
import tensorrt as trt
import torch


def _parser():
    parser = argparse.ArgumentParser(description="Model selector")
    parser.add_argument('--model', type=str, default='v2xvit')
    parser.add_argument('--type', type=str, default='onnx')
    opt = parser.parse_args()
    return opt

class _Arguments:
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

        assert self.fusion_method in ['late', 'early', 'intermediate']


def load_params(parser_opt=None):
    if parser_opt is None:
        parser_opt = _parser()

    valid_model_names = {
        "v2xvit",
        "ppif" # point pillar intermediate fusion
    }
    if parser_opt.model not in valid_model_names:
        raise ValueError(f"Invalid TRT_STAGE={parser_opt.model}. Use one of {sorted(valid_model_names)}")
    
    valid_compiler_type = {
        "torchscript",
        "onnx",
        "pytorch"
    }
    if parser_opt.type not in valid_compiler_type:
        raise ValueError(f"Invalid TRT_STAGE={parser_opt.type}. Use one of {sorted(valid_compiler_type)}")
    
    opt = _Arguments(parser_opt.model)

    hypes = yaml_utils.load_yaml(None, opt)

    # for convenient shape logs naming
    hypes['name'] = parser_opt.model

    hypes['dataset'] = hypes['validate_dir'].split('/')[-1]

    return hypes, opt, parser_opt


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
