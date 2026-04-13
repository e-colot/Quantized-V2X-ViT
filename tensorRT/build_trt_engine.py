import os
import importlib
from typing import Optional

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils


class Arguments:
    def __init__(self):
        self.model_dir = 'opencood/v2x-vit'
        self.fusion_method = 'intermediate'
        self.engine_path = 'tensorRT/v2xvit_fp16.engine'
        self.precision = 'fp32'  # 'fp16' or 'fp32'
        self.device = 'cuda:0'

        # Dynamic range for num_voxels (dim 0 of lidar tensors).
        self.min_num_voxels = 1000
        self.opt_num_voxels = 12000
        self.max_num_voxels = 40000

        # Fixed shapes used by this model path.
        self.max_points_per_voxel = 32
        self.num_point_features = 4
        self.max_cavs = 5

        # Script-first path can remove some trace-only Python artifacts.
        self.try_script_first = True
        self.try_script_submodules = False


class TRTInputAdapter(torch.nn.Module):
    """
    Adapter for TensorRT conversion that wraps the model to expose only
    backbone + fusion + head outputs. Post-processing (anchor decoding, NMS, etc.)
    is kept in Python and will be applied after TensorRT inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        voxel_features,
        voxel_coords,
        voxel_num_points,
        record_len,
        prior_encoding,
        spatial_correction_matrix,
    ):
        """
        Forward pass through backbone + fusion + head only.
        
        Returns:
            psm: Classification predictions (raw, pre-post-processing)
            rm: Regression predictions (raw, pre-post-processing)
        """
        model_input = {
            'processed_lidar': {
                'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points,
            },
            'record_len': record_len,
            'prior_encoding': prior_encoding,
            'spatial_correction_matrix': spatial_correction_matrix,
        }
        # Forward through: pillar_vfe -> scatter -> backbone -> fusion -> heads
        # (no post-processing applied here)
        output_dict = self.model(model_input)
        # Return only the head outputs (classification and regression)
        # Post-processing happens in Python after TensorRT inference
        return output_dict['psm'], output_dict['rm']


def _build_example_inputs(opt, device):
    voxels = opt.opt_num_voxels
    return (
        torch.randn(
            voxels,
            opt.max_points_per_voxel,
            opt.num_point_features,
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros(voxels, 4, dtype=torch.int32, device=device),
        torch.ones(voxels, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros(1, opt.max_cavs, 3, dtype=torch.float32, device=device),
        torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 4, 4).repeat(1, opt.max_cavs, 1, 1),
    )


def _build_trt_inputs(opt, torch_tensorrt):
    return [
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            opt_shape=(opt.opt_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            max_shape=(opt.max_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            dtype=torch.float32,
        ),
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels, 4),
            opt_shape=(opt.opt_num_voxels, 4),
            max_shape=(opt.max_num_voxels, 4),
            dtype=torch.int32,
        ),
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels,),
            opt_shape=(opt.opt_num_voxels,),
            max_shape=(opt.max_num_voxels,),
            dtype=torch.int32,
        ),
        torch_tensorrt.Input(shape=(1,), dtype=torch.int32),
        torch_tensorrt.Input(shape=(1, opt.max_cavs, 3), dtype=torch.float32),
        torch_tensorrt.Input(shape=(1, opt.max_cavs, 4, 4), dtype=torch.float32),
    ]


def _try_script_module(module, module_name: str) -> Optional[torch.jit.ScriptModule]:
    try:
        scripted = torch.jit.script(module)
        scripted = torch.jit.freeze(scripted)
        print(f'Successfully scripted {module_name}')
        return scripted
    except Exception as exc:
        print(f'Could not script {module_name}: {exc.__class__.__name__}: {exc}')
        return None


def _prepare_torchscript_module(adapter, example_inputs, opt):
    if opt.try_script_first:
        scripted_adapter = _try_script_module(adapter, 'adapter')
        if scripted_adapter is not None:
            return scripted_adapter

    print('Falling back to torch.jit.trace for adapter...')
    with torch.no_grad():
        traced = torch.jit.trace(adapter, example_inputs, strict=False)
        traced = torch.jit.freeze(traced)
    return traced


def main():
    opt = Arguments()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert opt.precision in ['fp16', 'fp32']

    try:
        torch_tensorrt = importlib.import_module('torch_tensorrt')
    except ImportError as exc:
        raise ImportError(
            'torch_tensorrt is required for direct Torch->TensorRT conversion. '
            'Install it in your environment before running this script.'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to build a TensorRT engine.')

    device = torch.device(opt.device)

    print('Loading OpenCOOD config and model...')
    hypes = yaml_utils.load_yaml(None, opt)
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    if opt.try_script_submodules:
        print('Attempting to script the whole model before adapter export...')
        scripted_model = _try_script_module(model, 'model')
        if scripted_model is not None:
            model = scripted_model
        else:
            print('Whole-model scripting failed; continuing with eager model.')

    print('Preparing TorchScript adapter')
    adapter = TRTInputAdapter(model).to(device).eval()
    example_inputs = _build_example_inputs(opt, device)
    scripted = _prepare_torchscript_module(adapter, example_inputs, opt)

    enabled_precisions = {torch.float16} if opt.precision == 'fp16' else {torch.float32}
    trt_inputs = _build_trt_inputs(opt, torch_tensorrt)

    print('Converting TorchScript module to TensorRT engine bytes...')
    engine_bytes = torch_tensorrt.ts.convert_method_to_trt_engine(
        scripted,
        'forward',
        inputs=trt_inputs,
        enabled_precisions=enabled_precisions,
    )

    os.makedirs(os.path.dirname(opt.engine_path), exist_ok=True)
    with open(opt.engine_path, 'wb') as f:
        f.write(engine_bytes)

    print(f'TensorRT engine saved to: {opt.engine_path}')


if __name__ == '__main__':
    main()
