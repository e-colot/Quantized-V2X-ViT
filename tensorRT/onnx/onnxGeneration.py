import os

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

class Arguments:
    model_dir = 'opencood/v2x-vit'
    fusion_method = 'intermediate'
    onnx_path = 'tensorRT/v2xvit.onnx'
    opset = 18


class ONNXInputAdapter(torch.nn.Module):
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
        output_dict = self.model(model_input)
        return output_dict['psm'], output_dict['rm']

def main():
    opt = Arguments()
    assert opt.fusion_method in ['late', 'early', 'intermediate']

    hypes = yaml_utils.load_yaml(None, opt)

    # Shapes and dynamic axes derived from dataset scan.
    input_names = [
        'voxel_features',
        'voxel_coords',
        'voxel_num_points',
        'record_len',
        'prior_encoding',
        'spatial_correction_matrix',
    ]

    num_voxels = torch.export.Dim('num_voxels')
    dynamic_shapes = (
        {0: num_voxels},
        {0: num_voxels},
        {0: num_voxels},
        None,
        None,
        None,
    )

    test_input_shapes = {
        'voxel_features': (11369, 32, 4),
        'voxel_coords': (11369, 4),
        'voxel_num_points': (11369,),
        'record_len': (1,),
        'prior_encoding': (1, 5, 3),
        'spatial_correction_matrix': (1, 5, 4, 4),
    }

    output_names = ['psm', 'rm']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    model = model.cpu()

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    test_input = (
        torch.randn(*test_input_shapes['voxel_features'], dtype=torch.float32),
        torch.zeros(*test_input_shapes['voxel_coords'], dtype=torch.int32),
        torch.ones(*test_input_shapes['voxel_num_points'], dtype=torch.int32),
        torch.ones(*test_input_shapes['record_len'], dtype=torch.int32),
        torch.zeros(*test_input_shapes['prior_encoding'], dtype=torch.float32),
        torch.eye(4, dtype=torch.float32)
        .view(1, 1, 4, 4)
        .repeat(1, 5, 1, 1),
    )

    export_model = ONNXInputAdapter(model)
    export_model.eval()

    os.makedirs(os.path.dirname(opt.onnx_path), exist_ok=True)
    print(f'Exporting ONNX to {opt.onnx_path}')
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            test_input,
            opt.onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            opset_version=opt.opset,
            do_constant_folding=True,
            export_params=True,
        )
    print('ONNX export completed.')

if __name__ == '__main__':
    main()