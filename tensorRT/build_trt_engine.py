import os
import importlib
from typing import Optional

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


class Arguments:
    def __init__(self):
        self.model_dir = 'opencood/v2x-vit'
        self.fusion_method = 'intermediate'
        self.engine_path = 'tensorRT/v2xvit_fp32.engine'
        self.graph_log_path = 'tensorRT/build_trt_engine.graphs.log'
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
        self.try_script_first = False
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

        # Forward through: pillar_vfe -> scatter -> backbone -> fusion -> heads
        # (no post-processing applied here)
        output_dict = self.model(voxel_features, voxel_coords, voxel_num_points, record_len, 
                spatial_correction_matrix, prior_encoding)
        # Return only the head outputs (classification and regression)
        # Post-processing happens in Python after TensorRT inference
        return output_dict['psm'], output_dict['rm']


def _sequence_start_indices(opencood_dataset):
    # len_record stores cumulative frame counts per scenario.
    cumulative = list(opencood_dataset.len_record)
    return [0] + cumulative[:-1]


def _sequence_last_indices(opencood_dataset):
    cumulative = list(opencood_dataset.len_record)
    return [x - 1 for x in cumulative]


def _scenario_edge_indices(opencood_dataset):
    starts = _sequence_start_indices(opencood_dataset)
    lasts = _sequence_last_indices(opencood_dataset)
    # Preserve order while deduplicating (single-frame scenarios).
    return list(dict.fromkeys(starts + lasts))


def _extract_model_inputs_from_batch(batch_data):
    ego = batch_data['ego']
    processed = ego['processed_lidar']
    return (
        processed['voxel_features'].to(torch.float32),
        processed['voxel_coords'].to(torch.int32),
        processed['voxel_num_points'].to(torch.int32),
        ego['record_len'].to(torch.int32),
        ego['prior_encoding'].to(torch.float32),
        ego['spatial_correction_matrix'].to(torch.float32),
    )


def _collect_validation_shapes(hypes):
    dataset = build_dataset(hypes, visualize=False, train=False)
    edge_indices = _scenario_edge_indices(dataset)

    per_sequence_shapes = []
    for dataset_idx in edge_indices:
        sample = dataset[dataset_idx]
        batch_data = dataset.collate_batch_test([sample])
        inputs = _extract_model_inputs_from_batch(batch_data)
        per_sequence_shapes.append({
            'dataset_idx': dataset_idx,
            'voxel_features': tuple(inputs[0].shape),
            'voxel_coords': tuple(inputs[1].shape),
            'voxel_num_points': tuple(inputs[2].shape),
            'record_len': tuple(inputs[3].shape),
            'prior_encoding': tuple(inputs[4].shape),
            'spatial_correction_matrix': tuple(inputs[5].shape),
        })

    return dataset, per_sequence_shapes


def _select_trace_dataset_index(per_sequence_shapes, opt):
    # Pick a real validation sample closest to TRT opt voxel count.
    return min(
        per_sequence_shapes,
        key=lambda item: abs(item['voxel_features'][0] - opt.opt_num_voxels),
    )['dataset_idx']


def _print_shape_summary(per_sequence_shapes):
    voxel_counts = [s['voxel_features'][0] for s in per_sequence_shapes]
    coord_counts = [s['voxel_coords'][0] for s in per_sequence_shapes]
    point_counts = [s['voxel_num_points'][0] for s in per_sequence_shapes]
    print(f'Collected first+last samples from each validation sequence: {len(per_sequence_shapes)} samples')
    print(
        'Voxel dim-0 ranges from dataset samples: '
        f'features={min(voxel_counts)}..{max(voxel_counts)}, '
        f'coords={min(coord_counts)}..{max(coord_counts)}, '
        f'num_points={min(point_counts)}..{max(point_counts)}'
    )


def _apply_voxel_shape_ranges_from_samples(opt, per_sequence_shapes):
    voxel_counts = sorted(s['voxel_features'][0] for s in per_sequence_shapes)
    observed_min = voxel_counts[0]
    observed_max = voxel_counts[-1]
    observed_opt = voxel_counts[len(voxel_counts) // 2]

    # Expand TRT dynamic profile to include observed first+last frame shapes.
    opt.min_num_voxels = min(opt.min_num_voxels, observed_min)
    opt.max_num_voxels = max(opt.max_num_voxels, observed_max)
    opt.opt_num_voxels = min(max(opt.opt_num_voxels, opt.min_num_voxels), opt.max_num_voxels)

    print(
        'TensorRT voxel profile after dataset sampling: '
        f'min={opt.min_num_voxels}, opt={opt.opt_num_voxels}, max={opt.max_num_voxels} '
        f'(observed median={observed_opt})'
    )


def _build_trt_inputs(opt, torch_tensorrt):
    return [
        # 1. voxel_features
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            opt_shape=(opt.opt_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            max_shape=(opt.max_num_voxels, opt.max_points_per_voxel, opt.num_point_features),
            dtype=torch.float32,
            name="voxel_features"
        ),
        # 2. voxel_coords
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels, 4),
            opt_shape=(opt.opt_num_voxels, 4),
            max_shape=(opt.max_num_voxels, 4),
            dtype=torch.int32,
            name="voxel_coords"
        ),
        # 3. voxel_num_points (THE CRITICAL ONE)
        torch_tensorrt.Input(
            min_shape=(opt.min_num_voxels,),
            opt_shape=(opt.opt_num_voxels,),
            max_shape=(opt.max_num_voxels,),
            dtype=torch.float32, 
            name="voxel_num_points"
        ),
        # 4. record_len
        torch_tensorrt.Input(shape=(1,), dtype=torch.int32, name="record_len"),
        # 5. prior_encoding
        torch_tensorrt.Input(shape=(1, opt.max_cavs, 3), dtype=torch.float32, name="prior_encoding"),
        # 6. spatial_correction_matrix (Matches your inference call order)
        torch_tensorrt.Input(shape=(1, opt.max_cavs, 4, 4), dtype=torch.float32, name="spatial_correction_matrix"),
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


def _write_graph_dump(graph_log_path: str, traced_module: torch.jit.ScriptModule):
    os.makedirs(os.path.dirname(graph_log_path), exist_ok=True)
    with open(graph_log_path, 'w', encoding='utf-8') as f:
        f.write('--- traced.graph ---\n')
        f.write(str(traced_module.graph))
        f.write('\n\n')
        f.write('--- traced.inlined_graph ---\n')
        f.write(str(traced_module.inlined_graph))
        f.write('\n')
    print(f'TorchScript graph dump written to: {graph_log_path}')

def _prepare_torchscript_module(adapter, opt, hypes):
    # Try Scripting first if enabled
    if opt.try_script_first:
        scripted_adapter = _try_script_module(adapter, 'adapter')
        if scripted_adapter is not None:
            return scripted_adapter
        print("Scripting failed, falling back to tracing...")

    # Tracing fallback with real validation data (1 sample per sequence for shape collection).
    print('Collecting trace shapes from validation dataset...')
    dataset, per_sequence_shapes = _collect_validation_shapes(hypes)
    # _print_shape_summary(per_sequence_shapes)
    _apply_voxel_shape_ranges_from_samples(opt, per_sequence_shapes)

    trace_idx = _select_trace_dataset_index(per_sequence_shapes, opt)
    print(f'Tracing the adapter using validation dataset index: {trace_idx}')
    trace_sample = dataset[trace_idx]
    trace_batch = dataset.collate_batch_test([trace_sample])
    trace_batch = train_utils.to_device(trace_batch, torch.device('cuda'))
    trace_inputs = _extract_model_inputs_from_batch(trace_batch)

    try:
        # Trace the model
        traced_adapter = torch.jit.trace(adapter, trace_inputs)

# ----------------------- START GRAPH CLEANUP -----------------------
        # 1. Inline everything to resolve scope-based naming collisions
        torch._C._jit_pass_inline(traced_adapter.graph)
        # 2. CRITICAL: De-fuse cuDNN ops back into standard Conv + ReLU
        # This prevents the 'aten::cudnn_convolution_relu' error
        pattern = """
            graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups):
                %res = aten::cudnn_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
                return (%res)
        """
        replacement = """
            graph(%input, %weight, %bias, %stride, %padding, %dilation, %groups):
                %transposed : bool = prim::Constant[value=0]()
                %conv = aten::convolution(%input, %weight, %bias, %stride, %padding, %dilation, %transposed, %groups)
                %res = aten::relu(%conv)
                return (%res)
        """
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement, traced_adapter.graph)
# ----------------------- GRAPH CLEANUP DONE -----------------------
        
        print("Successfully traced the adapter.")

        _write_graph_dump(opt.graph_log_path, traced_adapter)

        # Search specifically for the 'item' node in the graph string
        graph_str = str(traced_adapter.graph)
        if "aten::item" in graph_str:
            print("\n!!! FOUND aten::item in the graph !!!")

        return traced_adapter
    except Exception as e:
        print(f"Tracing failed: {e}")
        raise RuntimeError("Both scripting and tracing failed. Check model for incompatible ops.")


def main():
    opt = Arguments()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert opt.precision in ['fp16', 'fp32']

    torch_tensorrt = importlib.import_module('torch_tensorrt')

    device = torch.device('cuda')

    print('Loading OpenCOOD config and model...')
    hypes = yaml_utils.load_yaml(None, opt)
    model = train_utils.create_model(hypes).to('cuda')
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
    adapter = TRTInputAdapter(model).to('cuda').eval()
    scripted = _prepare_torchscript_module(adapter, opt, hypes)

    enabled_precisions = {torch.float16} if opt.precision == 'fp16' else {torch.float32}
    trt_inputs = _build_trt_inputs(opt, torch_tensorrt)

    print('Converting model using unified compile API...')
    
    # trt_inputs is already built correctly in your script
    
    trt_model = torch_tensorrt.compile(
        scripted,
        ir="ts", # This tells it to use the TorchScript backend
        inputs=trt_inputs,
        enabled_precisions=enabled_precisions,
        truncate_long_and_double=True,
        require_full_compilation=False,
        workspace_size=1 << 33, # 8GB
    )

    # Instead of engine_bytes, we save the compiled module
    print(f'Saving compiled TRT model to {opt.engine_path}.ts')
    torch.jit.save(trt_model, opt.engine_path + ".ts")


if __name__ == '__main__':
    main()
