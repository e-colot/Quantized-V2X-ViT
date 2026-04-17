import importlib
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


@dataclass
class Arguments:
    model_dir: str = "opencood/v2x-vit"
    output_dir: str = "tensorRT/split_debug/engines"
    log_dir: str = "tensorRT/split_debug/logs"
    precision: str = "fp32"  # fp16 or fp32

    min_num_voxels: int = 1000
    opt_num_voxels: int = 12000
    max_num_voxels: int = 40000

    max_points_per_voxel: int = 32
    num_point_features: int = 4


class Stage0PreScatter(torch.nn.Module):
    """
    Compile the model up to the point right before PointPillarScatter.
    Returns voxel_coords and pillar_features so scatter can be run outside TRT.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_coords, voxel_num_points):
        pillar_features = self.model.pillar_vfe(voxel_features, voxel_coords, voxel_num_points)
        return voxel_coords, pillar_features


def _sequence_start_indices(opencood_dataset):
    cumulative = list(opencood_dataset.len_record)
    return [0] + cumulative[:-1]


def _sequence_last_indices(opencood_dataset):
    cumulative = list(opencood_dataset.len_record)
    return [x - 1 for x in cumulative]


def _scenario_edge_indices(opencood_dataset):
    starts = _sequence_start_indices(opencood_dataset)
    lasts = _sequence_last_indices(opencood_dataset)
    return list(dict.fromkeys(starts + lasts))


def _extract_model_inputs_from_batch(batch_data):
    ego = batch_data["ego"]
    processed = ego["processed_lidar"]
    return (
        processed["voxel_features"].to(torch.float32),
        processed["voxel_coords"].to(torch.int32),
        processed["voxel_num_points"].to(torch.float32),
    )


def _collect_validation_shapes(hypes):
    dataset = build_dataset(hypes, visualize=False, train=False)
    edge_indices = _scenario_edge_indices(dataset)

    per_sequence_shapes = []
    for dataset_idx in edge_indices:
        sample = dataset[dataset_idx]
        batch_data = dataset.collate_batch_test([sample])
        inputs = _extract_model_inputs_from_batch(batch_data)
        per_sequence_shapes.append(
            {
                "dataset_idx": dataset_idx,
                "voxel_features": tuple(inputs[0].shape),
                "voxel_coords": tuple(inputs[1].shape),
                "voxel_num_points": tuple(inputs[2].shape),
            }
        )

    return dataset, per_sequence_shapes


def _select_trace_dataset_index(per_sequence_shapes, opt):
    return min(
        per_sequence_shapes,
        key=lambda item: abs(item["voxel_features"][0] - opt.opt_num_voxels),
    )["dataset_idx"]


def _apply_voxel_shape_ranges_from_samples(opt, per_sequence_shapes):
    voxel_counts = sorted(s["voxel_features"][0] for s in per_sequence_shapes)
    observed_min = voxel_counts[0]
    observed_max = voxel_counts[-1]

    opt.min_num_voxels = min(opt.min_num_voxels, observed_min)
    opt.max_num_voxels = max(opt.max_num_voxels, observed_max)
    opt.opt_num_voxels = min(max(opt.opt_num_voxels, opt.min_num_voxels), opt.max_num_voxels)

    print(
        "TensorRT voxel profile after dataset sampling: "
        f"min={opt.min_num_voxels}, opt={opt.opt_num_voxels}, max={opt.max_num_voxels}"
    )


def _write_graph_dump(log_dir: str, stage_name: str, traced_module: torch.jit.ScriptModule):
    os.makedirs(log_dir, exist_ok=True)
    graph_path = os.path.join(log_dir, f"{stage_name}.graphs.log")
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(f"[{stage_name}] --- traced.graph ---\n")
        f.write(str(traced_module.graph))
        f.write("\n\n")
        f.write(f"[{stage_name}] --- traced.inlined_graph ---\n")
        f.write(str(traced_module.inlined_graph))
        f.write("\n")
    print(f"[{stage_name}] graph dump written to {graph_path}")


def _enable_terminal_log(log_dir: str, log_filename: str = "terminal_output_pre_scatter.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    class _TeeStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)
    print(f"Terminal output log: {log_path}")
    return original_stdout, original_stderr, log_file


def _trace_module(module: torch.nn.Module, name: str, example_inputs: Sequence[torch.Tensor], log_dir: str):
    traced = torch.jit.trace(module, tuple(example_inputs))
    input_types = [str(inp.type()) for inp in traced.graph.inputs()]
    graph_text = str(traced.graph)
    inlined_graph_text = str(traced.inlined_graph)
    has_cpu_values = (
        ("device=cpu" in graph_text)
        or ("device=cpu" in inlined_graph_text)
        or ('Device = prim::Constant[value="cpu"]' in graph_text)
        or ('Device = prim::Constant[value="cpu"]' in inlined_graph_text)
    )
    print(f"[{name}] graph input types: {input_types}")
    print(f"[{name}] graph has cpu values: {has_cpu_values}")
    _write_graph_dump(log_dir, name, traced)
    return traced


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
            dtype=torch.float32,
        ),
    ]


def main():
    opt = Arguments()
    assert opt.precision in ["fp16", "fp32"]

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    original_stdout, original_stderr, log_file = _enable_terminal_log(opt.log_dir)

    try:
        torch_tensorrt = importlib.import_module("torch_tensorrt")

        print("Loading OpenCOOD config and model...")
        hypes = yaml_utils.load_yaml(None, opt)
        model = train_utils.create_model(hypes).to("cuda")
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        model.eval()

        print("Collecting trace input from validation dataset...")
        dataset, per_sequence_shapes = _collect_validation_shapes(hypes)
        _apply_voxel_shape_ranges_from_samples(opt, per_sequence_shapes)
        trace_idx = _select_trace_dataset_index(per_sequence_shapes, opt)

        sample = dataset[trace_idx]
        batch_data = dataset.collate_batch_test([sample])
        batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
        voxel_features, voxel_coords, voxel_num_points = _extract_model_inputs_from_batch(batch_data)

        stage_name = "stage0_pre_scatter"
        stage = Stage0PreScatter(model).to("cuda").eval()

        traced = _trace_module(
            stage,
            stage_name,
            (voxel_features, voxel_coords, voxel_num_points),
            opt.log_dir,
        )

        enabled_precisions = {torch.float16} if opt.precision == "fp16" else {torch.float32}
        trt_inputs = _build_trt_inputs(opt, torch_tensorrt)

        print(f"[{stage_name}] TRT conversion started")
        trt_module = torch_tensorrt.compile(
            traced,
            ir="ts",
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
            require_full_compilation=False,
            workspace_size=1 << 33,
        )

        module_path = os.path.join(opt.output_dir, f"{stage_name}.ts")
        torch.jit.save(trt_module, module_path)
        print(f"[{stage_name}] TRT conversion succeeded -> {module_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
