import json
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import get_roi_and_cav_mask
from opencood.tools import train_utils


@dataclass
class Arguments:
    model_dir: str = "opencood/v2x-vit"
    output_dir: str = "tensorRT/split_debug/engines"
    log_dir: str = "tensorRT/split_debug/logs"
    validation_shape_log_path: str = "tensorRT/split_debug/build_split_trt_engines.validation_shapes.log"
    precision: str = "fp32"  # fp16 or fp32
    max_cavs: int = 5

    min_num_voxels: int = 1000
    opt_num_voxels: int = 12000
    max_num_voxels: int = 40000

    max_points_per_voxel: int = 32
    num_point_features: int = 4


class StageBackbone(torch.nn.Module):
    def __init__(self, model, record_len_template: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("record_len", record_len_template.to(torch.int32))

    def forward(self, voxel_features, voxel_coords, voxel_num_points):
        pillar_features = self.model.pillar_vfe(voxel_features, voxel_coords, voxel_num_points)
        spatial_features = self.model.scatter(voxel_coords, pillar_features)
        spatial_features_2d = self.model.backbone(spatial_features)
        if self.model.shrink_flag:
            spatial_features_2d = self.model.shrink_conv(spatial_features_2d)
        if self.model.compression:
            spatial_features_2d = self.model.naive_compressor(spatial_features_2d)
        return spatial_features_2d


class Stage1aPillarVFE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_features, voxel_coords, voxel_num_points):
        return self.model.pillar_vfe(voxel_features, voxel_coords, voxel_num_points)


class Stage1bScatter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxel_coords, pillar_features):
        return self.model.scatter(voxel_coords, pillar_features)


class Stage1cBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial_features):
        return self.model.backbone(spatial_features)


class Stage1dNeck(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial_features_2d):
        out = spatial_features_2d
        if self.model.shrink_flag:
            out = self.model.shrink_conv(out)
        if self.model.compression:
            out = self.model.naive_compressor(out)
        return out


class StageRegroup(torch.nn.Module):
    def __init__(self, max_cav: int, record_len_template: torch.Tensor):
        super().__init__()
        self.max_cav = max_cav
        self.register_buffer("record_len", record_len_template.to(torch.int32))

    def forward(self, spatial_features_2d, prior_encoding):
        regroup_feature, mask = regroup(spatial_features_2d, self.record_len, self.max_cav)
        prior_encoding = prior_encoding.unsqueeze(-1).unsqueeze(-1)
        prior_encoding = prior_encoding.repeat(
            1,
            1,
            1,
            regroup_feature.shape[3],
            regroup_feature.shape[4],
        )
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        return regroup_feature, mask


class StageFusion(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, regroup_feature, mask, spatial_correction_matrix):
        fused_feature = self.model.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        return fused_feature


class Stage3aFusionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, regroup_feature, mask, spatial_correction_matrix):
        return self.model.fusion_net.encoder(regroup_feature, mask, spatial_correction_matrix)


class Stage3aRTE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, regroup_feature):
        encoder = self.model.fusion_net.encoder
        prior_encoding = regroup_feature[..., -3:]
        x = regroup_feature[..., :-3]
        if encoder.use_RTE:
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int32)
            x = encoder.rte(x, dt)
        return x, prior_encoding


class Stage3bSTTF(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, spatial_correction_matrix):
        encoder = self.model.fusion_net.encoder
        return encoder.sttf(x, spatial_correction_matrix)


class Stage3cFusionBlocks(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, prior_encoding, mask, spatial_correction_matrix):
        encoder = self.model.fusion_net.encoder
        if not encoder.use_roi_mask:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        else:
            com_mask = get_roi_and_cav_mask(
                x,
                mask,
                spatial_correction_matrix,
                encoder.discrete_ratio,
                encoder.downsample_rate,
            )

        for layer in encoder.layers:
            # Isolate V2XFusionBlocks (layer[0]) only.
            x = layer[0](x, mask=com_mask, prior_encoding=prior_encoding)

        return x


class Stage3cHGTCavAttentionDebug(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, prior_encoding, mask, spatial_correction_matrix):
        encoder = self.model.fusion_net.encoder
        if not encoder.use_roi_mask:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        else:
            com_mask = get_roi_and_cav_mask(
                x,
                mask,
                spatial_correction_matrix,
                encoder.discrete_ratio,
                encoder.downsample_rate,
            )

        # Debug-only stage: isolate the first CAV attention wrapper
        hgt_att = encoder.layers[0][0].layers[0][0]
        return hgt_att(x, mask=com_mask, prior_encoding=prior_encoding)


class Stage3dPyramidWindowAttentionDebug(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Debug-only stage: isolate the first pyramid window attention wrapper
        pwin_att = self.model.fusion_net.encoder.layers[0][0].layers[0][1]
        return pwin_att(x)


class Stage3eFusionBlocks(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, prior_encoding, mask, spatial_correction_matrix):
        encoder = self.model.fusion_net.encoder
        if not encoder.use_roi_mask:
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        else:
            com_mask = get_roi_and_cav_mask(
                x,
                mask,
                spatial_correction_matrix,
                encoder.discrete_ratio,
                encoder.downsample_rate,
            )

        for layer in encoder.layers:
            # Isolate V2XFusionBlocks (layer[0]) only.
            x = layer[0](x, mask=com_mask, prior_encoding=prior_encoding)

        return x


class Stage3fFusionPost(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        encoder = self.model.fusion_net.encoder
        for layer in encoder.layers:
            x = layer[1](x) + x
        fused_feature = x[:, 0]
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        return fused_feature


class Stage5Heads(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fused_feature):
        psm = self.model.cls_head(fused_feature)
        rm = self.model.reg_head(fused_feature)
        return psm, rm


def _extract_model_inputs_from_batch(batch_data):
    ego = batch_data["ego"]
    processed = ego["processed_lidar"]
    return (
        processed["voxel_features"].to(torch.float32),
        processed["voxel_coords"].to(torch.int32),
        processed["voxel_num_points"].to(torch.float32),
        ego["record_len"].to(torch.int32),
        ego["prior_encoding"].to(torch.float32),
        ego["spatial_correction_matrix"].to(torch.float32),
    )


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


def _load_validation_shapes_cache(validation_shape_log_path):
    if not os.path.exists(validation_shape_log_path):
        return None

    with open(validation_shape_log_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    shapes = cached.get("per_sequence_shapes")
    if not isinstance(shapes, list):
        return None

    print(f"Reusing cached validation shape log: {validation_shape_log_path}")
    return shapes


def _write_validation_shapes_cache(validation_shape_log_path, per_sequence_shapes):
    os.makedirs(os.path.dirname(validation_shape_log_path), exist_ok=True)
    with open(validation_shape_log_path, "w", encoding="utf-8") as f:
        json.dump({"per_sequence_shapes": per_sequence_shapes}, f, indent=2)
    print(f"Validation shape log written to: {validation_shape_log_path}")


def _collect_validation_shapes(hypes, validation_shape_log_path):
    dataset = build_dataset(hypes, visualize=False, train=False)
    cached_per_sequence_shapes = _load_validation_shapes_cache(validation_shape_log_path)
    if cached_per_sequence_shapes is not None:
        return dataset, cached_per_sequence_shapes

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
                "record_len": tuple(inputs[3].shape),
                "prior_encoding": tuple(inputs[4].shape),
                "spatial_correction_matrix": tuple(inputs[5].shape),
            }
        )

    _write_validation_shapes_cache(validation_shape_log_path, per_sequence_shapes)
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


def _clean_logs_dir(log_dir: str, preserve: Sequence[str] = ()): 
    os.makedirs(log_dir, exist_ok=True)
    preserve_set = set(preserve)
    for name in os.listdir(log_dir):
        if name in preserve_set:
            continue
        path = os.path.join(log_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    print("Cleaned previous logs")


def _enable_terminal_log(log_dir: str, log_filename: str = "terminal_output.log"):
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
    print(f"[{name}] trace complete.")
    _write_graph_dump(log_dir, name, traced)
    return traced, has_cpu_values


def _try_convert_stage(
    torch_tensorrt,
    traced_module,
    stage_name: str,
    trt_inputs: List,
    enabled_precisions: set,
    output_dir: str,
    require_full_compilation: bool,
):
    print(f"[{stage_name}] TRT conversion started")
    try:
        trt_module = torch_tensorrt.compile(
            traced_module,
            ir="ts",
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
            require_full_compilation=require_full_compilation,
            workspace_size=1 << 33,
        )
        module_path = os.path.join(output_dir, f"{stage_name}.ts")
        torch.jit.save(trt_module, module_path)
        print(f"[{stage_name}] TRT conversion succeeded -> {module_path}")
        return True, ""
    except Exception as exc:
        print(f"[{stage_name}] TRT conversion FAILED: {exc}")
        return False, str(exc)


def _build_inputs_backbone(opt, torch_tensorrt):
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
    ]

def _build_trt_inputs_from_example_tensors(torch_tensorrt, example_inputs):
    trt_inputs = []
    for x in example_inputs:
        # Check if it's an integer tensor or boolean
        if x.dtype == torch.int64 or x.dtype == torch.bool:
            dtype = torch.int32
        else:
            dtype = x.dtype

        # Use an explicit profile even for fixed-size tensors to avoid
        # converter shape-propagation paths that can materialize empty tensors.
        shape = tuple(int(d) for d in x.shape)
        trt_inputs.append(torch_tensorrt.Input(
            min_shape=shape,
            opt_shape=shape,
            max_shape=shape,
            dtype=dtype,
        ))
    return trt_inputs


def _build_inputs_regroup(torch_tensorrt, n_min, n_opt, n_max, c, h, w, max_cavs):
    return [
        torch_tensorrt.Input(
            min_shape=(n_min, c, h, w),
            opt_shape=(n_opt, c, h, w),
            max_shape=(n_max, c, h, w),
            dtype=torch.float32,
        ),
        torch_tensorrt.Input(shape=(1, max_cavs, 3), dtype=torch.float32),
    ]


def _build_inputs_fusion(torch_tensorrt, max_cavs, h, w, c_total):
    return [
        torch_tensorrt.Input(shape=(1, max_cavs, h, w, c_total), dtype=torch.float32),
        torch_tensorrt.Input(shape=(1, max_cavs), dtype=torch.float32),
        torch_tensorrt.Input(shape=(1, max_cavs, 4, 4), dtype=torch.float32),
    ]


def _build_inputs_heads(torch_tensorrt, h, w, c):
    return [torch_tensorrt.Input(shape=(1, c, h, w), dtype=torch.float32)]


def main():
    opt = Arguments()
    assert opt.precision in ["fp16", "fp32"]
    selected_stage = os.environ.get("TRT_STAGE", "all")
    require_full_compilation = os.environ.get("TRT_REQUIRE_FULL_COMPILATION", "1") not in {
        "0",
        "false",
        "False",
        "no",
        "NO",
    }
    valid_stage_names = {
        "all",
        "stage1a_pillar_vfe",
        "stage1b_scatter",
        "stage1c_backbone",
        "stage1d_neck",
        "stage2_regroup",
        "stage3a_rte",
        "stage3b_sttf",
        "stage3c_hgt_cav_attention_debug",
        "stage3d_pyramid_window_attention_debug",
        "stage3e_fusion_blocks",
        "stage3f_fusion_post",
        "stage5_heads",
    }
    if selected_stage not in valid_stage_names:
        raise ValueError(f"Invalid TRT_STAGE={selected_stage}. Use one of {sorted(valid_stage_names)}")

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    terminal_log_name = "terminal_output.log"
    original_stdout, original_stderr, log_file = _enable_terminal_log(opt.log_dir, terminal_log_name)

    try:
        _clean_logs_dir(opt.log_dir, preserve=(terminal_log_name,))
        torch_tensorrt = importlib.import_module("torch_tensorrt")
        print(f"TRT require_full_compilation={require_full_compilation}")

        print("Loading OpenCOOD config and model...")
        hypes = yaml_utils.load_yaml(None, opt)
        model = train_utils.create_model(hypes).to("cuda")
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        model.eval()

        print("Collecting trace input from validation dataset...")
        dataset, per_sequence_shapes = _collect_validation_shapes(hypes, opt.validation_shape_log_path)
        _apply_voxel_shape_ranges_from_samples(opt, per_sequence_shapes)
        trace_idx = _select_trace_dataset_index(per_sequence_shapes, opt)

        sample = dataset[trace_idx]
        batch_data = dataset.collate_batch_test([sample])
        batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
        (
            voxel_features,
            voxel_coords,
            voxel_num_points,
            record_len,
            prior_encoding,
            spatial_correction_matrix,
        ) = _extract_model_inputs_from_batch(batch_data)

        with torch.no_grad():
            stage1a_pillar_vfe = Stage1aPillarVFE(model).to("cuda").eval()
            pillar_features = stage1a_pillar_vfe(voxel_features, voxel_coords, voxel_num_points)

            stage1b_scatter = Stage1bScatter(model).to("cuda").eval()
            spatial_features = stage1b_scatter(voxel_coords, pillar_features)

            stage1c_backbone = Stage1cBackbone(model).to("cuda").eval()
            spatial_features_2d_backbone = stage1c_backbone(spatial_features)

            stage1d_neck = Stage1dNeck(model).to("cuda").eval()
            spatial_features_2d = stage1d_neck(spatial_features_2d_backbone)

            stage_regroup = StageRegroup(opt.max_cavs, record_len).to("cuda").eval()
            regroup_feature, mask = stage_regroup(spatial_features_2d, prior_encoding)

            stage3a_rte = Stage3aRTE(model).to("cuda").eval()
            x_after_rte, prior_encoding_stage3 = stage3a_rte(regroup_feature)

            stage3b_sttf = Stage3bSTTF(model).to("cuda").eval()
            x_after_sttf = stage3b_sttf(x_after_rte, spatial_correction_matrix)

            stage3c_hgt_debug = Stage3cHGTCavAttentionDebug(model).to("cuda").eval()
            x_after_hgt_debug = stage3c_hgt_debug(x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix)

            stage3d_pwindow_debug = Stage3dPyramidWindowAttentionDebug(model).to("cuda").eval()
            _ = stage3d_pwindow_debug(x_after_hgt_debug)

            stage3e_fusion_blocks = Stage3eFusionBlocks(model).to("cuda").eval()
            x_after_blocks = stage3e_fusion_blocks(x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix)

            stage3f_fusion_post = Stage3fFusionPost(model).to("cuda").eval()
            fused_feature = stage3f_fusion_post(x_after_blocks)

            stage5_heads = Stage5Heads(model).to("cuda").eval()
            _ = stage5_heads(fused_feature)

        enabled_precisions = {torch.float16} if opt.precision == "fp16" else {torch.float32}

        stages = [
            (
                "stage1a_pillar_vfe",
                stage1a_pillar_vfe,
                (voxel_features, voxel_coords, voxel_num_points),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (voxel_features, voxel_coords, voxel_num_points)),
            ),
            (
                "stage1b_scatter",
                stage1b_scatter,
                (voxel_coords, pillar_features),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (voxel_coords, pillar_features)),
            ),
            (
                "stage1c_backbone",
                stage1c_backbone,
                (spatial_features,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (spatial_features,)),
            ),
            (
                "stage1d_neck",
                stage1d_neck,
                (spatial_features_2d_backbone,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (spatial_features_2d_backbone,)),
            ),
            (
                "stage2_regroup",
                stage_regroup,
                (spatial_features_2d, prior_encoding),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (spatial_features_2d, prior_encoding)),
            ),
            (
                "stage3a_rte",
                stage3a_rte,
                (regroup_feature,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (regroup_feature,)),
            ),
            (
                "stage3b_sttf",
                stage3b_sttf,
                (x_after_rte, spatial_correction_matrix),
                _build_trt_inputs_from_example_tensors(
                    torch_tensorrt,
                    (x_after_rte, spatial_correction_matrix),
                ),
            ),
            (
                "stage3c_hgt_cav_attention_debug",
                stage3c_hgt_debug,
                (x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix),
                _build_trt_inputs_from_example_tensors(
                    torch_tensorrt,
                    (x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix),
                ),
            ),
            (
                "stage3d_pyramid_window_attention_debug",
                stage3d_pwindow_debug,
                (x_after_hgt_debug,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (x_after_hgt_debug,)),
            ),
            (
                "stage3e_fusion_blocks",
                stage3e_fusion_blocks,
                (x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix),
                _build_trt_inputs_from_example_tensors(
                    torch_tensorrt,
                    (x_after_sttf, prior_encoding_stage3, mask, spatial_correction_matrix),
                ),
            ),
            (
                "stage3f_fusion_post",
                stage3f_fusion_post,
                (x_after_blocks,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (x_after_blocks,)),
            ),
            (
                "stage5_heads",
                stage5_heads,
                (fused_feature,),
                _build_trt_inputs_from_example_tensors(torch_tensorrt, (fused_feature,)),
            ),
        ]

        summary: Dict[str, Dict[str, object]] = {}
        executed_stage_count = 0

        for stage_name, stage_module, example_inputs, trt_inputs in stages:
            if selected_stage != "all" and stage_name != selected_stage:
                continue
            if executed_stage_count > 0:
                print("\n" + "-" * 52 + "\n")
            if stage_name == "stage3d_pyramid_window_attention_debug":
                x_stage3d = example_inputs[0]
                print(
                    f"[{stage_name}] example input shape={tuple(x_stage3d.shape)} "
                    f"numel={x_stage3d.numel()} dtype={x_stage3d.dtype} device={x_stage3d.device}"
                )
            traced, has_cpu_values = _trace_module(stage_module, stage_name, example_inputs, opt.log_dir)
            ok, err = _try_convert_stage(
                torch_tensorrt,
                traced,
                stage_name,
                trt_inputs,
                enabled_precisions,
                opt.output_dir,
                require_full_compilation,
            )
            summary[stage_name] = {"trt_ok": ok, "cpu_in_graph": has_cpu_values, "error": err}
            executed_stage_count += 1
            if not ok:
                print(f"Stage failed but continuing for diagnostics: {stage_name}")

        print(f"\n{'='*15} STAGE CONVERSION SUMMARY {'='*15}")
        print(f"{'Stage':<40} | {'TRT conversion':<14} | {'CPU in graph':<12}")
        print("-" * 74)
        for stage_name, info in summary.items():
            status = "OK" if info["trt_ok"] else "FAILED"
            cpu_flag = "YES" if info["cpu_in_graph"] else "NO"
            print(f"{stage_name:<40} | {status:<14} | {cpu_flag:<12}")
        print("-" * 74)

        if not summary:
            print(f"No stage executed for TRT_STAGE={selected_stage}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
