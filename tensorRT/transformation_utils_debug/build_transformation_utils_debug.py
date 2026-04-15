import importlib
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.sub_modules.torch_transformation_utils import (
    affine_grid_sample_approx,
    combine_roi_and_cav_mask,
    convert_affinematrix_to_homography,
    get_discretized_transformation_matrix,
    get_roi_and_cav_mask,
    get_rotated_roi,
    get_transformation_matrix,
    normalize_homography,
    warp_affine,
)
from opencood.tools import train_utils


@dataclass
class Arguments:
    model_dir: str = "opencood/v2x-vit"
    output_dir: str = "tensorRT/tranformation_utils_debug/engines"
    log_dir: str = "tensorRT/tranformation_utils_debug/logs"
    precision: str = "fp32"
    max_cavs: int = 5


class StageGetDiscretizedTransformationMatrix(torch.nn.Module):
    def __init__(self, discrete_ratio: float, downsample_rate: float):
        super().__init__()
        self.discrete_ratio = float(discrete_ratio)
        self.downsample_rate = float(downsample_rate)

    def forward(self, spatial_correction_matrix):
        return get_discretized_transformation_matrix(
            spatial_correction_matrix,
            self.discrete_ratio,
            self.downsample_rate,
        )


class StageGetTransformationMatrix(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.dsize = (int(h), int(w))

    def forward(self, dist_correction_matrix):
        return get_transformation_matrix(dist_correction_matrix.reshape(-1, 2, 3), self.dsize)


class StageGetRotatedROI(torch.nn.Module):
    def __init__(self, b: int, l: int, c: int, h: int, w: int):
        super().__init__()
        self.shape = (int(b), int(l), int(c), int(h), int(w))

    def forward(self, correction_matrix):
        return get_rotated_roi(self.shape, correction_matrix)


class StageCombineRoiAndCavMask(torch.nn.Module):
    def forward(self, roi_mask, cav_mask):
        return combine_roi_and_cav_mask(roi_mask, cav_mask)


class StageGetRoiAndCavMask(torch.nn.Module):
    def __init__(self, h: int, w: int, c: int, discrete_ratio: float, downsample_rate: float):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.c = int(c)
        self.discrete_ratio = float(discrete_ratio)
        self.downsample_rate = float(downsample_rate)

    def forward(self, cav_mask, spatial_correction_matrix):
        b = int(cav_mask.shape[0])
        l = int(cav_mask.shape[1])
        return get_roi_and_cav_mask(
            (b, l, self.h, self.w, self.c),
            cav_mask,
            spatial_correction_matrix,
            self.discrete_ratio,
            self.downsample_rate,
        )


class StageConvertAffineToHomography(torch.nn.Module):
    def forward(self, affine_2x3):
        return convert_affinematrix_to_homography(affine_2x3)


class StageNormalizeHomography(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.src_size = (int(h), int(w))
        self.dst_size = (int(h), int(w))

    def forward(self, homography_3x3):
        return normalize_homography(homography_3x3, self.src_size, self.dst_size)


class StageAffineGridSampleApprox(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.dsize = (int(h), int(w))

    def forward(self, src, theta):
        return affine_grid_sample_approx(
            src,
            theta,
            self.dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class StageWarpAffine(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.dsize = (int(h), int(w))

    def forward(self, src, affine_2x3):
        return warp_affine(
            src,
            affine_2x3,
            self.dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def _extract_model_inputs_from_batch(batch_data):
    ego = batch_data["ego"]
    processed = ego["processed_lidar"]
    return (
        processed["voxel_features"],
        processed["voxel_coords"],
        processed["voxel_num_points"],
        ego["record_len"].to(torch.int32),
        ego["prior_encoding"],
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
            }
        )

    return dataset, per_sequence_shapes


def _select_trace_dataset_index(per_sequence_shapes):
    voxel_counts = [s["voxel_features"][0] for s in per_sequence_shapes]
    target = sorted(voxel_counts)[len(voxel_counts) // 2]
    return min(per_sequence_shapes, key=lambda item: abs(item["voxel_features"][0] - target))["dataset_idx"]


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


def _clean_logs_dir(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    print("Cleaned previous logs")


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
):
    print(f"[{stage_name}] TRT conversion started")
    try:
        engine_bytes = torch_tensorrt.ts.convert_method_to_trt_engine(
            traced_module,
            "forward",
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
        )
        engine_path = os.path.join(output_dir, f"{stage_name}.engine")
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[{stage_name}] TRT conversion succeeded -> {engine_path}")
        return True, ""
    except Exception as exc:
        print(f"[{stage_name}] TRT conversion FAILED: {exc}")
        return False, str(exc)


def _build_trt_inputs_from_example_tensors(torch_tensorrt, example_tensors: Sequence[torch.Tensor]):
    return [
        torch_tensorrt.Input(shape=tuple(tensor.shape), dtype=tensor.dtype)
        for tensor in example_tensors
    ]


def main():
    opt = Arguments()
    assert opt.precision in ["fp16", "fp32"]

    selected_stage = os.environ.get("TRT_STAGE", "all")
    valid_stage_names = {
        "all",
        "stage_discretized_transform",
        "stage_transformation_matrix",
        "stage_rotated_roi",
        "stage_combine_roi_and_cav_mask",
        "stage_get_roi_and_cav_mask",
        "stage_convert_affine_to_homography",
        "stage_normalize_homography",
        "stage_affine_grid_sample_approx",
        "stage_warp_affine",
    }
    if selected_stage not in valid_stage_names:
        raise ValueError(f"Invalid TRT_STAGE={selected_stage}. Use one of {sorted(valid_stage_names)}")

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    _clean_logs_dir(opt.log_dir)

    torch_tensorrt = importlib.import_module("torch_tensorrt")

    print("Loading OpenCOOD config and model...")
    hypes = yaml_utils.load_yaml(None, opt)
    model = train_utils.create_model(hypes).to("cuda")
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    print("Collecting trace input from validation dataset...")
    dataset, per_sequence_shapes = _collect_validation_shapes(hypes)
    trace_idx = _select_trace_dataset_index(per_sequence_shapes)

    sample = dataset[trace_idx]
    batch_data = dataset.collate_batch_test([sample])
    batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
    _, _, _, record_len, _, spatial_correction_matrix = _extract_model_inputs_from_batch(batch_data)

    encoder = model.fusion_net.encoder
    b = int(spatial_correction_matrix.shape[0])
    l = int(spatial_correction_matrix.shape[1])
    h = 48
    w = 176
    c = 1

    cav_mask = (torch.arange(l, device="cuda").view(1, l) < record_len.unsqueeze(1)).to(torch.float32)
    x = torch.randn((b, l, h, w, 256), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        stage_discretized_transform = StageGetDiscretizedTransformationMatrix(
            encoder.discrete_ratio,
            encoder.downsample_rate,
        ).to("cuda").eval()
        dist_matrix = stage_discretized_transform(spatial_correction_matrix)

        stage_transformation_matrix = StageGetTransformationMatrix(h, w).to("cuda").eval()
        transform_matrix = stage_transformation_matrix(dist_matrix)

        stage_rotated_roi = StageGetRotatedROI(b, l, c, h, w).to("cuda").eval()
        roi_mask = stage_rotated_roi(transform_matrix)

        stage_combine_roi_and_cav_mask = StageCombineRoiAndCavMask().to("cuda").eval()
        combined_mask = stage_combine_roi_and_cav_mask(roi_mask, cav_mask)

        stage_get_roi_and_cav_mask = StageGetRoiAndCavMask(
            h,
            w,
            c,
            encoder.discrete_ratio,
            encoder.downsample_rate,
        ).to("cuda").eval()
        _ = stage_get_roi_and_cav_mask(cav_mask, spatial_correction_matrix)

        stage_convert_affine_to_homography = StageConvertAffineToHomography().to("cuda").eval()
        homography = stage_convert_affine_to_homography(transform_matrix)

        stage_normalize_homography = StageNormalizeHomography(h, w).to("cuda").eval()
        normalized_homography = stage_normalize_homography(homography)

        src = torch.randn((b * l, 1, h, w), device="cuda", dtype=torch.float32)
        theta = normalized_homography[:, :2, :]

        stage_affine_grid_sample_approx = StageAffineGridSampleApprox(h, w).to("cuda").eval()
        _ = stage_affine_grid_sample_approx(src, theta)

        stage_warp_affine = StageWarpAffine(h, w).to("cuda").eval()
        _ = stage_warp_affine(src, transform_matrix)

    enabled_precisions = {torch.float16} if opt.precision == "fp16" else {torch.float32}

    stages = [
        (
            "stage_discretized_transform",
            stage_discretized_transform,
            (spatial_correction_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (spatial_correction_matrix,)),
        ),
        (
            "stage_transformation_matrix",
            stage_transformation_matrix,
            (dist_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (dist_matrix,)),
        ),
        (
            "stage_rotated_roi",
            stage_rotated_roi,
            (transform_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (transform_matrix,)),
        ),
        (
            "stage_combine_roi_and_cav_mask",
            stage_combine_roi_and_cav_mask,
            (roi_mask, cav_mask),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (roi_mask, cav_mask)),
        ),
        (
            "stage_get_roi_and_cav_mask",
            stage_get_roi_and_cav_mask,
            (cav_mask, spatial_correction_matrix),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (cav_mask, spatial_correction_matrix)),
        ),
        (
            "stage_convert_affine_to_homography",
            stage_convert_affine_to_homography,
            (transform_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (transform_matrix,)),
        ),
        (
            "stage_normalize_homography",
            stage_normalize_homography,
            (homography,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (homography,)),
        ),
        (
            "stage_affine_grid_sample_approx",
            stage_affine_grid_sample_approx,
            (src, theta),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, theta)),
        ),
        (
            "stage_warp_affine",
            stage_warp_affine,
            (src, transform_matrix),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, transform_matrix)),
        ),
    ]

    summary: Dict[str, Dict[str, object]] = {}
    executed_stage_count = 0

    for stage_name, stage_module, example_inputs, trt_inputs in stages:
        if selected_stage != "all" and stage_name != selected_stage:
            continue
        if executed_stage_count > 0:
            print("\n" + "-" * 52 + "\n")

        traced, has_cpu_values = _trace_module(stage_module, stage_name, example_inputs, opt.log_dir)
        ok, err = _try_convert_stage(
            torch_tensorrt,
            traced,
            stage_name,
            trt_inputs,
            enabled_precisions,
            opt.output_dir,
        )
        summary[stage_name] = {"trt_ok": ok, "cpu_in_graph": has_cpu_values, "error": err}
        executed_stage_count += 1
        if not ok:
            print(f"Stage failed but continuing for diagnostics: {stage_name}")

    print(f"\n{'='*15} TRANSFORMATION UTILS SUMMARY {'='*15}")
    print(f"{'Stage':<36} | {'TRT conversion':<14} | {'CPU in graph':<12}")
    print("-" * 76)
    for stage_name, info in summary.items():
        status = "OK" if info["trt_ok"] else "FAILED"
        cpu_flag = "YES" if info["cpu_in_graph"] else "NO"
        print(f"{stage_name:<36} | {status:<14} | {cpu_flag:<12}")
    print("-" * 76)

    if not summary:
        print(f"No stage executed for TRT_STAGE={selected_stage}")


if __name__ == "__main__":
    main()
