import importlib
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.sub_modules.torch_transformation_utils import (
    _3x3_cramer_inverse,
    _affine_grid_sample_approx_bilinear_sample,
    _affine_grid_sample_approx_prepare_norm_grid,
    _gather_from_hw,
    affine_grid_sample_approx,
    combine_roi_and_cav_mask,
    convert_affinematrix_to_homography,
    get_discretized_transformation_matrix,
    get_roi_and_cav_mask,
    get_rotation_matrix2d,
    get_rotated_roi,
    get_transformation_matrix,
    normal_transform_pixel,
    normalize_homography,
    warp_affine,
)
from opencood.tools import train_utils


@dataclass
class Arguments:
    model_dir: str = "opencood/v2x-vit"
    output_dir: str = "tensorRT/transformation_utils_debug/engines"
    log_dir: str = "tensorRT/transformation_utils_debug/logs"
    precision: str = "fp32"
    max_cavs: int = 5


class FunctionGetDiscretizedTransformationMatrix(torch.nn.Module):
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


class FunctionGetTransformationMatrix(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("dsize", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, dist_correction_matrix):
        return get_transformation_matrix(dist_correction_matrix.reshape(-1, 2, 3), self.dsize)


class FunctionGetRotatedROI(torch.nn.Module):
    def __init__(self, b: int, l: int, c: int, h: int, w: int):
        super().__init__()
        self.register_buffer("input_template", torch.ones((int(b), int(l), int(c), int(h), int(w)), dtype=torch.float32))
        self.register_buffer("spatial_size", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, correction_matrix):
        return get_rotated_roi(self.input_template, self.spatial_size, correction_matrix)


class FunctionCombineRoiAndCavMask(torch.nn.Module):
    def forward(self, roi_mask, cav_mask):
        return combine_roi_and_cav_mask(roi_mask, cav_mask)


class FunctionGetRoiAndCavMask(torch.nn.Module):
    def __init__(self, h: int, w: int, c: int, discrete_ratio: float, downsample_rate: float):
        super().__init__()
        self.register_buffer("input_template", torch.ones((1, 1, int(h), int(w), int(c)), dtype=torch.float32))
        self.discrete_ratio = float(discrete_ratio)
        self.downsample_rate = float(downsample_rate)

    def forward(self, cav_mask, spatial_correction_matrix):
        b = cav_mask.shape[0]
        l = cav_mask.shape[1]
        input_tensor = self.input_template.expand(b, l, -1, -1, -1)
        return get_roi_and_cav_mask(
            input_tensor,
            cav_mask,
            spatial_correction_matrix,
            self.discrete_ratio,
            self.downsample_rate,
        )


class FunctionConvertAffineToHomography(torch.nn.Module):
    def forward(self, affine_2x3):
        return convert_affinematrix_to_homography(affine_2x3)


class FunctionNormalizeHomography(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("src_size", torch.tensor([int(h), int(w)], dtype=torch.int32))
        self.register_buffer("dst_size", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, homography_3x3):
        return normalize_homography(homography_3x3, self.src_size, self.dst_size)


class FunctionAffineGridSampleApprox(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("dsize", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, src, theta):
        return affine_grid_sample_approx(
            src,
            theta,
            self.dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class FunctionAffineGridPrepareNormGrid(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("dsize", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, theta):
        return _affine_grid_sample_approx_prepare_norm_grid(
            theta,
            self.dsize,
            align_corners=True,
        )


class FunctionAffineGridBilinearSample(torch.nn.Module):
    def forward(self, src, norm_grid):
        return _affine_grid_sample_approx_bilinear_sample(
            src,
            norm_grid,
            torch.float32,
            padding_mode="zeros",
            align_corners=True,
        )


class FunctionGatherFromHw(torch.nn.Module):
    def forward(self, src, x_idx, y_idx):
        return _gather_from_hw(src, x_idx, y_idx)


class FunctionWarpAffine(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("dsize", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, src, affine_2x3):
        return warp_affine(
            src,
            affine_2x3,
            self.dsize,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class FunctionNormalTransformPixelFromAffine(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = int(h)
        self.w = int(w)

    def forward(self, affine_2x3):
        h_tensor = torch.tensor(float(self.h), device=affine_2x3.device, dtype=affine_2x3.dtype)
        w_tensor = torch.tensor(float(self.w), device=affine_2x3.device, dtype=affine_2x3.dtype)
        return normal_transform_pixel(
            h_tensor,
            w_tensor,
            device=affine_2x3.device,
            dtype=affine_2x3.dtype,
        )


class Function3x3CramerInverse(torch.nn.Module):
    def forward(self, matrix_3x3):
        return _3x3_cramer_inverse(matrix_3x3)


class FunctionGetRotationMatrix2D(torch.nn.Module):
    def __init__(self, h: int, w: int):
        super().__init__()
        self.register_buffer("dsize", torch.tensor([int(h), int(w)], dtype=torch.int32))

    def forward(self, affine_2x3):
        return get_rotation_matrix2d(affine_2x3, self.dsize)


def _extract_model_inputs_from_batch(batch_data):
    ego = batch_data["ego"]
    processed = ego["processed_lidar"]
    return (
        processed["voxel_features"].to(torch.float32),
        processed["voxel_coords"].to(torch.int32),
        processed["voxel_num_points"].to(torch.int32),
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


def _write_graph_dump(log_dir: str, function_name: str, traced_module: torch.jit.ScriptModule):
    os.makedirs(log_dir, exist_ok=True)
    graph_path = os.path.join(log_dir, f"{function_name}.graphs.log")
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(f"[{function_name}] --- traced.graph ---\n")
        f.write(str(traced_module.graph))
        f.write("\n\n")
        f.write(f"[{function_name}] --- traced.inlined_graph ---\n")
        f.write(str(traced_module.inlined_graph))
        f.write("\n")
    print(f"[{function_name}] graph dump written to {graph_path}")


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


def _try_convert_function(
    torch_tensorrt,
    traced_module,
    function_name: str,
    trt_inputs: List,
    enabled_precisions: set,
    output_dir: str,
):
    print(f"[{function_name}] TRT conversion started")
    try:
        engine_bytes = torch_tensorrt.ts.convert_method_to_trt_engine(
            traced_module,
            "forward",
            inputs=trt_inputs,
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
        )
        engine_path = os.path.join(output_dir, f"{function_name}.engine")
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[{function_name}] TRT conversion succeeded -> {engine_path}")
        return True, ""
    except Exception as exc:
        print(f"[{function_name}] TRT conversion FAILED: {exc}")
        return False, str(exc)


def _build_trt_inputs_from_example_tensors(torch_tensorrt, example_tensors: Sequence[torch.Tensor]):
    return [
        torch_tensorrt.Input(shape=tuple(tensor.shape), dtype=tensor.dtype)
        for tensor in example_tensors
    ]


def main():
    opt = Arguments()
    assert opt.precision in ["fp16", "fp32"]

    selected_function = os.environ.get("TRT_FUNCTION", "all")
    valid_function_names = {
        "all",
        "cramer_inverse_3x3",
        "rotation_matrix2d",
        "discretized_transform",
        "transformation_matrix",
        "rotated_roi",
        "combine_roi_and_cav_mask",
        "get_roi_and_cav_mask",
        "convert_affine_to_homography",
        "normalize_homography",
        "gather_from_hw",
        "affine_grid_prepare_norm_grid",
        "affine_grid_bilinear_sample",
        "affine_grid_sample_approx",
        "warp_affine",
    }
    if selected_function not in valid_function_names:
        raise ValueError(f"Invalid TRT_FUNCTION={selected_function}. Use one of {sorted(valid_function_names)}")

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
        function_discretized_transform = FunctionGetDiscretizedTransformationMatrix(
            encoder.discrete_ratio,
            encoder.downsample_rate,
        ).to("cuda").eval()
        dist_matrix = function_discretized_transform(spatial_correction_matrix)
        affine_2x3 = dist_matrix.reshape(-1, 2, 3)

        function_normal_transform_pixel = FunctionNormalTransformPixelFromAffine(h, w).to("cuda").eval()
        normal_transform = function_normal_transform_pixel(affine_2x3)

        function_cramer_inverse_3x3 = Function3x3CramerInverse().to("cuda").eval()
        _ = function_cramer_inverse_3x3(normal_transform)

        function_rotation_matrix2d = FunctionGetRotationMatrix2D(h, w).to("cuda").eval()
        _ = function_rotation_matrix2d(affine_2x3)

        function_transformation_matrix = FunctionGetTransformationMatrix(h, w).to("cuda").eval()
        transform_matrix = function_transformation_matrix(dist_matrix)

        function_rotated_roi = FunctionGetRotatedROI(b, l, c, h, w).to("cuda").eval()
        roi_mask = function_rotated_roi(transform_matrix)

        function_combine_roi_and_cav_mask = FunctionCombineRoiAndCavMask().to("cuda").eval()
        combined_mask = function_combine_roi_and_cav_mask(roi_mask, cav_mask)

        function_get_roi_and_cav_mask = FunctionGetRoiAndCavMask(
            h,
            w,
            c,
            encoder.discrete_ratio,
            encoder.downsample_rate,
        ).to("cuda").eval()
        _ = function_get_roi_and_cav_mask(cav_mask, spatial_correction_matrix)

        function_convert_affine_to_homography = FunctionConvertAffineToHomography().to("cuda").eval()
        homography = function_convert_affine_to_homography(transform_matrix)

        function_normalize_homography = FunctionNormalizeHomography(h, w).to("cuda").eval()
        normalized_homography = function_normalize_homography(homography)

        src = torch.randn((b * l, 1, h, w), device="cuda", dtype=torch.float32)
        theta = normalized_homography[:, :2, :]

        x_idx = torch.randint(0, w, (b * l, h, w), device="cuda", dtype=torch.long)
        y_idx = torch.randint(0, h, (b * l, h, w), device="cuda", dtype=torch.long)

        function_gather_from_hw = FunctionGatherFromHw().to("cuda").eval()
        _ = function_gather_from_hw(src, x_idx, y_idx)

        function_affine_grid_prepare_norm_grid = FunctionAffineGridPrepareNormGrid(h, w).to("cuda").eval()
        norm_grid = function_affine_grid_prepare_norm_grid(theta)

        function_affine_grid_bilinear_sample = FunctionAffineGridBilinearSample().to("cuda").eval()
        _ = function_affine_grid_bilinear_sample(src, norm_grid)

        function_affine_grid_sample_approx = FunctionAffineGridSampleApprox(h, w).to("cuda").eval()
        _ = function_affine_grid_sample_approx(src, theta)

        function_warp_affine = FunctionWarpAffine(h, w).to("cuda").eval()
        _ = function_warp_affine(src, transform_matrix)

    enabled_precisions = {torch.float16} if opt.precision == "fp16" else {torch.float32}

    functions = [
        (
            "cramer_inverse_3x3",
            function_cramer_inverse_3x3,
            (normal_transform,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (normal_transform,)),
        ),
        (
            "rotation_matrix2d",
            function_rotation_matrix2d,
            (affine_2x3,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (affine_2x3,)),
        ),
        (
            "discretized_transform",
            function_discretized_transform,
            (spatial_correction_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (spatial_correction_matrix,)),
        ),
        (
            "transformation_matrix",
            function_transformation_matrix,
            (dist_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (dist_matrix,)),
        ),
        (
            "combine_roi_and_cav_mask",
            function_combine_roi_and_cav_mask,
            (roi_mask, cav_mask),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (roi_mask, cav_mask)),
        ),
        (
            "convert_affine_to_homography",
            function_convert_affine_to_homography,
            (transform_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (transform_matrix,)),
        ),
        (
            "normalize_homography",
            function_normalize_homography,
            (homography,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (homography,)),
        ),
        (
            "gather_from_hw",
            function_gather_from_hw,
            (src, x_idx, y_idx),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, x_idx, y_idx)),
        ),
        (
            "affine_grid_prepare_norm_grid",
            function_affine_grid_prepare_norm_grid,
            (theta,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (theta,)),
        ),
        (
            "affine_grid_bilinear_sample",
            function_affine_grid_bilinear_sample,
            (src, norm_grid),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, norm_grid)),
        ),
        (
            "affine_grid_sample_approx",
            function_affine_grid_sample_approx,
            (src, theta),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, theta)),
        ),
        (
            "warp_affine",
            function_warp_affine,
            (src, transform_matrix),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (src, transform_matrix)),
        ),
        (
            "rotated_roi",
            function_rotated_roi,
            (transform_matrix,),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (transform_matrix,)),
        ),
        (
            "get_roi_and_cav_mask",
            function_get_roi_and_cav_mask,
            (cav_mask, spatial_correction_matrix),
            _build_trt_inputs_from_example_tensors(torch_tensorrt, (cav_mask, spatial_correction_matrix)),
        ),
    ]

    summary: Dict[str, Dict[str, object]] = {}
    executed_function_count = 0

    for function_name, function_module, example_inputs, trt_inputs in functions:
        if selected_function != "all" and function_name != selected_function:
            continue
        if executed_function_count > 0:
            print("\n" + "-" * 52 + "\n")

        traced, has_cpu_values = _trace_module(function_module, function_name, example_inputs, opt.log_dir)
        ok, err = _try_convert_function(
            torch_tensorrt,
            traced,
            function_name,
            trt_inputs,
            enabled_precisions,
            opt.output_dir,
        )
        summary[function_name] = {"trt_ok": ok, "cpu_in_graph": has_cpu_values, "error": err}
        executed_function_count += 1
        if not ok:
            print(f"Function failed but continuing for diagnostics: {function_name}")

    print(f"\n{'='*15} TRANSFORMATION UTILS SUMMARY {'='*15}")
    print(f"{'Function':<36} | {'TRT conversion':<14} | {'CPU in graph':<12}")
    print("-" * 76)
    for function_name, info in summary.items():
        status = "OK" if info["trt_ok"] else "FAILED"
        cpu_flag = "YES" if info["cpu_in_graph"] else "NO"
        print(f"{function_name:<36} | {status:<14} | {cpu_flag:<12}")
    print("-" * 76)

    if not summary:
        print(f"No function executed for TRT_FUNCTION={selected_function}")


if __name__ == "__main__":
    main()
