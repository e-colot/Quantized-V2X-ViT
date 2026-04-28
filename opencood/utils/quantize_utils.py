import os

import torch
import torch.onnx
from torch.utils.data import DataLoader
from tqdm import tqdm

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.utils import trt_utils, build_utils


def _warn_uncalibrated(model) -> int:
    """Warn about TensorQuantizers that were never reached during calibration."""
    uncalibrated = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if getattr(module, '_amax', None) is None:
                uncalibrated.append(name)
    if uncalibrated:
        print(f"\n[WARN] {len(uncalibrated)} quantizer(s) never reached during calibration.")
        print("       They will use identity quantization. Expand calibration data or disable them.\n")
        for name in uncalibrated:
            print(f"       - {name}")
        print()
    return len(uncalibrated)


def _register_fp8_onnx_symbolics():
    """
    Register ONNX symbolic functions for ModelOpt's TRT-native FP8 ops so the
    TorchScript exporter can lower them to standard QuantizeLinear /
    DequantizeLinear nodes.

    ModelOpt emits two custom autograd functions:
      - ScaledE4M3Function  (the fake-quant kernel)
    which get recorded into the TorchScript graph as:
      - trt::TRT_FP8QuantizeLinear
      - trt::TRT_FP8DequantizeLinear

    We intercept at the autograd-function level by replacing the ONNX symbolic
    on ScaledE4M3Function so it emits a standard Q→DQ pair instead.
    """
    try:
        from modelopt.torch.quantization import tensor_quant as tq

        # Find the ScaledE4M3Function class (name may vary slightly by version)
        fp8_cls = None
        for attr in dir(tq):
            obj = getattr(tq, attr)
            try:
                if (isinstance(obj, type)
                        and issubclass(obj, torch.autograd.Function)
                        and 'E4M3' in attr):
                    fp8_cls = obj
                    print(f"[INFO] Found FP8 autograd class: {attr}")
                    break
            except TypeError:
                pass

        if fp8_cls is None:
            print("[WARN] Could not find ScaledE4M3Function in modelopt.torch.quantization.tensor_quant")
            return False

        # Register a symbolic that emits QuantizeLinear → DequantizeLinear.
        # The standard ONNX opset-13 QuantizeLinear signature:
        #   QuantizeLinear(x, scale, zero_point) → y_quantized
        #   DequantizeLinear(x, scale, zero_point) → y_dequantized
        # We chain them so the round-trip preserves the fake-quant semantics.
        @staticmethod  # type: ignore[misc]
        def symbolic(g, x, amax, *args, **kwargs):
            # Derive a per-tensor scale from amax.
            # FP8 E4M3 max representable value = 448.0
            fp8_max = g.op("Constant", value_t=torch.tensor(448.0, dtype=torch.float32))
            scale   = g.op("Div", amax, fp8_max)

            # zero_point = 0, dtype = FLOAT8E4M3FN (20 in ONNX opset 20+)
            # For opset 17 we use INT8 as a stand-in — TRT will reinterpret.
            zero_point = g.op(
                "Constant",
                value_t=torch.tensor(0, dtype=torch.int8),
            )
            quantized   = g.op("QuantizeLinear",   x,          scale, zero_point)
            dequantized = g.op("DequantizeLinear",  quantized,  scale, zero_point)
            return dequantized

        fp8_cls.symbolic = symbolic  # type: ignore[attr-defined]

        # Also register via the torch.onnx custom op registry for the
        # trt::TRT_FP8QuantizeLinear / TRT_FP8DequantizeLinear nodes that
        # appear after the JIT graph has been lowered.
        def _q_symbolic(g, x, scale, *args):
            zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
            return g.op("QuantizeLinear", x, scale, zp)

        def _dq_symbolic(g, x, scale, *args):
            zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
            return g.op("DequantizeLinear", x, scale, zp)

        torch.onnx.register_custom_op_symbolic("trt::TRT_FP8QuantizeLinear",  _q_symbolic,  1)
        torch.onnx.register_custom_op_symbolic("trt::TRT_FP8DequantizeLinear", _dq_symbolic, 1)

        print("[INFO] Registered ONNX symbolics for trt::TRT_FP8QuantizeLinear / TRT_FP8DequantizeLinear")
        return True

    except Exception as e:
        print(f"[WARN] _register_fp8_onnx_symbolics failed: {e}")
        return False


# ── Quantization config ────────────────────────────────────────────────────────
QUANT_CFG = mtq.FP8_DEFAULT_CFG


def full_quantization(calibrationDatasetSize: int = 100):
    model, hypes, opt = trt_utils.load_model()
    model.eval()
    print(f"{'-' * 52}\n")

    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_loop(model):
        print("Starting Calibration...")
        for i, batch_data in enumerate(tqdm(data_loader, total=calibrationDatasetSize)):
            if i >= calibrationDatasetSize:
                break
            batch_data = train_utils.to_device(batch_data, device)
            with torch.no_grad():
                inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset
                )

    # ── PTQ ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*15} INSTRUMENTING & CALIBRATING {'='*15}")
    quantized_model = mtq.quantize(model, QUANT_CFG, forward_loop)
    mtq.print_quant_summary(quantized_model)
    _warn_uncalibrated(quantized_model)
    print("Quantization and Calibration complete")

    # ── Register ONNX symbolics for TRT-native FP8 ops ────────────────────────
    print(f"\n{'='*15} PREPARING FOR ONNX EXPORT {'='*15}")
    _register_fp8_onnx_symbolics()

    # ── ONNX export ───────────────────────────────────────────────────────────
    inputs, _, onnx_opt = build_utils.build_inputs(hypes)

    onnx_path = os.path.join(opt.model_dir, "onnx/quantized_model.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print(f"Exporting to ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            quantized_model,
            inputs,
            onnx_path,
            export_params=True,
            dynamo=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=onnx_opt["input_names"],
            output_names=onnx_opt["output_names"],
            dynamic_axes=onnx_opt["dynamic_axes"],
        )

    print("Export successful.")
