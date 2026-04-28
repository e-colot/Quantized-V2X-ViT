# builds a model to a tensorRT engine
import torch
import torch_tensorrt
import os
import onnx
import onnxsim
import tensorrt as trt

from opencood.utils import build_utils, onnx_utils, trt_utils


def _build_torchscript(model, inputs, hypes, ts_opt):
    print("Tracing model to TorchScript")
    traced_model = torch.jit.trace(model, inputs)
    traced_path = os.path.join(hypes['model_dir'], "logs/TS_graph.log")
    with open(traced_path, "w") as f:
        graph = traced_model.graph.copy()
        torch._C._jit_pass_inline(graph)
        f.write(str(graph))
        print(f"Saved TorchScript graph to {traced_path}")
    print(f"{'-'*63}")

    print("Compiling TorchScript traced model to TensorRT engine")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=ts_opt['trt_inputs'],
        enabled_precisions={torch.float32},
        truncate_long_and_double=False,
        require_full_compilation=True,
        workspace_size=1 << 33,
        allow_shape_tensors=True,
        ir='torchscript'
    )
    print(f"\n{'='*15} ENGINE SUCCESSFULLY BUILT {'='*15}")

    save_path = os.path.join(hypes['model_dir'], "trt_" + hypes['dataset'] + '.pt')
    torch.jit.save(trt_model, save_path)
    print(f'Engine stored in {save_path}')

def _build_onnx(model, inputs, hypes, onnx_opt):
    print('ONNX generation')
    
    onnx_path = os.path.join(hypes['model_dir'], "onnx/" + hypes['dataset'] + '.onnx')
    engine_path = os.path.join(hypes['model_dir'], "trt_" + hypes['dataset'] + '.engine')
    log_path = os.path.join(hypes['model_dir'], "logs/" + hypes['dataset'] + '_onnx.log')

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            input_names=onnx_opt['input_names'],
            output_names=onnx_opt['output_names'],
            dynamic_axes=onnx_opt['dynamic_axes'],
            opset_version=23, # FP4 supported for opset >= 23
            do_constant_folding=True,
            export_params=True,
        )

    print('ONNX validation')
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # print('ONNX simplification')
    # simplified, check_ok = onnxsim.simplify(onnx_model)
    # if check_ok:
    #     onnx.save(simplified, onnx_path)
    #     print("Simplification successful, saved simplified model")
    #     onnx_utils.log_onnx_structure(simplified, log_path)
    # else:
    #     print("[WARNING-ONNX] onnxsim check failed, using original ONNX")

    onnx.save(onnx_model, onnx_path)
    print("Saved ONNX model")
    onnx_utils.log_onnx_structure(onnx_model, log_path)
    print(f"{'-'*63}")
    
    print("Compiling ONNX model to TensorRT engine")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(trt_logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, trt_logger) as parser_trt, \
         builder.create_builder_config() as config:

        print(f"Parsing ONNX file: {onnx_path}")
        
        if hasattr(parser_trt, 'parse_from_file'):
            if not parser_trt.parse_from_file(onnx_path):
                for i in range(parser_trt.num_errors):
                    print(f"[ERROR] {parser_trt.get_error(i)}")
                raise RuntimeError("[ERROR-ONNX] parsing failed")
        else:
            # Fallback for older TRT versions
            with open(onnx_path, 'rb') as f:
                if not parser_trt.parse(f.read()):
                    for i in range(parser_trt.num_errors):
                        print(f"[ERROR] {parser_trt.get_error(i)}")
                    raise RuntimeError("[ERROR-ONNX] parsing failed")

        print("ONNX parsed successfully")

        profile = build_utils.build_onnx_profile(builder, onnx_opt['shapes'])
        config.add_optimization_profile(profile)

        # Build
        print("Compiling engine")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("[ERROR-ONNX] Engine build failed — check TRT_LOGGER output above")

        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
            print(f"\n{'='*15} ENGINE SUCCESSFULLY BUILT {'='*15}")
            print(f'Engine stored in {engine_path}')


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, hypes, opt = trt_utils.load_model()
    print(f"{'-'*63}")

    inputs, ts_opt, onnx_opt = build_utils.build_inputs(hypes)

    print(f"\n{'='*15} BUILDING TRT ENGINE {'='*15}")

    if opt.type == 'torchscript':
        _build_torchscript(model, inputs, hypes, ts_opt)
    elif opt.type == 'onnx':
        _build_onnx(model, inputs, hypes, onnx_opt)
    else:
        raise NotImplementedError(f"Cannot build with selected type: {opt.type}")
    
    print('-' * 52)

if __name__ == '__main__':
    main()
