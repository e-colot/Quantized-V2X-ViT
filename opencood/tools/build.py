# builds a model to a tensorRT engine
import torch
import torch_tensorrt
import os

from opencood.utils import build_utils


def _build_torchscript(model, inputs, opt, ts, hypes):
    print("Tracing model to TorchScript")
    traced_model = torch.jit.trace(model, inputs)
    traced_path = os.path.join(opt.model_dir, "TS_graph.log")
    with open(traced_path, "w") as f:
        graph = traced_model.graph.copy()
        torch._C._jit_pass_inline(graph)
        f.write(str(graph))
        print(f"Saved TorchScript graph to {traced_path}")
    print(f"{'-'*63}")

    print("Compiling TorchScript traced model to TensorRT engine")
    trt_model = torch_tensorrt.compile(
        traced_model,
        inputs=ts['trt_inputs'],
        enabled_precisions={torch.float32},
        truncate_long_and_double=False,
        require_full_compilation=True,
        workspace_size=1 << 33,
        allow_shape_tensors=True,
        ir='torchscript'
    )
    print(f"\n{'='*15} ENGINE SUCCESSFULLY BUILT {'='*15}")

    dataset_type = hypes['validate_dir'].split('/')[-1]
    save_path = os.path.join(opt.model_dir, "trt_" + dataset_type + '.pt')
    torch.jit.save(trt_model, save_path)
    print(f'Engine stored in {save_path}')


def main(parser_opt=None):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, hypes, opt, parser_opt = build_utils.load_model(parser_opt)
    inputs, ts, onnx = build_utils.build_inputs(hypes)

    print(f"\n{'='*15} BUILDING TRT ENGINE {'='*15}")

    if parser_opt.type == 'torchscript':
        _build_torchscript(model, inputs, opt, ts, hypes)
    
    print('-' * 52)

if __name__ == '__main__':
    main()
