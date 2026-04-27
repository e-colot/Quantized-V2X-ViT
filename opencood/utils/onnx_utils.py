import os

def log_onnx_structure(onnx_model, log_path):
    """Log all nodes, their names, types, inputs and outputs to a file."""
    graph = onnx_model.graph
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("ONNX Model Structure\n")
        f.write("=" * 60 + "\n\n")
        f.write("--- NODES AND ACTIVATIONS ---\n")
        for i, node in enumerate(graph.node):
            f.write(f"Node #{i:<4} | Op: {node.op_type:<15} | Name: {node.name}\n")
            f.write(f"  Inputs:  {', '.join(node.input)}\n")
            f.write(f"  Outputs: {', '.join(node.output)}\n")
            f.write("-" * 60 + "\n")
        f.write("\n--- INITIALIZERS (Weights/Biases) ---\n")
        for init in graph.initializer:
            f.write(f"Tensor Name: {init.name}\n")
    print(f"ONNX logs dumped in {log_path}")
