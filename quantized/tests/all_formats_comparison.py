import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer

print("\n" + "=" * 86)
print("MSE COMPARISON: ALL QUANTIZATION FORMATS (512x512 random input)")
print("=" * 86)

x_large = torch.randn(512, 512, dtype=torch.float32)

quantizers = [
    ("FP16", 10, 5, AffineFakeQuantizer('fp16')),
    ("BP16", 7, 8, AffineFakeQuantizer('bp16')),
    ("FP8 E5M2", 2, 5, AffineFakeQuantizer('fp8_e5m2')),
    ("FP8 E4M3FN", 3, 4, AffineFakeQuantizer('fp8_e4m3fn')),
    ("FP4", 1, 2, AffineFakeQuantizer('fp4')),
]

print("\nFormat      | Mantissa Bits | Exponent Bits |      MSE")
print("-" * 86)
for name, mantissa_bits, exponent_bits, quantizer in quantizers:
    q = quantizer(x_large)
    mse = torch.nn.functional.mse_loss(q, x_large).item()
    print("{:<11} |{:^14}|{:^15}| {:.6e}".format(name, mantissa_bits, exponent_bits, mse))
print("-" * 86)
print("\n" + "=" * 86)
