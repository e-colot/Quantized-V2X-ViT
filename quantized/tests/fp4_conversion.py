import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer

quantizer_fp4 = AffineFakeQuantizer({'type': 'fp4'})

    # Special values (NaN, Inf, zero)
test_special = torch.ones(16, dtype=torch.float32)
test_special[0] = torch.inf
test_special[1] = -torch.inf
test_special[2] = torch.nan
test_special[3] = 0
quantized_special = quantizer_fp4(test_special)

assert not torch.isinf(quantized_special[0]), "+Inf should be clipped"
assert not torch.isinf(quantized_special[1]), "-Inf should be clipped"
assert torch.isnan(quantized_special[2]), "NaN should be preserved"
assert quantized_special[3] == 0.0, "Zero not preserved"
print("Special values (NaN, Inf, zero) check ------------------- passed")

# Overflows
large_values = torch.ones(32, dtype=torch.float32)
large_values[0:8] = 1e6
large_values[8:16] = -1e6
large_values[16:24] = 1e20
large_values[24:32] = -1e20
quantized_large = quantizer_fp4(large_values)

assert torch.all(torch.isfinite(quantized_large)), "Large values should be clipped"
print("No overflows -------------------------------------------- passed")

print("\nFP4 quantizer ------------------------------------------- success")

# MSE Comparison across all formats
print("\n" + "=" * 70)
print("MSE COMPARISON: ALL QUANTIZATION FORMATS (512x512 random input)")
print("=" * 70)

x_large = torch.randn(512, 512, dtype=torch.float32)

quantizer_fp16 = AffineFakeQuantizer({'type': 'fp16'})
quantizer_bp16 = AffineFakeQuantizer({'type': 'bp16'})
quantizer_fp8 = AffineFakeQuantizer({'type': 'fp8'})
quantizer_fp4 = AffineFakeQuantizer({'type': 'fp4'})

q_fp16 = quantizer_fp16(x_large)
q_bp16 = quantizer_bp16(x_large)
q_fp8 = quantizer_fp8(x_large)
q_fp4 = quantizer_fp4(x_large)

mse_fp16 = torch.nn.functional.mse_loss(q_fp16, x_large).item()
mse_bp16 = torch.nn.functional.mse_loss(q_bp16, x_large).item()
mse_fp8 = torch.nn.functional.mse_loss(q_fp8, x_large).item()
mse_fp4 = torch.nn.functional.mse_loss(q_fp4, x_large).item()

print("\nFormat  | Mantissa Bits | Exponent Bits |      MSE")
print("-" * 70)
print("FP16    |      10       |       5       | {:.6e}".format(mse_fp16))
print("BP16    |       7       |       8       | {:.6e}".format(mse_bp16))
print("FP8     |       2       |       5       | {:.6e}".format(mse_fp8))
print("FP4     |       1       |       2       | {:.6e}".format(mse_fp4))
print("-" * 70)

print("\n" + "=" * 70)

