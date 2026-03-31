import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer

quantizer_fp4 = AffineFakeQuantizer('fp4')

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

