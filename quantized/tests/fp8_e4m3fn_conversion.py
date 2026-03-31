import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer

quantizer = AffineFakeQuantizer({'type': 'fp8_e4m3fn'})

# Special values (NaN, Inf, zero)
special_values = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=torch.float32)
quantized_special = quantizer(special_values)

assert torch.isnan(quantized_special[0]), "NaN not preserved"
assert not torch.isinf(quantized_special[1]) and quantized_special[1] > 0, "+Inf should be clipped to max finite"
assert not torch.isinf(quantized_special[2]) and quantized_special[2] < 0, "-Inf should be clipped to min finite"
assert quantized_special[3] == 0.0, "Zero not preserved"
print("Special values (NaN, Inf, zero) check ------------------- passed")

# Overflow to max finite (no infinities in e4m3fn)
large_values = torch.tensor([1e10, -1e10], dtype=torch.float32)
quantized_large = quantizer(large_values)

assert torch.isfinite(quantized_large[0]) and quantized_large[0] > 0, "Large positive should saturate to max finite"
assert torch.isfinite(quantized_large[1]) and quantized_large[1] < 0, "Large negative should saturate to min finite"
print("Overflows check ----------------------------------------- passed")

# Subnormal values
subnormal_values = torch.tensor([3e-3, -3e-3], dtype=torch.float32)
quantized_subnormal = quantizer(subnormal_values)

assert quantized_subnormal[0] >= 0.0, "Subnormal should be non-negative"
assert quantized_subnormal[1] <= 0.0, "Negative subnormal should be non-positive"
assert quantized_subnormal[0] != 0.0 or quantized_subnormal[1] != 0.0, "Subnormals should not all be zero"
print("Subnormal values check ---------------------------------- passed")

# Random tensor test
x_random = torch.randn(500, 500, dtype=torch.float32)
quantized_random = quantizer(x_random)

reference_random = x_random.clamp(torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max).to(torch.float8_e4m3fn).to(torch.float32)
match_mse = torch.nn.functional.mse_loss(quantized_random, reference_random).item()
print("Random input, comparison with built-in fp8_e4m3fn conversion -----")
print("\n                MSE = {}".format(match_mse))

if match_mse <= 1e-10:
    print("\nFP8_E4M3FN quantizer ----------------------------------- success")
