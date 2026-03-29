import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer

quantizer = AffineFakeQuantizer({'type' : 'fp8'})

    # Special values (NaN, Inf, zero)
special_values = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=torch.float32)
quantized_special = quantizer(special_values)

assert torch.isnan(quantized_special[0]), "NaN not preserved"
assert torch.isinf(quantized_special[1]) and quantized_special[1] > 0, "+Inf not preserved"
assert torch.isinf(quantized_special[2]) and quantized_special[2] < 0, "-Inf not preserved"
assert quantized_special[3] == 0.0, "Zero not preserved"
print("Special values (NaN, Inf, zero) check ------------------- passed")

    # Overflow to infinity
# fp8 max normal is ~57344
large_values = torch.tensor([7e4, -7e4], dtype=torch.float32)
quantized_large = quantizer(large_values)

assert torch.isinf(quantized_large[0]) and quantized_large[0] > 0, "Large positive should overflow to +inf"
assert torch.isinf(quantized_large[1]) and quantized_large[1] < 0, "Large negative should overflow to -inf"
print("Overflows check ----------------------------------------- passed")

    # Subnormal values
# fp8 min normal is ~1.5e-5, smaller values become subnormal
subnormal_values = torch.tensor([8e-6, -8e-6], dtype=torch.float32)
quantized_subnormal = quantizer(subnormal_values)

# Subnormals should be quantized (not zero unless too small)
assert quantized_subnormal[0] >= 0.0, "Subnormal should be non-negative"
assert quantized_subnormal[1] <= 0.0, "Negative subnormal should be non-positive"
# At least one should be non-zero
assert quantized_subnormal[0] != 0.0 or quantized_subnormal[1] != 0.0, "Subnormals should not all be zero"
print("Subnormal values check ---------------------------------- passed")

    # Random tensor test
x_random = torch.randn(500, 500, dtype=torch.float32)
quantized_random = quantizer(x_random)

reference_random = x_random.to(torch.float8_e5m2).to(torch.float32)
match_mse = torch.nn.functional.mse_loss(quantized_random, reference_random).item()
print("Random input, comparison with built-in fp8 conversion ----------")
print("\n                MSE = {}".format(match_mse))

if match_mse <= 1e-10:
    print("\nFP8 quantizer ------------------------------------------ success")
