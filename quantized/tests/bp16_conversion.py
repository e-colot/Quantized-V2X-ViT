import torch
from opencood.tools.quantization_utils import AffineFakeQuantizer
import struct

quantizer = AffineFakeQuantizer('bp16')

    # Special values (NaN, Inf, zero)
special_values = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=torch.float32)
quantized_special = quantizer(special_values)

assert torch.isnan(quantized_special[0]), "NaN not preserved"
assert torch.isinf(quantized_special[1]) and quantized_special[1] > 0, "+Inf not preserved"
assert torch.isinf(quantized_special[2]) and quantized_special[2] < 0, "-Inf not preserved"
assert quantized_special[3] == 0.0, "Zero not preserved"
print("Special values (NaN, Inf, zero) check ------------------- passed")

    # Overflow to infinity
def hex_to_f32(h):
    return struct.unpack('>f', struct.pack('>I', h))[0]
max_pos = hex_to_f32(0x7f7fffff)
max_neg = hex_to_f32(0xff7fffff)

# Create the tensor
large_values = torch.tensor([max_pos, max_neg], dtype=torch.float32)
quantized_large = quantizer(large_values)

assert torch.isinf(quantized_large[0]) and quantized_large[0] > 0, "Large positive should overflow to +inf"
assert torch.isinf(quantized_large[1]) and quantized_large[1] < 0, "Large negative should overflow to -inf"
print("Overflows check ----------------------------------------- passed")

    # Random tensor test
x_random = torch.randn(500, 500, dtype=torch.float32)
quantized_random = quantizer(x_random)

reference_random = x_random.to(torch.bfloat16).to(torch.float32)
match_mse = torch.nn.functional.mse_loss(quantized_random, reference_random).item()
print("Random input, comparison with built-in bp16 conversion ---------")
print("\n                MSE = {}".format(match_mse))

if match_mse <= 1e-10:
    print("\nBP16 quantizer ----------------------------------------- success")
