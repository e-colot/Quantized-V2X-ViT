import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
import torch.nn as nn


def load_quantization_yaml(path, hypes):
    """
    Loads the quantization hypes located in 'path', then merge the resulting dictionnary with 'hypes'.
    """

    q_hypes = yaml_utils.load_yaml(path)

    model_args = hypes['model']['args']
    for k, v in q_hypes.items():
        if k in model_args.keys():
            # already existing key
            if model_args[k].keys() & v.keys():
                # check for no identical "sub-key"
                common_keys = model_args[k].keys() & v.keys()
                raise Exception("Quantization hype and model hype both have the following key(s): {}".format(", ".join(common_keys)))

            model_args[k] |= v

        else:
            model_args[k] = v


class AffineFakeQuantizerAutograd(torch.autograd.Function):
    # Custom autograd path to keep Straight-Through Estimator behavior.
    @staticmethod
    def _round_to_nearest_even(x):
        """Round non-negative values to nearest integer with ties to even (IEEE 754 standard)"""
            # to nearest
        floor_x = torch.floor(x)
        frac = x - floor_x
        rounded = torch.where(frac > 0.5, floor_x + 1.0, floor_x)

            # ties to even
        tie_mask = torch.isclose(frac, torch.full_like(frac, 0.5), atol=1e-7, rtol=0.0)
        floor_is_even = (torch.remainder(floor_x, 2.0) == 0.0)
        rounded = torch.where(tie_mask & (~floor_is_even), floor_x + 1.0, rounded)

        return rounded

    @staticmethod
    def forward(ctx, x, mantissa_bits, exponent_bits):
        # This assumes fp32 input and fake-quantizes to the target floating format.

            # new format parameters
        bias = (2 ** (exponent_bits - 1)) - 1
        max_exp_unbiased = bias
        min_normal = 2.0 ** (1 - bias)
        mantissa_levels = float(2 ** mantissa_bits)
            # subnormal removes the abrupt underflow between min_normal and 0
        min_subnormal = 2.0 ** (1 - bias - mantissa_bits)
        max_subnormal_code = float((2 ** mantissa_bits) - 1)

        out = torch.empty_like(x)

            # keeping specific non-number values as such
        nan_mask = torch.isnan(x)
        pos_inf_mask = torch.isposinf(x)
        neg_inf_mask = torch.isneginf(x)
        zero_mask = (x == 0)

        out[nan_mask] = torch.nan
        out[pos_inf_mask] = torch.inf
        out[neg_inf_mask] = -torch.inf
        out[zero_mask] = x[zero_mask]

        finite_nonzero_mask = torch.isfinite(x) & (~zero_mask)
        if torch.any(finite_nonzero_mask):
                # only work on data to quantize
            x_finite = x[finite_nonzero_mask]
            sign = torch.sign(x_finite)
            abs_x = torch.abs(x_finite)

            q_abs = torch.empty_like(abs_x)

            subnormal_mask = abs_x < min_normal
            normal_mask = ~subnormal_mask

            if torch.any(subnormal_mask):
                sub = abs_x[subnormal_mask]
                    # application of quantize/dequantize formula
                sub_code = AffineFakeQuantizerAutograd._round_to_nearest_even(sub / min_subnormal).clamp(0.0, max_subnormal_code)
                q_abs[subnormal_mask] = sub_code * min_subnormal

            if torch.any(normal_mask):
                norm = abs_x[normal_mask]

                # frexp: norm = m * 2**e, m in [0.5, 1)
                mantissa, exponent = torch.frexp(norm)
                exponent_unbiased = exponent.to(norm.dtype) - 1.0
                significand = mantissa * 2.0  # in [1, 2)

                mantissa_code = AffineFakeQuantizerAutograd._round_to_nearest_even((significand - 1.0) * mantissa_levels)

                # Carry when mantissa rounds up from 1.111... to 10.000...
                carry_mask = (mantissa_code == mantissa_levels)
                exponent_unbiased = exponent_unbiased + carry_mask.to(exponent_unbiased.dtype)
                mantissa_code = torch.where(carry_mask, torch.zeros_like(mantissa_code), mantissa_code)

                overflow_mask = exponent_unbiased > max_exp_unbiased
                q_norm = (1.0 + mantissa_code / mantissa_levels) * torch.pow(2.0, exponent_unbiased)
                q_norm = torch.where(overflow_mask, torch.full_like(q_norm, torch.inf), q_norm)
                q_abs[normal_mask] = q_norm

            out[finite_nonzero_mask] = q_abs * sign

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass the gradient through unchanged.
        return grad_output, None, None


class AffineFakeQuantizer(nn.Module):
    """
    Fake quantizer where output remains in fp32.
    params is a dict containing:
        'type' - currently supports 'fp16' and 'bp16'.
    """

    def __init__(self, params):
        super().__init__()
        self.mantissa_bits = 0
        self.exponent_bits = 0

        match params['type']:
            case 'fp16':
                self.mantissa_bits = 10
                self.exponent_bits = 5
            case _:
                raise ValueError("Unsupported quantization type: {}".format(params['type']))

    def forward(self, x):
        return AffineFakeQuantizerAutograd.apply(
            x,
            self.mantissa_bits,
            self.exponent_bits
        )
