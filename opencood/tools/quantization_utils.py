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
    def forward(ctx, x, mantissa_bits, exponent_bits, block_size, fn):
        # This assumes fp32 input and fake-quantizes to the target floating format.

        # new format parameters
        bias = (2 ** (exponent_bits - 1)) - 1
        max_exp_unbiased = bias
        if not fn:
            # no reserved symbols mean 1 more available exponent bit
            max_exp_unbiased += 1
        min_normal = 2.0 ** (1 - bias)
        mantissa_levels = float(2 ** mantissa_bits)
            # subnormal removes the abrupt underflow between min_normal and 0
        min_subnormal = 2.0 ** (1 - bias - mantissa_bits)
        max_subnormal_code = float((2 ** mantissa_bits) - 1)

        max_mantissa_val = (2**mantissa_bits - 1) / (2**mantissa_bits)
        max_representable_val = (1.0 + max_mantissa_val) * (2**max_exp_unbiased)

        saved_shape = x.shape;
        if block_size != 0:
            # Reshape already, to have the right empty_like output
            x = x.view(-1, block_size).clone()

        out = torch.empty_like(x)

            # handling special values before scaling to avoid propagating NaN's and Inf
            # keeping specific non-number values as such
        nan_mask = torch.isnan(x)
        pos_inf_mask = torch.isposinf(x)
        neg_inf_mask = torch.isneginf(x)
        zero_mask = (x == 0)

        if fn:
            out[pos_inf_mask] = torch.inf
            out[neg_inf_mask] = -torch.inf
        else:
            out[pos_inf_mask] = torch.finfo(torch.float32).max
            out[neg_inf_mask] = torch.finfo(torch.float32).min

        out[zero_mask] = 0
        out[nan_mask] = torch.nan

        if block_size != 0:
            # scale needs special values to be removed but they are only removed from out, not x
            # only if fn is false
            if not fn:
                x[nan_mask] = 0 # zero to not influence max() afterwards
                x[pos_inf_mask] = torch.finfo(torch.float32).max
                x[neg_inf_mask] = torch.finfo(torch.float32).min                
            max_vals = x.abs().max(dim=1, keepdim=True).values

            # raw bcs stored in fp32
            s_raw = max_vals / max_representable_val 
            # scale stored in fp8 E4M3 according to nvidia docs
            s_block = s_raw.clamp(torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max).to(torch.float8_e4m3fn).to(torch.float32)
            x = x / (s_block + 1e-8) # avoid division by zero

        # due to fn flag:
            # removes NaN's, because they are set to 0 for scaling, but shouldn't be quantized
            # also contains previous +/-inf as they should be quantized 
        finite_nonzero_mask = torch.isfinite(x) & (~zero_mask) & (~nan_mask)
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
                if fn:
                    q_norm = torch.where(overflow_mask, torch.full_like(q_norm, torch.inf), q_norm)
                else:
                    q_norm = torch.where(overflow_mask, torch.full_like(q_norm, max_representable_val), q_norm)
                q_abs[normal_mask] = q_norm

            out[finite_nonzero_mask] = q_abs * sign

        if block_size != 0:
            # rescale and reshape
            out = (out * s_block).view(saved_shape)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass the gradient through unchanged.
        return grad_output, None, None, None, None, None


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
        self.block_size = 0
        self.fn = True # does infinites belong to the representation

        match params['type'].lower():
            case 'fp16':
                self.mantissa_bits = 10
                self.exponent_bits = 5
                self.fn = True
            case 'bp16':
                # the conversion fp32 -> bp16 could be highly optimized (no exponent change)
                # might be usefull to implement at some point
                self.mantissa_bits = 7
                self.exponent_bits = 8
                self.fn = True
            case 'fp8':
                # fp8_e5m3 by default
                self.mantissa_bits = 2
                self.exponent_bits = 5
                self.fn = True
            case 'fp8_e5m3':
                self.mantissa_bits = 2
                self.exponent_bits = 5
                self.fn = True
            case 'fp8_e4m3fn':
                self.mantissa_bits = 3
                self.exponent_bits = 4
                self.fn = False
            case 'fp4':
                self.mantissa_bits = 1
                self.exponent_bits = 2
                self.block_size = 16
                self.fn = False
            case _:
                raise ValueError("Unsupported quantization type: {}".format(params['type']))
        
    def forward(self, x):
        return AffineFakeQuantizerAutograd.apply(
            x,
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
            self.fn
        )
