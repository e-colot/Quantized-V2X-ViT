import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import re
import yaml


def load_quantization_yaml(path, hypes):
    """
    Loads the quantization hypes located in 'path', then merge the resulting dictionnary with 'hypes'.
    """

    q_hypes = yaml_utils.load_yaml(path)

    model_args = hypes['model']['args']
    for k, v in q_hypes.items():
        if k =='name':
            continue
        if k in model_args.keys():
            # already existing key
            if model_args[k].keys() & v.keys():
                # check for no identical "sub-key"
                common_keys = model_args[k].keys() & v.keys()
                raise Exception("Quantization hype and model hype both have the following key(s): {}".format(", ".join(common_keys)))

            model_args[k] |= v

        else:
            model_args[k] = v

    quantization_name = q_hypes.get('name', None)
    if quantization_name is not None:
        print("Quantization profile loaded: " + quantization_name)


def _qparser_cfg(quantize_cfg=None,
                 split_param_keys=None,
                 default_type='fp32',
                 cfg_name='quantized module'):
    """
    Generic parser for quantized module configs.

    Supported config formats:
      1) {'type': <dtype>} for shared quantization across all split params
      2) {<split_key_1>: <dtype>, ...} for split quantization per param
    """
    if split_param_keys is None or len(split_param_keys) == 0:
        raise ValueError('split_param_keys must be a non-empty list/tuple of config keys.')

    if quantize_cfg is None or len(quantize_cfg) == 0:
        quantize_cfg = {'type': default_type}

    if not isinstance(quantize_cfg, dict):
        raise ValueError("{} config must be a dictionary.".format(cfg_name))

    split_keys = tuple(split_param_keys)
    has_shared = 'type' in quantize_cfg
    has_split = len(set(split_keys) & set(quantize_cfg.keys())) > 0

    if has_shared and has_split:
        raise ValueError(
            "{} config is ambiguous: use either 'type' or all of {}.".format(
                cfg_name,
                ', '.join("'{}'".format(k) for k in split_keys)
            )
        )

    if has_shared:
        shared_cfg = {'type': quantize_cfg['type']}
        return tuple(shared_cfg for _ in split_keys)

    missing_keys = set(split_keys) - set(quantize_cfg.keys())
    if missing_keys:
        raise ValueError(
            "{} config must contain either 'type' or all of {}. Missing: {}".format(
                cfg_name,
                ', '.join("'{}'".format(k) for k in split_keys),
                ', '.join(sorted(missing_keys))
            )
        )

    return tuple({'type': quantize_cfg[k]} for k in split_keys)


def _find_last_checkpoint_epoch(saved_path):
    if os.path.exists(os.path.join(saved_path, 'latest.pth')):
        return 10000

    file_list = glob.glob(os.path.join(saved_path, '*epoch*.pth'))
    if not file_list:
        return 0

    epochs_exist = []
    for file_path in file_list:
        result = re.findall(r'.*epoch(.*)\.pth.*', file_path)
        if result:
            epochs_exist.append(int(result[0]))

    return max(epochs_exist) if epochs_exist else 0


def _normalize_loaded_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for candidate_key in ('state_dict', 'model_state_dict', 'model'):
            if candidate_key in checkpoint and isinstance(checkpoint[candidate_key], dict):
                checkpoint = checkpoint[candidate_key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError('Loaded checkpoint is not a valid state_dict dictionary.')

    normalized = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        normalized[key] = value
    return normalized


def _load_checkpoint_compat_rules(compat_yaml_path):
    if not compat_yaml_path:
        return []
    if not os.path.exists(compat_yaml_path):
        raise FileNotFoundError('Compatibility YAML not found: {}'.format(compat_yaml_path))

    with open(compat_yaml_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    rules = config.get('rules', [])
    if not isinstance(rules, list):
        raise ValueError("'rules' in compatibility YAML must be a list.")
    return rules


def _rule_matches_key(key, rule):
    when_prefix = rule.get('when_prefix')
    when_contains = rule.get('when_contains')
    old_suffix = rule.get('old_suffix')

    if when_prefix and not key.startswith(when_prefix):
        return False
    if when_contains and when_contains not in key:
        return False
    if old_suffix and not key.endswith(old_suffix):
        return False
    return True


def _apply_compat_rules_to_key(key, rules):
    remapped_key = key
    for rule in rules:
        if not _rule_matches_key(remapped_key, rule):
            continue

        old_suffix = rule.get('old_suffix')
        new_suffix = rule.get('new_suffix')
        if old_suffix is None or new_suffix is None:
            continue

        remapped_key = remapped_key[:-len(old_suffix)] + new_suffix
        break
    return remapped_key


def load_saved_model_with_compat(saved_path,
                                 model,
                                 compat_yaml_path=None,
                                 verbose=True):
    """
    Load latest saved model checkpoint with optional key remapping rules.

    The compatibility YAML can define remapping rules as:
      rules:
        - when_prefix: pfn_layers.
          old_suffix: .linear.weight
          new_suffix: .linear.linear.weight
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    initial_epoch = _find_last_checkpoint_epoch(saved_path)
    if initial_epoch <= 0:
        return initial_epoch, model

    model_file = os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch) \
        if initial_epoch != 10000 else os.path.join(saved_path, 'latest.pth')

    if verbose:
        print('resuming by loading epoch %d' % initial_epoch)
        print('checkpoint path: %s' % model_file)

    raw_checkpoint = torch.load(model_file, map_location='cpu')
    checkpoint_state = _normalize_loaded_checkpoint(raw_checkpoint)

    rules = _load_checkpoint_compat_rules(compat_yaml_path)
    remapped_state = {}
    remapped_count = 0
    for key, value in checkpoint_state.items():
        new_key = _apply_compat_rules_to_key(key, rules)
        if new_key != key:
            remapped_count += 1
        remapped_state[new_key] = value

    model_state = model.state_dict()
    filtered_state = {}
    unexpected_keys = []
    shape_mismatch_keys = []
    for key, value in remapped_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch_keys.append(key)
            continue
        filtered_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in filtered_state]

    model.load_state_dict(filtered_state, strict=False)

    if verbose:
        print('\n' + "="*50)
        print('compat remapped keys: {}'.format(remapped_count))
        print('loaded keys: {}'.format(len(filtered_state)))
        print('missing model keys: {}'.format(len(missing_keys)))
        print('unexpected checkpoint keys: {}'.format(len(unexpected_keys)))
        print('shape mismatch keys: {}'.format(len(shape_mismatch_keys)))

        # Keep output concise while still surfacing actionable key names.
        max_print = 5
        if missing_keys:
            print('missing model key examples: {}'.format(', '.join(missing_keys[:max_print])))
        if unexpected_keys:
            print('unexpected checkpoint key examples: {}'.format(', '.join(unexpected_keys[:max_print])))
        if shape_mismatch_keys:
            print('shape mismatch key examples: {}'.format(', '.join(shape_mismatch_keys[:max_print])))
        print("="*50 + "\n")

    del raw_checkpoint
    return initial_epoch, model


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
    
class Passtrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass the gradient through unchanged.
        return grad_output
    
class AffineFakeQuantizer(nn.Module):
    """
    Fake quantizer where output remains in fp32.
    Accepts a type string directly (e.g., 'fp8', 'fp16', 'bp16').
    """

    def __init__(self, dtype):
        super().__init__()
        self.mantissa_bits = 0
        self.exponent_bits = 0
        self.block_size = 0
        self.fn = True # does infinites belong to the representation
        self.passthrough = False

        match dtype.lower():
            case 'fp32':
                self.passthrough = True
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
                # fp8_e5m2 by default
                self.mantissa_bits = 2
                self.exponent_bits = 5
                self.fn = True
            case 'fp8_e5m2':
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
                raise ValueError("Unsupported quantization type: {}".format(dtype))
        
    def forward(self, x):
        if self.passthrough:
            return Passtrough.apply(x)
        else:
            return AffineFakeQuantizerAutograd.apply(
                x,
                self.mantissa_bits,
                self.exponent_bits,
                self.block_size,
                self.fn
            )

## ============== QUANTIZED MODULES ==================

class QuantizedLinear(nn.Module):
    """
    Fake-quantized linear layer that supports shared or split quantization config.
    """

    def __init__(self, in_features, out_features, bias=True, quantize_cfg=None, cfg_name='quantized linear'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        a_cfg, w_cfg, b_cfg = _qparser_cfg(
            quantize_cfg=quantize_cfg,
            split_param_keys=('type_a', 'type_w', 'type_b'),
            cfg_name=cfg_name,
        )
        self.quant_a = AffineFakeQuantizer(a_cfg['type'])
        self.quant_w = AffineFakeQuantizer(w_cfg['type'])
        self.quant_b = AffineFakeQuantizer(b_cfg['type'])

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        x_q = self.quant_a(x)
        w_q = self.quant_w(self.linear.weight)
        b_q = self.quant_b(self.linear.bias) if self.linear.bias is not None else None
        return F.linear(x_q, w_q, b_q)

class QuantizedConv2D(nn.Module):
    """
    Fake-quantized Conv2D layer with activation/kernel/bias quantization config.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 quantize_cfg=None,
                 cfg_name='quantized conv2d'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        a_cfg, k_cfg, b_cfg = _qparser_cfg(
            quantize_cfg=quantize_cfg,
            split_param_keys=('type_a', 'type_k', 'type_b'),
            cfg_name=cfg_name,
        )
        self.quant_a = AffineFakeQuantizer(a_cfg['type'])
        self.quant_k = AffineFakeQuantizer(k_cfg['type'])
        self.quant_b = AffineFakeQuantizer(b_cfg['type'])

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x_q = self.quant_a(x)
        k_q = self.quant_k(self.conv.weight)
        b_q = self.quant_b(self.conv.bias) if self.conv.bias is not None else None
        return self.conv._conv_forward(x_q, k_q, b_q)


class QuantizedConvTranspose2D(nn.Module):
    """
    Fake-quantized ConvTranspose2D layer with activation/kernel/bias quantization config.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 quantize_cfg=None,
                 cfg_name='quantized convtranspose2d'):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        a_cfg, k_cfg, b_cfg = _qparser_cfg(
            quantize_cfg=quantize_cfg,
            split_param_keys=('type_a', 'type_k', 'type_b'),
            cfg_name=cfg_name,
        )
        self.quant_a = AffineFakeQuantizer(a_cfg['type'])
        self.quant_k = AffineFakeQuantizer(k_cfg['type'])
        self.quant_b = AffineFakeQuantizer(b_cfg['type'])

    @property
    def weight(self):
        return self.conv_t.weight

    @property
    def bias(self):
        return self.conv_t.bias

    def forward(self, x):
        x_q = self.quant_a(x)
        k_q = self.quant_k(self.conv_t.weight)
        b_q = self.quant_b(self.conv_t.bias) if self.conv_t.bias is not None else None
        return F.conv_transpose2d(
            x_q,
            k_q,
            b_q,
            stride=self.conv_t.stride,
            padding=self.conv_t.padding,
            output_padding=self.conv_t.output_padding,
            groups=self.conv_t.groups,
            dilation=self.conv_t.dilation,
        )


class QuantizedEmbedding(nn.Module):
    """
    Fake-quantized Embedding layer with weight/output quantization config.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.0,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 device=None,
                 dtype=None,
                 quantize_cfg=None,
                 cfg_name='quantized embedding'):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

        a_cfg, w_cfg = _qparser_cfg(
            quantize_cfg=quantize_cfg,
            split_param_keys=('type_a', 'type_w'),
            cfg_name=cfg_name,
        )
        self.quant_a = AffineFakeQuantizer(a_cfg['type'])
        self.quant_w = AffineFakeQuantizer(w_cfg['type'])

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, x):
        w_q = self.quant_w(self.embedding.weight)
        out = F.embedding(
            x,
            w_q,
            padding_idx=self.embedding.padding_idx,
            max_norm=self.embedding.max_norm,
            norm_type=self.embedding.norm_type,
            scale_grad_by_freq=self.embedding.scale_grad_by_freq,
            sparse=self.embedding.sparse,
        )
        return self.quant_a(out)

