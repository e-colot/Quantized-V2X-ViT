# -*- coding: utf-8 -*-

import os
import re
from typing import Dict, Tuple, List, Any

import torch
import yaml


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_rules_path() -> str:
    return os.path.join(_project_root(), 'quantized', 'checkpoint_compat.yaml')


def _unwrap_checkpoint(raw_checkpoint):
    if isinstance(raw_checkpoint, dict):
        if 'state_dict' in raw_checkpoint and isinstance(raw_checkpoint['state_dict'], dict):
            return raw_checkpoint['state_dict']
        if 'model_state_dict' in raw_checkpoint and isinstance(raw_checkpoint['model_state_dict'], dict):
            return raw_checkpoint['model_state_dict']
    return raw_checkpoint


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            out[key[len('module.'):]] = value
        else:
            out[key] = value
    return out


def _load_rules(rules_path: str) -> dict:
    if not os.path.exists(rules_path):
        return {}

    with open(rules_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_rules(key: str, rules: dict) -> str:
    new_key = key

    for item in rules.get('rename_exact', []):
        src = item.get('from')
        dst = item.get('to')
        if src is None or dst is None:
            continue
        if new_key == src:
            new_key = dst

    for item in rules.get('rename_substrings', []):
        src = item.get('from')
        dst = item.get('to')
        if src is None or dst is None:
            continue
        new_key = new_key.replace(src, dst)

    for item in rules.get('rename_regex', []):
        pattern = item.get('pattern')
        repl = item.get('repl')
        if pattern is None or repl is None:
            continue
        new_key = re.sub(pattern, repl, new_key)

    return new_key


def remap_checkpoint_for_model(
    checkpoint_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    rules_path: str = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    rules_path = rules_path or default_rules_path()
    rules = _load_rules(rules_path)

    raw_state = _unwrap_checkpoint(checkpoint_state)
    state_dict = _strip_module_prefix(raw_state)

    remapped = {}
    renamed_pairs = []
    for old_key, value in state_dict.items():
        new_key = _apply_rules(old_key, rules)
        remapped[new_key] = value
        if new_key != old_key:
            renamed_pairs.append(f'{old_key} -> {new_key}')

    # Keep only keys that exist in model and have matching shape.
    compatible = {}
    shape_mismatch = []
    skipped_unknown = []

    for key, value in remapped.items():
        if key not in model_state:
            skipped_unknown.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(
                f'{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}'
            )
            continue
        compatible[key] = value

    # Filter out new buffers that don't exist in old checkpoints (e.g., TensorRT optim buffers)
    new_buffers_to_exclude = {
        k for k in model_state.keys() 
        if k.endswith('.num_windows_tensor')  # TensorRT PyramidWindow buffer
    }
    
    missing_in_checkpoint = [
        k for k in model_state.keys() 
        if k not in compatible 
        and k not in new_buffers_to_exclude
    ]
    
    # Adjust total model keys to exclude new non-loadable buffers for accurate coverage calculation
    total_model_keys_loadable = len(model_state) - len(new_buffers_to_exclude)

    report = {
        'rules_path': rules_path,
        'total_checkpoint_keys': len(state_dict),
        'total_model_keys': total_model_keys_loadable,
        'compatible_keys': len(compatible),
        'renamed_pairs': renamed_pairs,
        'shape_mismatch': shape_mismatch,
        'skipped_unknown': skipped_unknown,
        'missing_in_checkpoint': missing_in_checkpoint,
    }
    return compatible, report


def _print_samples(title: str, rows: List[str], max_items: int):
    if not rows:
        return
    print(f"\n{title} (showing up to {max_items})")
    for item in rows[:max_items]:
        print(f"- {item}")


def print_remap_report(report: Dict[str, Any], max_items: int = 10):
    total_checkpoint_keys = report['total_checkpoint_keys']
    total_model_keys = report['total_model_keys']
    compatible_keys = report['compatible_keys']
    load_ratio = 0.0 if total_model_keys == 0 else (compatible_keys / total_model_keys) * 100

    renamed_pairs = report['renamed_pairs']
    skipped_unknown = report['skipped_unknown']
    shape_mismatch = report['shape_mismatch']
    missing_in_checkpoint = report['missing_in_checkpoint']

    print(f"\n{'=' * 15} CHECKPOINT COMPAT {'=' * 15}")
    print(f"Rules file: {report['rules_path']}")
    print(f"{'Metric':<32} | {'Value':<16}")
    print('-' * 52)
    print(f"{'Checkpoint keys':<32} | {total_checkpoint_keys:<16}")
    print(f"{'Model keys':<32} | {total_model_keys:<16}")
    print(f"{'Loadable keys':<32} | {compatible_keys:<16}")
    print(f"{'Model load coverage':<32} | {load_ratio:>6.2f}%{'':<10}")
    print(f"{'Renamed keys':<32} | {len(renamed_pairs):<16}")
    print(f"{'Unknown keys skipped':<32} | {len(skipped_unknown):<16}")
    print(f"{'Shape mismatches':<32} | {len(shape_mismatch):<16}")
    print(f"{'Missing model keys':<32} | {len(missing_in_checkpoint):<16}")
    print('-' * 52)
    print(' ')
