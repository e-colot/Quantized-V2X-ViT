# -*- coding: utf-8 -*-

import os
import re
from typing import Dict, Tuple, List, Any

import torch
import yaml


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


def _compose_tensor_sources(
    source_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    rules: dict,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    composed = {}
    consumed_sources = []

    for item in rules.get('compose_tensors', []):
        target_pattern = item.get('target_pattern')
        source_templates = item.get('source_templates', [])
        op = item.get('op', 'stack')
        dim = int(item.get('dim', 0))

        if not target_pattern or not source_templates:
            continue

        target_regex = re.compile(target_pattern)

        for target_key in model_state.keys():
            match = target_regex.match(target_key)
            if not match:
                continue
            if target_key in source_state:
                continue

            resolved_sources = [match.expand(template) for template in source_templates]
            if not all(src in source_state for src in resolved_sources):
                continue

            tensors = [source_state[src] for src in resolved_sources]
            if op == 'stack':
                value = torch.stack(tensors, dim=dim)
            elif op == 'cat':
                value = torch.cat(tensors, dim=dim)
            else:
                continue

            composed[target_key] = value
            consumed_sources.extend(resolved_sources)

    return composed, consumed_sources


def _matches_any_regex(key: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if pattern is None:
            continue
        if re.search(pattern, key):
            return True
    return False


def remap_checkpoint_for_model(
    checkpoint_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    rules_path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    rules_path = os.path.join(rules_path, 'checkpoint_compat.yaml')
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

    composed, consumed_sources = _compose_tensor_sources(remapped, model_state, rules)
    remapped.update(composed)
    consumed_sources = set(consumed_sources)

    # Keep only keys that exist in model and have matching shape.
    compatible = {}
    shape_mismatch = []
    skipped_unknown = []

    for key, value in remapped.items():
        if key in consumed_sources:
            continue
        if key not in model_state:
            skipped_unknown.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(
                f'{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}'
            )
            continue
        compatible[key] = value

    ignore_missing_key_patterns = rules.get('ignore_missing_keys_regex', [])

    # Filter out new buffers that don't exist in old checkpoints using YAML-driven rules.
    new_buffers_to_exclude = {
        k for k in model_state.keys()
        if _matches_any_regex(k, ignore_missing_key_patterns)
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

    # Verbosity for debugging mismatches
    if load_ratio < 100.0:
        _print_samples("SHAPE MISMATCHES", shape_mismatch, max_items)
        _print_samples("MISSING MODEL KEYS (Not found in checkpoint)", missing_in_checkpoint, max_items)
        _print_samples("SKIPPED UNKNOWN (Checkpoint keys not in model)", skipped_unknown, max_items)

    print('-' * 52)
