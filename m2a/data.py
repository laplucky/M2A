import os
import json
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

import torch


def read_json_samples(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
    """Read samples from JSON file and build complete conversations"""
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    full_prompts = []
    for sample in samples:
        if max_n is not None and len(full_prompts) >= max_n:
            break

        prompt = sample['prompt']
        reasoning = sample.get('reasoning', '')
        response = sample.get('response', '')

        # Build chat messages
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build complete conversation
        full_prompt = formatted_prompt + reasoning + "\n\n" + response
        full_prompts.append(full_prompt)

    return full_prompts

@dataclass
class PreparedSample:
    """Preprocessed sample"""
    input_ids: torch.Tensor
    nbr: List[int]
    pairs_q: List[Tuple[int, int]] = None
    pairs_k: List[Tuple[int, int]] = None
    # New: sampling based only on t
    v_t: List[int] = None
    o_t: List[int] = None
    ffn_t: List[int] = None

def locate_segments(text: str, tokenizer) -> List[int]:
    """Locate Assistant and thinking boundary token indices in text"""
    markers = ["<think>", "</think>", "<function=", "</function>", "<tool_call>", "</tool_call>"]
    bound_char = []
    for pat in markers:
        bound_char.extend([m.start() for m in re.finditer(re.escape(pat), text)])
    # Character to token mapping
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def char2tok(c):
        for i, (s, e) in enumerate(offsets):
            if s <= c < e:
                return i
        return None

    bound_idx = []
    for c in bound_char:
        t = char2tok(c)
        if t is not None:
            bound_idx.append(t)

    return sorted(list(set(bound_idx)))

def prepare_samples_unified(texts: List[str], tokenizer, radius: int, merge_types: str,
                   q_rows_per_text: int, k_rows_per_text: int,
                   v_rows_per_text: int, o_rows_per_text: int, ffn_rows_per_text: int,
                   rng: random.Random) -> List[PreparedSample]:
    """Unified sample preprocessing: locate boundaries, build neighborhoods and pairs based on merge types"""
    print("🔄 Preprocessing samples...")
    prepped = []

    for text in tqdm(texts, desc="Preprocessing"):
        bound_idx = locate_segments(text, tokenizer)

        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        T = enc["input_ids"].shape[1]
        ast_nbr = set()
        start_nbr = set()
        end_nbr = set()

        # Parse required types
        merge_q = 'q' in merge_types.lower()
        merge_k = 'k' in merge_types.lower()
        merge_v = 'v' in merge_types.lower()
        merge_o = 'o' in merge_types.lower()
        merge_f = 'f' in merge_types.lower()

        # Store boundary information for pair generation
        think_start, think_end = None, None
        function_start, function_end = None, None
        is_agent_data = False

        if len(bound_idx) > 3:
            is_agent_data = True
            function_start, function_end = bound_idx[-2], bound_idx[-1]
            think_start, think_end = bound_idx[-4], bound_idx[-3]
            for t in range(function_start, min(T, function_start+radius+1)):
                start_nbr.add(t)
            for t in range(max(0, function_end-radius), function_end+1):
                end_nbr.add(t)
            for t in range(think_start, min(T, think_start+radius+1)):
                start_nbr.add(t)
            for t in range(max(0, think_end-radius), think_end+1):
                end_nbr.add(t)

        elif len(bound_idx) == 2:
            think_start, think_end = bound_idx[-2], bound_idx[-1]
            for t in range(think_start, min(T, think_start+radius+1)):
                start_nbr.add(t)
            for t in range(max(0, think_end-radius), think_end+1):
                end_nbr.add(t)

        else:
            continue

        nbr = sorted(list(start_nbr) + list(end_nbr))

        if not nbr:
            continue

        # Generate pairs and sampling based on requirements
        sample_data = {
            "input_ids": enc["input_ids"],
            "start_nbr": start_nbr,
            "end_nbr": end_nbr,
            "nbr": nbr
        }

        if merge_q or merge_k:
            start_pairs = []
            end_pairs = []

            if is_agent_data:
                # think pairs
                start_pairs.extend([(think_start, i) for i in start_nbr])
                end_pairs.extend([(i, think_end) for i in end_nbr])
                # function pairs
                start_pairs.extend([(function_start, i) for i in start_nbr])
                end_pairs.extend([(i, function_end) for i in end_nbr])
            else:
                start_pairs = [(think_start, i) for i in start_nbr]
                end_pairs = [(i, think_end) for i in end_nbr]

            pairs = start_pairs + end_pairs
            rng.shuffle(pairs)
            if merge_q:
                sample_data["pairs_q"] = pairs[:q_rows_per_text]
            if merge_k:
                sample_data["pairs_k"] = pairs[:k_rows_per_text]

        if merge_v or merge_o or merge_f:
            ts = list(nbr)
            rng.shuffle(ts)
            if merge_v:
                sample_data["v_t"] = ts[:v_rows_per_text]
            if merge_o:
                sample_data["o_t"] = ts[:o_rows_per_text]
            if merge_f:
                sample_data["ffn_t"] = ts[:ffn_rows_per_text]

        # Only pass fields supported by PreparedSample
        valid_sample_data = {
            "input_ids": sample_data["input_ids"],
            "nbr": sample_data["nbr"]
        }
        # print("sample_data: ", sample_data)
        # print("vaild_sample_data: ", valid_sample_data)
        # Optional fields
        if "pairs_q" in sample_data:
            valid_sample_data["pairs_q"] = sample_data["pairs_q"]
        if "pairs_k" in sample_data:
            valid_sample_data["pairs_k"] = sample_data["pairs_k"]
        if "v_t" in sample_data:
            valid_sample_data["v_t"] = sample_data["v_t"]
        if "o_t" in sample_data:
            valid_sample_data["o_t"] = sample_data["o_t"]
        if "ffn_t" in sample_data:
            valid_sample_data["ffn_t"] = sample_data["ffn_t"]

        prepped.append(PreparedSample(**valid_sample_data))

    print(f"✅ Preprocessing completed, valid samples: {len(prepped)}")
    return prepped


def read_json_samples_recursive(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
    """Read samples from JSON file(s) recursively"""
    full_prompts = []

    def get_json_files(search_path):
        if os.path.isfile(search_path):
            yield search_path
        else:
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.json'):
                        yield os.path.join(root, file)

    for file_path in get_json_files(path):
        if max_n is not None and len(full_prompts) >= max_n:
            break

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Cannot read file {file_path}: {e}")
            continue

        samples = data if isinstance(data, list) else [data]

        for sample in samples:
            if max_n is not None and len(full_prompts) >= max_n:
                break

            if isinstance(sample, dict):
                prompt = sample.get('prompt', sample.get('text', str(sample)))
                reasoning = sample.get('reasoning', '')
                response = sample.get('response', '')
            elif isinstance(sample, str):
                prompt, reasoning, response = sample, '', ''
            else:
                prompt, reasoning, response = str(sample), '', ''

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_prompt = formatted_prompt + reasoning + "\n\n" + response
            full_prompts.append(full_prompt)

    return full_prompts
