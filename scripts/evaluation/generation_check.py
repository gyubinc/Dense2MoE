#!/usr/bin/env python3
"""MoE 라우터 평가 스크립트."""

import argparse
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import torch

from config.moe import get_data_config, get_model_config, get_moe_config_manager
from src.utils import (
    setup_cuda_environment,
    setup_logging,
    setup_random_seed,
)

import logging


import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config.moe import get_gpu_config, get_model_config
from src.core.dataset import build_chat_prompt, build_chat_response
from src.models.model import MoEModel
from src.utils.utils import setup_random_seed




logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MoE router")
    parser.add_argument(
        "--moe-model-path",
        type=str,
        default=None,
        help="Path to trained router state (baseline if omitted)",
    )


    parser.add_argument(
        "--cuda-num",
        type=int,
        default=6,
        help="Override GPU index for CUDA_VISIBLE_DEVICES (e.g., 3)",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    config_manager = get_moe_config_manager()
    config_raw = config_manager.load_config()

    if args.cuda_num is not None:
        cuda_value = str(args.cuda_num)
        if isinstance(config_raw, dict):
            gpu_section = config_raw.setdefault("gpu", {})
            if not isinstance(gpu_section, dict):
                gpu_section = {}
                config_raw["gpu"] = gpu_section
            gpu_section["cuda_visible_devices"] = cuda_value
        logger.info("Overriding CUDA visible devices via --cuda-num: %s", cuda_value)


    setup_cuda_environment()
    setup_random_seed()

    # print_gpu_memory_summary("before evaluation")

    model_cfg = get_model_config()
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(get_gpu_config().device)
    # breakpoint()
    model = MoEModel()
    layer_expert_names = model.get_expert_names()
    if isinstance(layer_expert_names, list) and layer_expert_names and not isinstance(layer_expert_names[0], list):
        layer_expert_names = [layer_expert_names]






    # args.moe_model_path = "moe_models/moe_5e4_noaux/final_model/pytorch_model.bin"
    args.moe_model_path = None



    
    if args.moe_model_path is not None: 
        state = torch.load(args.moe_model_path, map_location="cpu")
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    device = device


    # LLaMA DecoderLayer 내부의 mlp인 MLPWithExperts의 router 파라미터 수 출력
    lora_param_count = 0
    for name, buf in model.named_buffers():
        if ".mlp.experts." not in name:
            continue
        if name.endswith(("gateA", "gateB", "upA", "upB", "downA", "downB")):
            lora_param_count += buf.numel()
    print(f"Total LoRA params: {lora_param_count}")

    
    
    
    # text = "Hello, My name is Gyubin. How are you?"

    

    # CLI clear
    os.system("clear")
    decoded = ""
    text = ""
    prompt = ""

    def _get_layer_names(layer_idx: int, count: int) -> list[str]:
        names: list[str] = []
        if isinstance(layer_expert_names, list) and 0 <= layer_idx < len(layer_expert_names):
            names = list(layer_expert_names[layer_idx])
        if len(names) < count:
            names += [f"expert_{i}" for i in range(len(names), count)]
        return names[:count]

    def _aggregate_counts(layer_idx: int, counts: list[float]) -> dict[str, float]:
        names = _get_layer_names(layer_idx, len(counts))
        aggregated: dict[str, float] = {}
        for idx, value in enumerate(counts):
            name = names[idx] if idx < len(names) else f"expert_{idx}"
            aggregated[name] = aggregated.get(name, 0.0) + float(value)
        return aggregated

    while text != "exit":
        print('\n')
        text = input("User: ")

        # prompt = prompt + decoded + build_chat_prompt(text)

        prompt = build_chat_prompt(text)

        # print(prompt)
        encoding = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        generated = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding.get("attention_mask"),
            max_new_tokens=300,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        router_accumulator = {}
        forward_out = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding.get("attention_mask"),
            labels=encoding["input_ids"],
            return_router_loss=True,
        )
        stats = getattr(forward_out, "router_stats", None)
        if stats:
            for item in stats:
                layer = item["layer"]
                counts = item["usage_counts"].to(torch.float64).cpu()
                if layer not in router_accumulator:
                    router_accumulator[layer] = counts.clone()
                else:
                    router_accumulator[layer] += counts
        model.clear_all_router_stats()

        router_average = None
        router_counts = {}
        agg_totals= {}
        for layer, tensor in router_accumulator.items():
            counts = tensor.tolist()
            names_to_counts = _aggregate_counts(layer, counts)
            router_counts[layer] = names_to_counts
            for name, value in names_to_counts.items():
                agg_totals[name] = agg_totals.get(name, 0.0) + value
        num_layers = len(router_counts)
        if num_layers:
            router_average = {
                name: total_value / num_layers for name, total_value in agg_totals.items()
            }
        # breakpoint()
        average_dict = {}
        if router_average:
            total = sum(router_average.values()) or 1.0
            for name, value in router_average.items():
                share = value/total
                average_dict[name] = round(float(share * 100), 2)
            # average_dict["router_average"] = router_average
            print(average_dict)


        # breakpoint()    
        input_len = encoding["input_ids"].shape[1]
        new_tokens = generated[0][input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("Assistant: ", decoded)
        decoded = build_chat_response(decoded)
        



    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
