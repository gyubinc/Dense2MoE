#!/usr/bin/env python3
"""Qwen 기본 모델 혹은 도메인 LoRA를 단일 도메인 데이터셋에서 평가한다."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트를 sys.path에 추가하여 src, config 모듈을 찾을 수 있게 함
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 이제 프로젝트 모듈을 임포트
from src.utils.wandb_utils import (
    finish_wandb,
    generate_run_name,
    init_wandb,
    parse_command_args_for_wandb,
    log_something,
)


from config.domains import domain_manager
from config.moe import get_gpu_config, get_model_config
from src.core.dataset import build_chat_prompt
from src.utils import print_gpu_memory_summary, setup_logging, setup_random_seed


logger = logging.getLogger(__name__)


def _extract_answer_letter(text: str) -> str:
    if not text:
        return ""
    work = text.strip()
    for marker in ["Answer:", "답:", "정답:"]:
        if marker in work:
            work = work.split(marker, 1)[1]
            break
    for ch in work:
        upper = ch.upper()
        if upper in {"A", "B", "C", "D", "E"}:
            return upper
    return ""


@dataclass
class DomainSample:
    prompt: str
    answer: str


class DomainEvalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
    ) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: List[DomainSample] = []
        for item in raw:
            prompt = item.get("question", "")
            prompt = domain_manager.format_prompt(item.get("domain", "general"), question=prompt)
            response = item.get("train_answer", "")
            answer_letter = _extract_answer_letter(response)
            if not answer_letter:
                continue
            text = build_chat_prompt(prompt)
            samples.append(DomainSample(text, answer_letter))
            if max_samples is not None and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(f"No valid samples found in {data_path}")

        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DomainSample:
        return self.samples[idx]

    def __iter__(self):
        for sample in self.samples:
            yield sample


def _load_model(
    adapter_path: Optional[str],
    device_override: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    model_cfg = get_model_config()
    gpu_cfg = get_gpu_config()
    device = torch.device(device_override or gpu_cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if adapter_path:
        from peft import PeftModel

        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    # device_map="auto"가 사용된 경우 model.to(device)를 호출하면 분산 구조가 깨지므로 호출하지 않음
    if getattr(model, "is_fully_shifted", False) or \
       (hasattr(model, "hf_device_map") and model.hf_device_map):
        # 분산 로드된 경우 첫 번째 파라미터의 장치를 사용
        device = next(model.parameters()).device
    else:
        model.to(device)

    model.eval()
    return model, tokenizer, device


def _auto_adapter_path(domain: str) -> str:
    return os.path.join("domain_models", domain, "final_adapter")


from tqdm import tqdm

def extract_option_text(options_str: str, letter: str) -> str:
    # 알파벳 -> 정답으로 변환
    for line in options_str.splitlines():
        line = line.strip()
        if line.startswith(letter + "."):
            return line.split(".", 1)[1].strip()
    return None

def evaluate(model, tokenizer, device, dataset: DomainEvalDataset) -> tuple[int, int]:
    correct = 0
    correct_text = 0
    total = 0

    with torch.inference_mode():
        for sample in tqdm(dataset, desc="Evaluating", total=len(dataset)):
            sample.prompt = sample.prompt + "Answer:"
            encoding = tokenizer(sample.prompt, return_tensors="pt").to(device)
            answer_text = extract_option_text(sample.prompt, sample.answer.strip()[0])
            generated = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding.get("attention_mask"),
                max_new_tokens=20,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0][encoding["input_ids"].shape[1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = _extract_answer_letter(decoded)
            if pred == sample.answer:
                correct += 1
            elif answer_text in decoded:
                correct_text += 1
            total += 1

    return correct, total, correct_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA-augmented Qwen on domain datasets")
    parser.add_argument(
        "--domain",
        required=True,
        choices=domain_manager.get_available_domains(),
        help="Domain dataset to evaluate (train/test handled automatically)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples evaluated",
    )
    parser.add_argument(
        "--adapter-path",
        required=False,
        help="Explicit path to PEFT adapter directory",
    )
    parser.add_argument(
        "--adapter-domain",
        choices=domain_manager.get_available_domains(),
        help="Convenience flag to load adapter from domain_models/<domain>/final_adapter",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device identifier (e.g., cuda:0). Defaults to config value.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Wandb run name",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging(level="INFO")
    args = parse_args()

    setup_random_seed()

    adapter_path = args.adapter_path
    if args.adapter_domain and adapter_path:
        raise ValueError("Specify either --adapter-domain or --adapter-path, not both")
    if args.adapter_domain:
        adapter_path = _auto_adapter_path(args.adapter_domain)


    wandb_run = None
    if args.use_wandb:
        try:
            if args.wandb_name:
                run_name = args.wandb_name
            else:
                run_name = generate_run_name(
                    "evaluation",
                    model_type="domain",
                    domain=args.domain,
                    adapter_path=adapter_path,
            )
            wandb_config = parse_command_args_for_wandb(args)
            wandb_run = init_wandb(
                project_name="Llama_MoE",
                entity="gyubin5009",
                run_name=run_name,
                config=wandb_config,
                tags=["evaluation", "domain", args.domain],
            )
        except Exception as exc:
            logger.warning("Failed to initialize wandb: %s", exc)
            wandb_run = None


    logger.info("Evaluating domain=%s | split=%s | adapter=%s", args.domain, args.split, adapter_path or "<base>")
    if adapter_path:
        logger.info("Loading adapter from %s", adapter_path)

    print_gpu_memory_summary("before_model_load")
    model, tokenizer, device = _load_model(adapter_path, device_override=args.device)
    print_gpu_memory_summary("after_model_load")

    domain_cfg = domain_manager.get_domain(args.domain)
    data_path = domain_cfg.get_file_path(args.split)
    dataset = DomainEvalDataset(data_path, tokenizer, max_samples=args.max_samples)

    correct, total, correct_text = evaluate(model, tokenizer, device, dataset)
    accuracy = correct / total if total else 0.0
    accuracy_text = correct_text / total if total else 0.0

    average_dict = {}

    average_dict["domain"] = args.domain
    average_dict["accuracy"] = round(float(accuracy * 100), 2)
    average_dict["accuracy_text"] = round(float(accuracy_text * 100), 2)
    average_dict['total_accuracy'] = round(float((accuracy + accuracy_text) * 100), 2)
    average_dict["total_samples"] = total
    average_dict["top_k"] = 0  # wandb에 top_k 0으로 올림

    if wandb_run:
        log_something(average_dict=average_dict)
        finish_wandb()

    logger.info("Accuracy: %.4f (%d/%d)", accuracy, correct, total)
    logger.info("Accuracy (text): %.4f (%d/%d)", correct_text / total if total else 0.0, correct_text, total)
    print_gpu_memory_summary("after_evaluation")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    raise SystemExit(main())


