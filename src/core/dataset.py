#!/usr/bin/env python3
"""
Unified dataset for Llama-MoE project
Supports both domain-specific training and MoE router training
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from config.domains import domain_manager
from config.moe import get_model_config, get_data_config


logger = logging.getLogger(__name__)


CHAT_BOS = "<|begin_of_text|>"
CHAT_USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
CHAT_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
CHAT_EOT = "<|eot_id|>"


def build_chat_prompt(user_prompt: str) -> str:
    """새 챗 템플릿 구조에 맞게 사용자 프롬프트를 변환."""
    body = user_prompt.rstrip()
    return (
        f"{CHAT_BOS}\n\n"
        f"{CHAT_USER_HEADER}\n\n"
        f"{body}\n\n"
        f"{CHAT_EOT}\n\n"
        f"{CHAT_ASSISTANT_HEADER}\n\n"
    )


def build_chat_response(response_text: str) -> str:
    """assistant 블록에 들어갈 응답 텍스트를 생성."""
    body = response_text.rstrip()
    if body.endswith(CHAT_EOT):
        return body
    return f"{body}\n\n{CHAT_EOT}"


@dataclass
class RouterSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    domain: str
    prompt: str
    answer: str
    instruction_length: int


def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    markers = ["Answer:", "답:", "정답:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for ch in text:
        upper = ch.upper()
        if upper in {"A", "B", "C", "D", "E"}:
            return upper
    return ""


def _encode_conversation(tokenizer, instruction_text: str, response_text: str, max_length: int):
    instruction_ids = tokenizer(
        instruction_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    instruction_len = instruction_ids["input_ids"].shape[1]
    remaining = max_length - instruction_len
    if remaining <= 0:
        input_ids = instruction_ids["input_ids"]
        attention_mask = instruction_ids["attention_mask"].to(torch.float32)
        labels = torch.full_like(input_ids, -100)
        return input_ids, attention_mask, labels, instruction_len

    response_ids = tokenizer(
        response_text,
        truncation=True,
        max_length=remaining,
        return_tensors="pt",
    )
    input_ids = torch.cat(
        [instruction_ids["input_ids"], response_ids["input_ids"]], dim=1
    )
    attention_mask = torch.cat(
        [instruction_ids["attention_mask"], response_ids["attention_mask"]], dim=1
    ).to(torch.float32)
    labels = input_ids.clone()
    labels[0, :instruction_len] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels, instruction_len


class RouterDataset(Dataset):
    """통합 데이터셋에서 라우터 학습에 필요한 샘플을 생성한다."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int,
        max_samples: Optional[int] = None,
        fallback_domain: Optional[str] = None,
    ) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raise ValueError("Expected list of samples")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fallback_domain = fallback_domain

        self.samples: List[RouterSample] = []
        for item in raw_data:
            if not self._is_valid_sample(item):
                continue
            self.samples.append(self._encode(item))
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError(f"No valid router samples found in {data_path}")

        logger.info("Loaded %d router samples from %s", len(self.samples), data_path)

    def _is_valid_sample(self, item: Dict[str, Any]) -> bool:
        required = {"question", "train_answer"}
        if not self.fallback_domain:
            required.add("domain")
        return required.issubset(item.keys())

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        domain = item.get("domain", "general")
        question = item.get("question", "")
        return domain_manager.format_prompt(domain, question=question)

    def _encode(self, item: Dict[str, Any]) -> RouterSample:
        prompt = self._format_prompt(item)
        response = item["train_answer"]

        instruction = build_chat_prompt(prompt)
        target = build_chat_response(response)

        input_ids, attention_mask, labels, instruction_len = _encode_conversation(
            self.tokenizer, instruction, target, self.max_length
        )

        answer_letter = _extract_answer_letter(response)
        domain_name = item.get("domain", self.fallback_domain)
        if domain_name is None:
            raise ValueError("Domain information missing and no fallback provided")

        return RouterSample(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            labels=labels.squeeze(0),
            domain=domain_name,
            prompt=instruction,
            answer=answer_letter,
            instruction_length=instruction_len,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "input_ids": sample.input_ids,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
            "domain": sample.domain,
            "prompt": sample.prompt,
            "answer": sample.answer,
            "instruction_length": sample.instruction_length,
        }


class DomainDataset(Dataset):
    """도메인별 LoRA 학습 시 사용하는 데이터셋."""

    def __init__(
        self,
        domain: str,
        tokenizer,
        max_length: int,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        self.domain = domain
        self.tokenizer = tokenizer
        self.max_length = max_length

        domain_cfg = domain_manager.get_domain(domain)
        data_path = domain_cfg.get_file_path(split)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found for domain={domain}, split={split}: {data_path}"
            )

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raise ValueError(f"Expected list of samples in {data_path}")

        self.samples: List[RouterSample] = []
        for item in raw_data:
            response = item.get("train_answer") or item.get("metric_answer") or ""
            prompt = domain_manager.format_prompt(domain, question=item.get("question", ""))
            instruction = build_chat_prompt(prompt)
            target = build_chat_response(response)

            encoded = _encode_conversation(self.tokenizer, instruction, target, self.max_length)
            if encoded is None:
                continue
            input_ids, attention_mask, labels, instruction_len = encoded
            answer_letter = _extract_answer_letter(response)
            self.samples.append(
                RouterSample(
                    input_ids=input_ids.squeeze(0),
                    attention_mask=attention_mask.squeeze(0),
                    labels=labels.squeeze(0),
                    domain=domain,
                    prompt=instruction,
                    answer=answer_letter,
                    instruction_length=instruction_len,
                )
            )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError(f"No valid samples found in {data_path}")

        logger.info(
            "Loaded %d %s samples for domain=%s",
            len(self.samples),
            split,
            domain,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "input_ids": sample.input_ids,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
            "domain": sample.domain,
            "prompt": sample.prompt,
            "answer": sample.answer,
            "instruction_length": sample.instruction_length,
        }


class RouterCollator:
    """배치 단위 패딩과 메타데이터 수집을 담당한다."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention, batch_first=True, padding_value=0
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            ),
            "domains": [f["domain"] for f in features],
            "prompt": [f["prompt"] for f in features],
            "answer": [f["answer"] for f in features],
            "instruction_length": [f["instruction_length"] for f in features],
        }
        return batch