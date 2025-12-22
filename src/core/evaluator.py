#!/usr/bin/env python3
"""
Qwen-MoE Evaluator - Router-focused evaluation system
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config.moe import get_gpu_config, get_model_config
from .dataset import RouterCollator, RouterDataset
from ..models.model import MoEModel
from ..utils.utils import setup_random_seed


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    accuracy: float
    correct: int
    accuracy_text: float
    total: int
    router_counts: Optional[Dict[int, Dict[str, float]]] = None
    router_average: Optional[Dict[str, float]] = None


class RouterEvaluator:
    """생성 기반 평가로 MoE 라우터 성능을 측정한다."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        model_cfg = get_model_config()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        device = torch.device(get_gpu_config().device)
        self.model = MoEModel()
        if model_path is not None:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.device = device
        raw_names = self.model.get_expert_names()
        self.expert_names_by_layer: List[List[str]] = []
        if isinstance(raw_names, list):
            if raw_names and isinstance(raw_names[0], list):
                self.expert_names_by_layer = raw_names  # type: ignore[assignment]
            else:
                self.expert_names_by_layer = [raw_names]  # type: ignore[list-item]

    def _get_layer_expert_names(self, layer_idx: int, count: int) -> List[str]:
        names: List[str] = []
        if 0 <= layer_idx < len(self.expert_names_by_layer):
            names = list(self.expert_names_by_layer[layer_idx])
        if len(names) < count:
            names += [f"expert_{i}" for i in range(len(names), count)]
        return names[:count]

    def _aggregate_counts(self, layer_idx: int, counts: List[float]) -> Dict[str, float]:
        names = self._get_layer_expert_names(layer_idx, len(counts))
        aggregated: Dict[str, float] = {}
        for idx, value in enumerate(counts):
            name = names[idx] if idx < len(names) else f"expert_{idx}"
            aggregated[name] = aggregated.get(name, 0.0) + float(value)
        return aggregated

    def _build_loader(self, dataset: RouterDataset) -> DataLoader:
        collator = RouterCollator(self.tokenizer.pad_token_id)
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)

    def _extract_answer_letter(self, text: str) -> str:
        if not text:
            return ""
        work = text.strip()
        work_upper = work.upper()

        for marker in ["ANSWER:", "ANSWER :", "ANSWER"]:
            if marker in work_upper:
                idx = work_upper.find(marker)
                work = work[idx + len(marker):]
                break

        work = work.strip()
        for ch in work:
            if ch.upper() in {"A", "B", "C", "D", "E"}:
                return ch.upper()
        return ""

    def extract_option_text(self, options_str: str, letter: str) -> str:
        # 알파벳 -> 정답으로 변환
        for line in options_str.splitlines():
            line = line.strip()
            if line.startswith(letter + "."):
                return line.split(".", 1)[1].strip()
        return None

    def evaluate(self, dataset: RouterDataset) -> EvaluationResult:
        loader = self._build_loader(dataset)
        correct = 0
        correct_text = 0
        total = 0
        router_accumulator: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", unit="sample"):
                prompt = batch["prompt"][0]
                answer = batch["answer"][0]

                # 여기서 결국은 넣어줌
                prompt += "Answer:"

                answer_text = self.extract_option_text(prompt, answer.strip()[0])
                
                
                encoding = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                ).to(self.device)
                # print(self.tokenizer.decode(encoding["input_ids"][0]))
                forward_check = True
                if forward_check:
                    forward_out = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
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
                    self.model.clear_all_router_stats()

                generated = self.model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding.get("attention_mask"),
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                generated_tokens = generated[0]
                input_len = encoding["input_ids"].shape[1]
                new_tokens = generated_tokens[input_len:]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                # print(self.tokenizer.decode(generated_tokens))
                pred_letter = self._extract_answer_letter(decoded)
                # breakpoint()
                if pred_letter == answer:
                    correct += 1
                elif answer_text in decoded:
                    correct_text += 1
                # if pred_letter != answer:
                #     breakpoint()
                total += 1

        router_counts = None
        router_average = None
        if router_accumulator:
            router_counts = {}
            agg_totals: Dict[str, float] = {}
            for layer, tensor in router_accumulator.items():
                counts = tensor.tolist()
                names_to_counts = self._aggregate_counts(layer, counts)
                router_counts[layer] = names_to_counts
                for name, value in names_to_counts.items():
                    agg_totals[name] = agg_totals.get(name, 0.0) + value
            num_layers = len(router_counts)
            if num_layers:
                router_average = {
                    name: total_value / num_layers for name, total_value in agg_totals.items()
                }
        accuracy = correct / total if total else 0.0
        accuracy_text = correct_text / total if total else 0.0

        
        
        return EvaluationResult(
            accuracy=accuracy,
            correct=correct,
            accuracy_text=accuracy_text,
            total=total,
            router_counts=router_counts,
            router_average=router_average,
        )