import os
import math
import logging
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from config.moe import (
    get_data_config,
    get_gpu_config,
    get_model_config,
    get_moe_config,
    get_moe_config_manager,
)
from config.training import get_training_config
from ..models.model import MoEModel
from .dataset import DomainDataset, RouterCollator, RouterDataset
from ..utils.utils import print_gpu_memory_summary, setup_random_seed
from ..utils.wandb_utils import log_eval_accuracy, log_training_metrics

# AutoTokenizer import
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """tqdm 기반 진행률 바를 제공하는 콜백."""

    def __init__(self, description: str = "LoRA Training"):
        self.description = description
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        self.progress_bar = tqdm(total=total, desc=self.description, unit="step")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress_bar is not None and logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            postfix = {}
            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.2e}"
            if postfix:
                self.progress_bar.set_postfix(postfix, refresh=False)

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class RouterTrainConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    load_balancing_loss_weight: float
    logging_steps: int
    eval_steps: int
    bf16: bool
    gradient_checkpointing: bool

    @classmethod
    def from_config(cls, overrides: Optional[Dict[str, Any]] = None) -> "RouterTrainConfig":
        moe_cfg = get_moe_config()
        overrides = overrides or {}
        eval_override = overrides.get("eval_batch_size")
        if eval_override is not None:
            eval_batch_size = int(eval_override)
        elif moe_cfg.eval_batch_size is not None:
            eval_batch_size = int(moe_cfg.eval_batch_size)
        else:
            eval_batch_size = int(moe_cfg.batch_size)

        batch_size = int(overrides.get("batch_size", moe_cfg.batch_size))

        return cls(
            epochs=int(overrides.get("epochs", moe_cfg.num_epochs)),
            batch_size=batch_size,
            eval_batch_size=max(1, eval_batch_size),
            gradient_accumulation_steps=int(
                overrides.get("gradient_accumulation_steps", moe_cfg.gradient_accumulation_steps)
            ),
            learning_rate=float(overrides.get("learning_rate", moe_cfg.learning_rate)),
            weight_decay=float(overrides.get("weight_decay", moe_cfg.weight_decay)),
            warmup_ratio=float(overrides.get("warmup_ratio", moe_cfg.warmup_ratio)),
            max_grad_norm=float(overrides.get("max_grad_norm", moe_cfg.max_grad_norm)),
            load_balancing_loss_weight=float(
                overrides.get("load_balancing_loss_weight", moe_cfg.load_balancing_loss_weight)
            ),
            logging_steps=int(overrides.get("logging_steps", moe_cfg.logging_steps)),
            eval_steps=int(overrides.get("eval_steps", moe_cfg.eval_steps)),
            bf16=bool(overrides.get("bf16", moe_cfg.bf16)),
            gradient_checkpointing=bool(overrides.get("gradient_checkpointing", False)),
        )


class RouterTrainer:
    """라우터 전용 학습 루프를 담당하는 헬퍼."""

    def __init__(
        self,
        output_dir: str,
        max_samples: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        target: Optional[str] = "router",
    ) -> None:
        # Apply runtime overrides to global config manager
        if config_overrides:
            get_moe_config_manager().set_runtime_overrides(config_overrides)
        overrides = config_overrides or {}

        self.config = RouterTrainConfig.from_config(config_overrides)
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.device = torch.device(get_gpu_config().device)
        eval_inference_override = overrides.get("eval_inference_batch_size")
        if eval_inference_override is not None:
            eval_inference_bs = int(eval_inference_override)
        else:
            eval_inference_bs = self.config.eval_batch_size
        self.eval_inference_batch_size = max(1, eval_inference_bs)
        decode_override = overrides.get("eval_decode_max_tokens")
        if decode_override is not None:
            decode_limit = int(decode_override)
        else:
            decode_limit = 32
        self.eval_decode_max_tokens = decode_limit if decode_limit > 0 else None
        try:
            import wandb  # type: ignore

            self._use_wandb = wandb.run is not None
        except Exception:
            self._use_wandb = False

        _ensure_dir(self.output_dir)
        self.checkpoint_filename = "pytorch_model.bin"
        self.best_checkpoint_dir = os.path.join(self.output_dir, "best_model")
        if os.path.exists(self.best_checkpoint_dir):
            shutil.rmtree(self.best_checkpoint_dir)
        _ensure_dir(self.best_checkpoint_dir)
        self._best_accuracy = float("-inf")
        self._best_step = -1
        setup_random_seed()
        
        # target 정규화
        if isinstance(target, str) and target.strip():
            normalized_target = target.strip().lower()
        else:
            normalized_target = "router"
        self.target = normalized_target

        # MoEModel 생성 (tokenizer 포함)
        self.model = MoEModel(target=self.target)
        self.model.to(self.device)
        self.model.setup_training_parameters(target=self.target)
        
        # MoEModel의 tokenizer 재사용
        self.tokenizer = self.model.tokenizer

        
        train_from_router = False
        if train_from_router:
            if get_moe_config().top_k == 1:
                router_ckpt = "/data/disk5/internship_disk/gyubin/Llama_MoE/top1_router_models/epoch1_12600_noaux_5e4/final_model/pytorch_model.bin"
            elif get_moe_config().top_k == 2:
                router_ckpt = "/data/disk5/internship_disk/gyubin/Llama_MoE/top2_router_models/epoch1_12600_noaux_2e4/final_model/pytorch_model.bin"

            state = torch.load(router_ckpt, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            del state


        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        data_cfg = get_data_config()
        model_cfg = get_model_config()
        config_raw = get_moe_config_manager().load_config()
        data_section = config_raw.get("data", {}) if isinstance(config_raw, dict) else {}

        train_path = data_section.get("train_data_path", "data/processed/total/total_train.json")
        eval_path = data_section.get("eval_data_path", "data/processed/total/total_test.json")

        train_dataset = RouterDataset(
            train_path,
            tokenizer=self.tokenizer,
            max_length=model_cfg.max_length,
            max_samples=self.max_samples,
        )
        eval_max = (
            min(data_cfg.eval_max_samples, self.max_samples)
            if self.max_samples is not None
            else data_cfg.eval_max_samples
        )
        eval_dataset = RouterDataset(
            eval_path,
            tokenizer=self.tokenizer,
            max_length=model_cfg.max_length,
            max_samples=eval_max,
        )

        collator = RouterCollator(self.tokenizer.pad_token_id)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        eval_batch_size = max(1, self.config.eval_batch_size)
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        self.router_params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.router_params:
            raise RuntimeError("Router parameters are not trainable")

        steps_per_epoch = len(self.train_loader)
        total_batches = steps_per_epoch * self.config.epochs
        total_steps = math.ceil(total_batches / self.config.gradient_accumulation_steps)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
        self.total_samples = len(self.train_dataset) * self.config.epochs

        self.optimizer = AdamW(
            self.router_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            "Router trainer initialised: epochs=%d, batches/epoch=%d, total_steps=%d, warmup=%d",
            self.config.epochs,
            steps_per_epoch,
            total_steps,
            warmup_steps,
        )

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.bf16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_router_loss=True,
            )
            main_loss = outputs.loss
            
            router_loss = getattr(outputs, "router_loss", None)

            # ✅ NaN 방어
            if not torch.isfinite(main_loss):
                logger.warning("main_loss non-finite detected — applying nan_to_num")
                logger.warning("-" * 30)
                main_loss = torch.nan_to_num(main_loss, nan=0.0, posinf=1e4, neginf=-1e4)

            if router_loss is None:
                router_loss = torch.zeros((), device=self.device, dtype=main_loss.dtype)
            elif not torch.isfinite(router_loss):
                logger.warning("router_loss non-finite detected — applying nan_to_num")
                logger.warning("-" * 30)
                router_loss = torch.nan_to_num(router_loss, nan=0.0, posinf=1e4, neginf=-1e4)


            
            
            # breakpoint()
            total_loss = main_loss + self.config.load_balancing_loss_weight * router_loss
        


        return {
            "total_loss": total_loss,
            "main_loss": main_loss.detach(),
            "router_loss": router_loss.detach(),
        }

    def _log_step(
        self,
        step: int,
        stats: Dict[str, torch.Tensor],
        avg_total: Optional[float] = None,
        avg_main: Optional[float] = None,
        avg_router: Optional[float] = None,
    ) -> None:
        total_val = stats["total_loss"].item()
        main_val = stats["main_loss"].item()
        router_val = stats["router_loss"].item()

        logger.info(
            "[step %d] latest loss=%.4f | main=%.4f | router=%.4f",
            step,
            total_val,
            main_val,
            router_val,
        )
        if avg_total is not None:
            logger.info(
                "[step %d] window_avg loss=%.4f | main=%.4f | router=%.4f",
                step,
                avg_total,
                avg_main if avg_main is not None else float("nan"),
                avg_router if avg_router is not None else float("nan"),
            )

        print_gpu_memory_summary(f"step {step}")

        if self._use_wandb:
            try:
                if hasattr(self, "scheduler") and getattr(self.scheduler, "get_last_lr", None):
                    lr = float(self.scheduler.get_last_lr()[0])
                else:
                    lr = self.config.learning_rate

                logged_total = avg_total if avg_total is not None else total_val
                logged_main = avg_main if avg_main is not None else main_val
                logged_router = (
                    avg_router if avg_router is not None else router_val
                )

                log_training_metrics(
                    epoch=getattr(self, "_current_epoch", -1),
                    train_loss=logged_total,
                    learning_rate=lr,
                    step=step,
                    main_loss=logged_main,
                    router_loss=logged_router,
                )
            except Exception as exc:
                logger.warning("Failed to log training metrics to wandb: %s", exc)
        
        

    @staticmethod 
    def _extract_answer_letter(text: str) -> str:
        if not text:
            return ""
        work = text.strip()
        work_upper = work.upper()

        for marker in ["ANSWER:", "ANSWER :", "ANSWER"]:
            if marker in work_upper:
                idx = work_upper.find(marker)
                work = work[idx + len(marker) :]
                break

        work = work.strip()
        for ch in work:
            upper = ch.upper()
            if upper in {"A", "B", "C", "D", "E"}:
                return upper
        return ""

    def _evaluate(self, step: int) -> Dict[str, float]:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            eval_iterator = tqdm(
                self.eval_loader,
                desc=f"Eval @ step {step}",
                unit="batch",
                leave=False,
            )
            for batch in eval_iterator:
                answers = batch.get("answer", [])
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_size = input_ids.size(0)
                for start in range(0, batch_size, self.eval_inference_batch_size):
                    end = min(start + self.eval_inference_batch_size, batch_size)
                    sub_input_ids = input_ids[start:end]
                    sub_attention_mask = attention_mask[start:end]
                    sub_labels = labels[start:end]
                    sub_answers = answers[start:end]

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=self.config.bf16,
                    ):
                        outputs = self.model(
                            input_ids=sub_input_ids,
                            attention_mask=sub_attention_mask,
                        )
                        sub_logits = outputs.logits

                    shift_logits = sub_logits[:, :-1, :]
                    shift_labels = sub_labels[:, 1:]
                    shift_predictions = shift_logits.argmax(dim=-1)
                    shift_label_mask = shift_labels != -100

                    for local_idx, answer in enumerate(sub_answers):
                        valid_mask = shift_label_mask[local_idx]
                        if not bool(valid_mask.any()):
                            total += 1
                            continue

                        predicted_ids = shift_predictions[local_idx][valid_mask]

                        if self.tokenizer.eos_token_id is not None:
                            eos_positions = (predicted_ids == self.tokenizer.eos_token_id).nonzero(
                                as_tuple=False
                            )
                            if eos_positions.numel() > 0:
                                eos_index = eos_positions[0].item()
                                predicted_ids = predicted_ids[:eos_index]

                        if (
                            self.eval_decode_max_tokens is not None
                            and predicted_ids.numel() > self.eval_decode_max_tokens
                        ):
                            predicted_ids = predicted_ids[: self.eval_decode_max_tokens]

                        decoded = (
                            self.tokenizer.decode(
                                predicted_ids.detach().cpu().tolist(),
                                skip_special_tokens=True,
                            )
                            if predicted_ids.numel() > 0
                            else ""
                        )
                        prediction = self._extract_answer_letter(decoded)
                        
                        if prediction == answer:
                            correct += 1
                        total += 1

                        if hasattr(self.model, "clear_all_router_stats"):
                            self.model.clear_all_router_stats()
                
                if total:
                    eval_iterator.set_postfix({"acc": f"{correct / total:.3f}"})
        
        accuracy = correct / total if total else 0.0
        metrics = {
            "eval_accuracy": accuracy,
            "eval_correct": float(correct),
            "eval_total": float(total),
        }
        logger.info("[eval %d] accuracy=%.4f (%d/%d)", step, accuracy, correct, total)
        if self._use_wandb:
            try:
                log_eval_accuracy(step, accuracy)
            except Exception as exc:
                logger.warning("Failed to log eval accuracy to wandb: %s", exc)
        self.model.train()
        return metrics
    
    def _save_checkpoint(self, target_dir: str) -> None:
        _ensure_dir(target_dir)
        torch.save(self.model.state_dict(), os.path.join(target_dir, self.checkpoint_filename))
        self.tokenizer.save_pretrained(target_dir)

    @staticmethod
    def _copy_checkpoint_directory(source_dir: str, target_dir: str) -> bool:
        if not os.path.isdir(source_dir):
            return False
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        return True
    
    def train(self) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        global_step = 0
        accum = 0
        interval_loss_total = 0.0
        interval_main_total = 0.0
        interval_router_total = 0.0
        interval_samples = 0

        self.model.train()

        with tqdm(total=self.total_samples, desc="Router Training", unit="sample") as progress_bar:
            for epoch in range(1, self.config.epochs + 1):
                self._current_epoch = epoch
                logger.info("Epoch %d/%d", epoch, self.config.epochs)
                progress_bar.set_description(f"Epoch {epoch}/{self.config.epochs}")

                for batch_idx, batch in enumerate(self.train_loader, start=1):
                    stats = self._forward(batch)


                    loss = stats["total_loss"] / self.config.gradient_accumulation_steps
                    loss.backward()
                    accum += 1



                    batch_size = batch["input_ids"].size(0)
                    progress_bar.update(batch_size)
                    interval_samples += batch_size
                    interval_loss_total += stats["total_loss"].item() * batch_size
                    interval_main_total += stats["main_loss"].item() * batch_size
                    interval_router_total += stats["router_loss"].item() * batch_size

                    if torch.isnan(loss):
                        breakpoint()


                    if accum >= self.config.gradient_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(
                            self.router_params, self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        accum = 0
                        global_step += 1

                        avg_loss = (
                            interval_loss_total / interval_samples
                            if interval_samples > 0
                            else float("nan")
                        )
                        avg_main = (
                            interval_main_total / interval_samples
                            if interval_samples > 0
                            else float("nan")
                        )
                        avg_router = (
                            interval_router_total / interval_samples
                            if interval_samples > 0
                            else float("nan")
                        )
                        progress_bar.set_postfix_str(
                            f"epoch={epoch} batch={batch_idx}/{len(self.train_loader)} window_avg={avg_loss:.3f} main={avg_main:.3f} router={avg_router:.3f}"
                        )

                        if (
                            self.config.logging_steps > 0
                            and global_step % self.config.logging_steps == 0
                            and interval_samples > 0
                        ):
                            self._log_step(
                                global_step,
                                stats,
                                avg_total=avg_loss,
                                avg_main=avg_main,
                                avg_router=avg_router,
                            )
                            interval_loss_total = 0.0
                            interval_main_total = 0.0
                            interval_router_total = 0.0
                            interval_samples = 0

                        if (
                            self.config.eval_steps > 0
                            and global_step % self.config.eval_steps == 0
                        ):
                            eval_metrics = self._evaluate(global_step)
                            history.append({"step": global_step, **eval_metrics})
                            accuracy = eval_metrics.get("eval_accuracy", 0.0)
                            if accuracy > self._best_accuracy:
                                self._best_accuracy = accuracy
                                self._best_step = global_step
                                self._save_checkpoint(self.best_checkpoint_dir)
                                logger.info(
                                    "New best eval accuracy=%.4f at step %d. Checkpoint saved to %s",
                                    accuracy,
                                    global_step,
                                    self.best_checkpoint_dir,
                                )

                torch.cuda.empty_cache()

        _ensure_dir(self.output_dir)
        final_dir = os.path.join(self.output_dir, "final_model")
        if (
            self._best_accuracy > float("-inf")
            and self._copy_checkpoint_directory(self.best_checkpoint_dir, final_dir)
        ):
            logger.info(
                "Best model (step %d, accuracy=%.4f) saved to %s",
                self._best_step,
                self._best_accuracy,
                final_dir,
            )
        else:
            _ensure_dir(final_dir)
            self._save_checkpoint(final_dir)
            logger.info("No evaluation checkpoint found. Saved latest model to %s", final_dir)

        summary = {
            "total_steps": global_step,
            "history": history,
            "config": asdict(self.config),
            "model_path": final_dir,
            "best_eval_accuracy": self._best_accuracy if self._best_accuracy > float("-inf") else None,
            "best_step": self._best_step if self._best_accuracy > float("-inf") else None,
        }
        logger.info("Training finished. Model saved to %s", final_dir)

        return summary


def train_domain(
    domain: str,
    max_samples: Optional[int] = None,
    output_dir: str = "domain_models",
    use_wandb: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """도메인별 LoRA 어댑터를 학습하고 저장 경로를 반환한다."""
    overrides = overrides or {}

    training_cfg = get_training_config("domain")
    lora_cfg = training_cfg.lora
    data_cfg = get_data_config()
    model_cfg = get_model_config()
    gpu_cfg = get_gpu_config()

    epochs = int(overrides.get("epochs", training_cfg.epochs))
    batch_size = int(overrides.get("batch_size", training_cfg.batch_size))
    grad_accum = int(overrides.get("gradient_accumulation_steps", training_cfg.gradient_accumulation_steps))
    learning_rate = float(overrides.get("learning_rate", training_cfg.learning_rate))
    weight_decay = float(overrides.get("weight_decay", training_cfg.weight_decay))
    warmup_ratio = float(overrides.get("warmup_ratio", training_cfg.warmup_ratio))
    max_grad_norm = float(overrides.get("max_grad_norm", training_cfg.max_grad_norm))
    logging_steps = int(overrides.get("logging_steps", training_cfg.logging_steps))
    eval_steps = int(overrides.get("eval_steps", training_cfg.eval_steps))
    gradient_checkpointing = bool(
        overrides.get("gradient_checkpointing", training_cfg.gradient_checkpointing)
    )
    eval_batch_size = int(overrides.get("eval_batch_size", batch_size))
    eval_max_samples = overrides.get("eval_max_samples", data_cfg.eval_max_samples)

    device_target = device or gpu_cfg.device
    if not use_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")
    else:
        if os.environ.get("WANDB_DISABLED") == "true":
            os.environ.pop("WANDB_DISABLED")

    setup_random_seed()

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = device_target if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if not torch.cuda.is_available():
        model.to(device_target)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_dataset = DomainDataset(
        domain=domain,
        tokenizer=tokenizer,
        max_length=model_cfg.max_length,
        split="train",
        max_samples=max_samples,
    )

    eval_dataset = None
    evaluation_strategy = "no"
    if eval_steps > 0 and eval_max_samples:
        try:
            eval_dataset = DomainDataset(
                domain=domain,
                tokenizer=tokenizer,
                max_length=model_cfg.max_length,
                split="test",
                max_samples=eval_max_samples,
            )
            evaluation_strategy = "steps"
        except FileNotFoundError:
            logger.warning(
                "Evaluation dataset for domain=%s not found; skipping evaluation.", domain
            )
            eval_dataset = None

    collator = RouterCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)

    domain_root = os.path.join(output_dir, domain)
    checkpoints_dir = os.path.join(domain_root, "checkpoints")
    logs_dir = os.path.join(domain_root, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)

    bf16_flag = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    ta_fields = set(getattr(TrainingArguments, "__dataclass_fields__", {}).keys())
    ta_kwargs: Dict[str, Any] = {
        "output_dir": checkpoints_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "logging_steps": logging_steps,
        "logging_dir": logs_dir,
        "save_total_limit": 1,
    }

    if "evaluation_strategy" in ta_fields:
        ta_kwargs["evaluation_strategy"] = evaluation_strategy
        if evaluation_strategy != "no":
            ta_kwargs["eval_steps"] = eval_steps
    else:
        if evaluation_strategy != "no":
            if "do_eval" in ta_fields:
                ta_kwargs["do_eval"] = True
            if "eval_steps" in ta_fields:
                ta_kwargs["eval_steps"] = eval_steps

    if "save_strategy" in ta_fields:
        ta_kwargs["save_strategy"] = "no"
    elif evaluation_strategy != "no" and "save_steps" in ta_fields:
        ta_kwargs["save_steps"] = max(1, eval_steps)

    if "bf16" in ta_fields:
        ta_kwargs["bf16"] = bf16_flag
    if "fp16" in ta_fields:
        ta_kwargs["fp16"] = False
    if gradient_checkpointing and "gradient_checkpointing" in ta_fields:
        ta_kwargs["gradient_checkpointing"] = True
    if "report_to" in ta_fields:
        ta_kwargs["report_to"] = ["wandb"] if use_wandb else []
    if "lr_scheduler_type" in ta_fields:
        ta_kwargs["lr_scheduler_type"] = "cosine"
    if "remove_unused_columns" in ta_fields:
        ta_kwargs["remove_unused_columns"] = False
    if "dataloader_pin_memory" in ta_fields:
        ta_kwargs["dataloader_pin_memory"] = False
    if "dataloader_num_workers" in ta_fields:
        ta_kwargs["dataloader_num_workers"] = 0
    if "optim" in ta_fields:
        ta_kwargs["optim"] = "adamw_torch"
    if "run_name" in ta_fields:
        ta_kwargs["run_name"] = f"{domain}-lora-training"
    if "disable_tqdm" in ta_fields:
        ta_kwargs["disable_tqdm"] = True

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[TrainingProgressCallback(description=f"{domain} LoRA Training")],
    )

    logger.info(
        "Starting LoRA training for domain=%s (samples=%d, epochs=%d, batch=%d, grad_accum=%d)",
        domain,
        len(train_dataset),
        epochs,
        batch_size,
        grad_accum,
    )

    train_output = trainer.train()
    trainer.save_state()

    os.makedirs(domain_root, exist_ok=True)
    adapter_dir = os.path.join(domain_root, "final_adapter")
    if os.path.exists(adapter_dir):
        shutil.rmtree(adapter_dir)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    current_lr = (
        trainer.optimizer.param_groups[0]["lr"]
        if trainer.optimizer and trainer.optimizer.param_groups
        else learning_rate
    )

    train_loss = train_output.metrics.get("train_loss")
    result = {
        "train_loss": float(train_loss) if train_loss is not None else None,
        "adapter_path": adapter_dir,
        "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else epochs,
        "learning_rate": float(current_lr),
        "global_step": trainer.state.global_step,
        "train_runtime": train_output.metrics.get("train_runtime"),
        "train_samples": len(train_dataset),
    }

    logger.info(
        "LoRA adapter for domain=%s saved to %s (train_loss=%s)",
        domain,
        adapter_dir,
        result["train_loss"],
    )

    torch.cuda.empty_cache()
    return result


def train_router(
    max_samples: Optional[int] = None,
    output_dir: str = "moe_models",
    training_overrides: Optional[Dict[str, Any]] = None,
    target: Optional[str] = "router",
) -> Dict[str, Any]:
    trainer = RouterTrainer(
        output_dir=output_dir,
        max_samples=max_samples,
        config_overrides=training_overrides,
        target=target,
    )
    return trainer.train()

