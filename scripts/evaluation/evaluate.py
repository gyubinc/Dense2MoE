#!/usr/bin/env python3
"""MoE 라우터 평가 스크립트."""

import argparse
import logging
import os
import sys
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import torch

from src.utils.wandb_utils import (
    finish_wandb,
    generate_run_name,
    init_wandb,
    parse_command_args_for_wandb,
    log_something,
)

from config.moe import get_data_config, get_model_config, get_moe_config_manager
from config.domains import domain_manager
from src.core.dataset import RouterCollator, RouterDataset
from src.core.evaluator import RouterEvaluator
from src.utils import (
    print_gpu_memory_summary,
    setup_cuda_environment,
    setup_logging,
    setup_random_seed,
)


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
        "--domain",
        type=str,
        required=True,
        choices=get_data_config().domains,
        help="Domain used to filter evaluation samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum evaluation samples",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log accuracy and router stats to Weights & Biases",
    )
    parser.add_argument(
        "--cuda-num",
        type=int,
        default=None,
        help="Override GPU index for CUDA_VISIBLE_DEVICES (e.g., 3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k experts selection for evaluation (1 for hard routing, 2+ for soft routing)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override wandb run name",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    config_manager = get_moe_config_manager()
    config_raw = config_manager.load_config()

    # Apply runtime overrides for top_k if specified
    if args.top_k is not None:
        config_manager.set_runtime_overrides({"top_k": args.top_k})
        logger.info("Overriding top_k via --top-k: %d", args.top_k)

    if args.cuda_num is not None:
        cuda_value = str(args.cuda_num)
        if isinstance(config_raw, dict):
            gpu_section = config_raw.setdefault("gpu", {})
            if not isinstance(gpu_section, dict):
                gpu_section = {}
                config_raw["gpu"] = gpu_section
            gpu_section["cuda_visible_devices"] = cuda_value
        # breakpoint()
        logger.info("Overriding CUDA visible devices via --cuda-num: %s", cuda_value)

    wandb_run = None
    if args.use_wandb:
        try:
            from config.moe import get_wandb_project_name
            model_cfg = get_model_config()
            project_name = get_wandb_project_name(model_cfg.model_type)
            
            if args.wandb_run_name is not None:
                run_name = args.wandb_run_name
            else:
                run_name = generate_run_name(
                    "evaluation",
                    model_type="moe",
                    domain=args.domain,
                    moe_base=not bool(args.moe_model_path),
                    moe_model_path=args.moe_model_path,
                    top_k=args.top_k,
                )
            wandb_config = parse_command_args_for_wandb(args)
            if args.top_k is not None:
                wandb_config["top_k"] = args.top_k
            wandb_run = init_wandb(
                project_name=project_name,
                entity="gyubin5009",
                run_name=run_name,
                config=wandb_config,
                tags=["evaluation", "router", args.domain, model_cfg.model_type],
            )
        except Exception as exc:
            logger.warning("Failed to initialize wandb: %s", exc)
            wandb_run = None

    setup_cuda_environment()
    setup_random_seed()

    # print_gpu_memory_summary("before evaluation")
    
    try:
        evaluator = RouterEvaluator(model_path=args.moe_model_path)
    except Exception as exc:
        logger.error("Failed to initialize MoE model: %s", exc)
        if wandb_run:
            finish_wandb()
        raise
    data_cfg = get_data_config()
    tokenizer = evaluator.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = getattr(evaluator.model, "base_model", None)
    if base_model is not None and hasattr(base_model, "config"):
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "generation_config"):
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    config_raw = config_manager.load_config()
    data_section = config_raw.get("data", {}) if isinstance(config_raw, dict) else {}
    if args.domain in data_section.get("domains", []):
        domain_cfg = domain_manager.get_domain(args.domain)
        eval_path = domain_cfg.get_file_path("test")
    else:
        breakpoint()
        eval_path = data_section.get("eval_data_path", "data/processed/total/total_test.json")

    fallback = args.domain if eval_path != data_section.get("eval_data_path", "data/processed/total/total_test.json") else None
    dataset = RouterDataset(
        eval_path,
        tokenizer=tokenizer,
        max_length=get_model_config().max_length,
        max_samples=args.max_samples,
        fallback_domain=fallback,
    )
    if fallback is None:
        dataset.samples = [s for s in dataset.samples if s.domain == args.domain]
    result = evaluator.evaluate(dataset)


    '''
    output of the evaluate function

    return EvaluationResult(
        accuracy=accuracy,
        correct=correct,
        total=total,
        router_counts=router_counts,
        router_average=router_average,
    '''

    average_dict = {}
    logger.info("Accuracy: %.4f (%d/%d)", result.accuracy, result.correct, result.total)
    if result.router_counts:
        for layer, counts in sorted(result.router_counts.items()):
            total = sum(counts.values()) or 1.0
            info = ", ".join(
                f"{name}:{(value / total):.2%}" for name, value in counts.items()
            )
            logger.info("Layer %02d routing: %s", layer, info)
        if result.router_average:
            total = sum(result.router_average.values()) or 1.0
            info = ", ".join(
                f"{name}:{(value / total):.2%}" for name, value in result.router_average.items()
            )
            
            for name, value in result.router_average.items():
                share = value / total
                average_dict[name] = round(float(share * 100), 2)
            logger.info("Average routing across layers: %s", info)

    # add result.accuracy as accuracy, result.total as total_samples to average_dict
    average_dict["domain"] = args.domain
    average_dict["accuracy"] = round(float(result.accuracy * 100), 2)
    average_dict["accuracy_text"] = round(float(result.accuracy_text * 100), 2)
    average_dict['total_accuracy'] = round(float((result.accuracy + result.accuracy_text) * 100), 2)
    average_dict["total_samples"] = result.total
    


    if wandb_run:
        log_something(average_dict=average_dict)
        finish_wandb()
    

    # if wandb_run:
    #     try:
    #         router_flat = None
    #         if result.router_counts:
    #             router_flat = {
    #                 f"layer_{layer}_{name}_share": value
    #                 for layer, counts in result.router_counts.items()
    #                 for name, value in counts.items()
    #             }
    #         if result.router_average:
    #             if router_flat is None:
    #                 router_flat = {}
    #             for name, value in result.router_average.items():
    #                 router_flat[f"average_{name}_share"] = value
    #         log_evaluation_metrics(
    #             accuracy=result.accuracy,
    #             correct_predictions=result.correct,
    #             total_samples=result.total,
    #             domain=args.domain,
    #             router_statistics=router_flat,
    #         )
    #     finally:
    #         finish_wandb()

    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
