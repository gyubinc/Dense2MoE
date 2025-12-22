#!/usr/bin/env python3
"""
MoE Router training script
Unified script for training MoE router on combined dataset
"""

import argparse
import logging
import sys
import os
import time
from typing import Any, Dict

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import from unified modules
from src.core.trainer import RouterTrainer, RouterTrainConfig
from src.utils import (
    setup_cuda_environment,
    validate_environment,
)
from src.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_system_metrics,
    log_training_metrics,
    parse_command_args_for_wandb,
)

def setup_logging_config():
    """라우터 학습 로그 포맷과 핸들러를 구성한다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('router_training.log')
        ]
    )

def main():
    """MoE 라우터 학습을 수행한다."""
    parser = argparse.ArgumentParser(description="Train MoE Router with flexible component selection")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for training (None for all)",
    )
    parser.add_argument(
        "--output-dir",
        default="moe_models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU ID to use (default: from config)",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable wandb logging"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size",
    )
    

    parser.add_argument(
        "--load-balancing-loss-weight",
        type=float,
        default=None,
        help="Override load balancing loss weight",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k experts selection (1 for hard routing, 2+ for soft routing)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="router",
        help="Override target (attention, mlp, router, attention+mlp)",
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging_config()
    logger = logging.getLogger(__name__)
    
    resolved_target = args.target or "router"
    
    # Log training configuration
    logger.info("🚀 Starting MoE Router Training")
    logger.info("📁 Output directory: %s", args.output_dir)
    logger.info("📊 Max samples: %s", args.max_samples)
    logger.info("🎮 GPU ID: %s", args.gpu_id)
    logger.info("🎯 Training target: %s", resolved_target)
    
    # Build overrides once (shared by wandb and trainer)
    overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        overrides["eval_batch_size"] = args.eval_batch_size
    if args.load_balancing_loss_weight is not None:
        overrides["load_balancing_loss_weight"] = args.load_balancing_loss_weight
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    overrides["target"] = resolved_target

    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            trainer_config = RouterTrainConfig.from_config(overrides)
            # lr_str = f"{trainer_config.learning_rate:.2e}" if trainer_config.learning_rate else "na"
            lr = trainer_config.learning_rate
            lr_str = f"{lr:.2e}".replace("e-0", "e-").replace("e+0", "e+")
            loss_w = getattr(trainer_config, "load_balancing_loss_weight", None)
            loss_str = (
                f"{loss_w:.2e}".replace("e-0", "e-").replace("e+0", "e+")
                if loss_w is not None
                else "na"
            )
            max_samples_str = str(args.max_samples) if args.max_samples is not None else "all"
            target = resolved_target
            run_name = f"moe-{target}-lr{lr_str}-aux{loss_str}-epoch{trainer_config.epochs}-max{max_samples_str}-{time.strftime('%m%d')}"

            wandb_config = parse_command_args_for_wandb(args)
            wandb_config.update({
                "learning_rate": trainer_config.learning_rate,
                "load_balancing_loss_weight": loss_w,
                "epochs": trainer_config.epochs,
                "batch_size": trainer_config.batch_size,
                "eval_batch_size": trainer_config.eval_batch_size,
                "max_samples": args.max_samples,
                "top_k": args.top_k if args.top_k is not None else overrides.get("top_k", 1),
            })
            tags = ["moe_training", "router"]

            # Get model type for dynamic wandb project name
            from config.moe import get_model_config, get_wandb_project_name
            model_cfg = get_model_config()
            project_name = get_wandb_project_name(model_cfg.model_type)
            tags.append(model_cfg.model_type)
            
            wandb_run = init_wandb(
                project_name=project_name,
                entity="gyubin5009",
                run_name=run_name,
                config=wandb_config,
                tags=tags,
            )
            logger.info("📊 Wandb logging enabled: %s", run_name)
        except Exception as exc:
            logger.warning("Failed to initialize wandb: %s", exc)
            wandb_run = None
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        setup_cuda_environment()
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Print GPU memory before training
    # print_gpu_memory_summary("Before training")
    
    start_time = time.time()
    
    try:
        trainer = RouterTrainer(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            config_overrides=overrides,
            target=resolved_target,
        )
        results = trainer.train()

        training_time = time.time() - start_time
        
        logger.info("✅ MoE training completed successfully")
        logger.info("   Training time: %.2f seconds", training_time)
        logger.info("   Steps: %s", results.get("total_steps"))
        logger.info("   Model saved to: %s", results.get("model_path"))
        
        # Log final metrics to wandb
        if wandb_run:
            try:
                latest = (results.get("history") or [])[-1] if results.get("history") else {}
                log_training_metrics(
                    epoch=latest.get("step", results.get("total_steps", 0)),
                    train_loss=latest.get("eval_loss", 0.0),
                    learning_rate=overrides.get("learning_rate", 0.0),
                    training_time=training_time,
                    model_type="moe_router",
                )
                log_system_metrics()
            except Exception as e:
                logger.warning(f"Failed to log metrics to wandb: {e}")
        
        
        # Save training summary
        logger.info("📊 Training summary available under: %s", results.get("model_path"))

        if wandb_run:
            try:
                finish_wandb()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")
        
        return 0
        
    except Exception as e:
        logger.error("❌ Router training failed: %s", e)
        import traceback

        traceback.print_exc()

        # Finish wandb run even on failure
        if wandb_run:
            try:
                finish_wandb()
            except Exception as wandb_e:
                logger.warning(f"Failed to finish wandb run: {wandb_e}")
        
        return 1

if __name__ == "__main__":
    exit(main())
