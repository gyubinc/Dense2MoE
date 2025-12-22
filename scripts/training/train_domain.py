#!/usr/bin/env python3
"""
Domain-specific LoRA training script
Unified script for training domain-specific LoRA adapters
"""

import argparse
import logging
import sys
import os
import time
import json
from typing import Dict, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
# Import from unified modules
from config.domains import domain_manager
from src.core.trainer import train_domain
from src.utils import (
    setup_logging, 
    validate_environment, 
    print_gpu_memory_summary, 
    clear_gpu_memory, 
    setup_cuda_environment, 
    setup_random_seed,
    save_config_to_output_dir
)
from src.utils.wandb_utils import (
    init_wandb,
    generate_run_name,
    log_training_metrics,
    log_system_metrics,
    finish_wandb,
    parse_command_args_for_wandb
)

def setup_logging_config():
    """도메인 학습 로그 포맷과 핸들러를 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('domain_training.log')
        ]
    )

def main():
    """도메인 전용 LoRA 학습을 수행한다."""
    parser = argparse.ArgumentParser(description="Train domain-specific LoRA adapters")
    parser.add_argument("--domain", required=True, 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to train (medical, law, math, code)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use for training (default: all)")
    parser.add_argument("--output-dir", default="domain_models",
                       help="Output directory for trained models")
    parser.add_argument("--device", default=None,
                       help="Device to use for training (default: from config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Training batch size (default: from config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (default: from config)")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to config file (default: config/config.yaml)")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable wandb logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging_config()
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting LoRA training for {args.domain} domain")
    logger.info(f"Configuration: max_samples={args.max_samples}, output_dir={args.output_dir}")
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            from config.moe import get_model_config, get_wandb_project_name
            model_cfg = get_model_config()
            project_name = get_wandb_project_name(model_cfg.model_type)
            
            run_name = generate_run_name("domain_training", domain=args.domain)
            wandb_config = parse_command_args_for_wandb(args)
            wandb_run = init_wandb(
                project_name=project_name,
                entity="gyubin5009",
                run_name=run_name,
                config=wandb_config,
                tags=["domain_training", args.domain, model_cfg.model_type]
            )
            logger.info(f"📊 Wandb logging enabled: {run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None
    
    # Setup CUDA environment from config
    setup_cuda_environment()
    
    # Setup random seed for reproducibility
    setup_random_seed()
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Check data availability
    availability = domain_manager.check_data_availability(args.domain)
    if not availability[args.domain]:
        logger.error(f"❌ Data not available for {args.domain} domain")
        return 1
    
    # Print GPU memory before training
    print_gpu_memory_summary("Before training")

    overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    start_time = time.time()

    try:
        # Train domain
        results = train_domain(
            domain=args.domain,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            overrides=overrides,
            device=args.device,
        )
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ {args.domain.upper()} domain training completed successfully")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        train_loss = results.get("train_loss")
        if train_loss is not None:
            logger.info(f"   Final loss: {train_loss:.4f}")
        else:
            logger.info("   Final loss: N/A")
        logger.info(f"   Adapter saved to: {results['adapter_path']}")
        
        # Log final metrics to wandb
        if wandb_run:
            try:
                log_training_metrics(
                    epoch=results.get('epoch', -1),
                    train_loss=train_loss,
                    learning_rate=results.get('learning_rate', args.learning_rate or 0),
                    training_time=training_time,
                    domain=args.domain
                )
                log_system_metrics()
            except Exception as e:
                logger.warning(f"Failed to log metrics to wandb: {e}")
        
        # Save training summary
        summary = {
            "domain": args.domain,
            "training_time": training_time,
            "results": results,
            "config": {
                "max_samples": args.max_samples,
                "output_dir": args.output_dir,
                "epochs": overrides.get("epochs"),
                "batch_size": overrides.get("batch_size"),
                "learning_rate": overrides.get("learning_rate"),
                "device": args.device or "config"
            }
        }
        
        summary_path = os.path.join(args.output_dir, f"{args.domain}_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to: {summary_path}")
        
        # Save config file to output directory
        domain_output_dir = os.path.join(args.output_dir, args.domain)
        config_saved_path = save_config_to_output_dir(args.config, domain_output_dir)
        if config_saved_path:
            logger.info(f"📝 Config file saved to: {config_saved_path}")
        
        # Finish wandb run
        if wandb_run:
            try:
                finish_wandb()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

        clear_gpu_memory()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed for {args.domain} domain: {e}")
        import traceback
        traceback.print_exc()
        
        # Finish wandb run even on failure
        if wandb_run:
            try:
                finish_wandb()
            except Exception as wandb_e:
                logger.warning(f"Failed to finish wandb run: {wandb_e}")

        clear_gpu_memory()
        
        return 1

if __name__ == "__main__":
    exit(main())
