#!/usr/bin/env python3
"""Wandb 로깅을 일관되게 관리하는 유틸리티."""

import os
import json
import wandb
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_wandb_key() -> str:
    """wandb API 키를 로컬 JSON에서 읽어온다."""

    # key 위치 본인에 맞게 설정
    # key는 dict형식 ('wandb_token': 'api_key')
    key_path = "/data/disk5/internship_disk/gyubin/wandb_key.json"

    try:
        with open(key_path, 'r') as f:
            key_data = json.load(f)
            return key_data['wandb_token']
    except Exception as e:
        logger.error(f"Failed to load wandb key: {e}")
        raise

def generate_run_name(command_type: str, **kwargs) -> str:
    """
    Generate wandb run name based on command type and parameters
    
    Args:
        command_type: Type of command (domain_training, moe_training, evaluation)
        **kwargs: Additional parameters for name generation
    
    Returns:
        Generated run name
    """
    # Get current date in MMDD format
    current_date = datetime.now().strftime("%m%d")
    
    if command_type == "domain_training":
        domain = kwargs.get('domain', 'unknown')
        return f"domain-{domain}-{current_date}"
    
    elif command_type == "moe_training":
        return f"moe-router-{current_date}"
    
    elif command_type == "evaluation":
        model_type = kwargs.get('model_type', 'unknown')
        domain = kwargs.get('domain', 'unknown')
        
        if model_type == "moe":
            if kwargs.get('moe_base', False):
                return f"eval-moe-base"
            else:
                moe_model_path = kwargs.get('moe_model_path', 'unknown')
                moe_model_path = moe_model_path.split('/')[-3]
                return f"eval-{moe_model_path}"
        elif model_type == "domain":
            if kwargs.get('adapter_path'):
                # Extract adapter type from path
                adapter_path = kwargs.get('adapter_path', '')
                adapter_name = adapter_path.split('/')[1]
                
                return f"eval-adapter-{adapter_name}"
            else:
                return f"eval-llama-base"
    
    return f"unknown-{current_date}"

def init_wandb(
    project_name: str = "Llama_MoE",
    entity: str = "gyubin5009",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None
) -> Any:
    """
    Initialize wandb run
    
    Args:
        project_name: Name of the wandb project
        entity: Wandb entity/username
        run_name: Name of the run (auto-generated if None)
        config: Configuration dictionary to log
        tags: List of tags for the run
    
    Returns:
        Initialized wandb run
    """
    try:
        # Load wandb key
        wandb_key = load_wandb_key()
        os.environ["WANDB_API_KEY"] = wandb_key
        
        # Initialize wandb
        run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config=config,
            tags=tags or [],
            reinit=True,
            settings=wandb.Settings(_disable_stats=True)
        )
        
        logger.info(f"✅ Wandb initialized: {run.name}")
        return run
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize wandb: {e}")
        raise

def log_training_metrics(
    epoch: int,
    train_loss: float,
    learning_rate: float,
    step: Optional[int] = None,
    **kwargs
):
    """Log training metrics to wandb"""
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        **kwargs
    }
    
    # Add evaluation metrics if available
    eval_accuracy = kwargs.get('eval_accuracy')
    if eval_accuracy is not None:
        metrics['eval_accuracy'] = eval_accuracy
        
    eval_loss = kwargs.get('eval_loss')
    if eval_loss is not None:
        metrics['eval_loss'] = eval_loss
    
    wandb.log(metrics, step=step)

def log_something(average_dict: dict):
    """임시 집계 결과를 wandb에 전송한다."""
    wandb.log(average_dict)


def log_eval_accuracy(step: int, accuracy: float, **kwargs) -> None:
    """평가 정확도를 wandb에 기록한다."""
    payload = {
        "eval_accuracy": accuracy,
        **kwargs,
    }
    wandb.log(payload, step=step)

def log_system_metrics():
    """GPU 메모리 등 시스템 상태를 wandb에 기록한다."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            wandb.log({
                "gpu_memory_allocated": gpu_memory,
                "gpu_memory_reserved": gpu_memory_reserved,
                "gpu_memory_max": gpu_memory_max
            })
    except Exception as e:
        logger.warning(f"Failed to log system metrics: {e}")

def finish_wandb():
    """wandb 런을 안전하게 종료한다."""
    try:
        wandb.finish()
        # logger.info("✅ Wandb run finished")
    except Exception as e:
        logger.warning(f"Failed to finish wandb run: {e}")

def parse_command_args_for_wandb(args) -> Dict[str, Any]:
    """
    Parse command line arguments to extract wandb configuration
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Dictionary with wandb configuration
    """
    config = {}
    
    # Common parameters
    if hasattr(args, 'max_samples') and args.max_samples:
        config['max_samples'] = args.max_samples
    if hasattr(args, 'epochs') and args.epochs:
        config['epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config['batch_size'] = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if hasattr(args, 'output_dir') and args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Domain-specific parameters
    if hasattr(args, 'domain') and args.domain:
        config['domain'] = args.domain
    
    # MoE-specific parameters
    if hasattr(args, 'moe_model_path') and args.moe_model_path:
        config['moe_model_path'] = args.moe_model_path
    if hasattr(args, 'moe_base') and args.moe_base:
        config['moe_base'] = args.moe_base
    if hasattr(args, 'adapter_path') and args.adapter_path:
        config['adapter_path'] = args.adapter_path
    
    return config
