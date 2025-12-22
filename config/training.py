#!/usr/bin/env python3
"""LoRA 및 라우터 학습 설정 정의."""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """도메인 학습 시 사용되는 LoRA 설정."""
    r: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["gate_proj", "up_proj", "down_proj"]

@dataclass
class TrainingConfig:
    """공통 학습 설정."""
    # Basic training parameters
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimization
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    
    # Evaluation
    eval_steps: int = 500
    
    # Logging and saving
    logging_steps: int = 10
    
    # Data processing
    
    # LoRA configuration
    lora: LoRAConfig = None
    
    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()

class TrainingConfigManager:
    """학습 종류별 설정을 로딩/캐싱하는 관리자."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """YAML 구성 파일을 한 번만 읽어 캐시한다."""
        if self._config is None:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config
    
    def get_domain_training_config(self) -> TrainingConfig:
        """도메인별 LoRA 학습 설정을 반환한다."""
        config = self.load_config()
        
        # Extract training config
        training_config = config.get('training', {})
        lora_config = config.get('lora', {})
        
        return TrainingConfig(
            epochs=training_config.get('epochs', 3),
            batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
            learning_rate=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            gradient_checkpointing=training_config.get('gradient_checkpointing', False),
            eval_steps=training_config.get('eval_steps', 500),
            logging_steps=training_config.get('logging_steps', 10),
            lora=LoRAConfig(
                r=lora_config.get('r', 64),
                alpha=lora_config.get('alpha', 128),
                dropout=lora_config.get('dropout', 0.1),
                target_modules=lora_config.get('target_modules', ["gate_proj", "up_proj", "down_proj"]),
            )
        )
    
    def get_router_training_config(self) -> TrainingConfig:
        """라우터 학습에 최적화된 설정을 반환한다."""
        config = self.load_config()
        
        # Router training typically uses different settings
        training_config = config.get('training', {})
        
        return TrainingConfig(
            epochs=training_config.get('router_epochs', 3),
            batch_size=training_config.get('router_batch_size', 2),
            gradient_accumulation_steps=training_config.get('router_gradient_accumulation_steps', 8),
            learning_rate=training_config.get('router_learning_rate', 5e-5),  # Lower LR for router
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),  # Enable for router
            eval_steps=training_config.get('eval_steps', 500),
            logging_steps=training_config.get('logging_steps', 10),
            lora=None  # Router training doesn't use LoRA
        )

# Global training config manager
_training_config_manager = None

def get_training_config_manager() -> TrainingConfigManager:
    """Get global training config manager"""
    global _training_config_manager
    if _training_config_manager is None:
        _training_config_manager = TrainingConfigManager()
    return _training_config_manager

def get_training_config(training_type: str = "domain") -> TrainingConfig:
    """Get training configuration for specified type"""
    manager = get_training_config_manager()
    if training_type == "domain":
        return manager.get_domain_training_config()
    elif training_type == "router":
        return manager.get_router_training_config()
    else:
        raise ValueError(f"Unknown training type: {training_type}")
