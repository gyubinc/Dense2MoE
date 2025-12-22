#!/usr/bin/env python3
"""Unified MoE 설정 정의 - Llama/Qwen 등 다양한 모델 지원."""

import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Model Registry - 지원하는 모델 타입별 기본값 정의
MODEL_REGISTRY = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "num_layers": 28,
    },
    "qwen": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "num_layers": 36,
    },
}

def get_model_type_from_config(config: Dict[str, Any]) -> str:
    """설정에서 모델 타입 추출"""
    return config.get('model', {}).get('type', 'llama')

def get_wandb_project_name(model_type: str = None) -> str:
    """모델 타입에 따른 wandb 프로젝트명 반환"""
    if model_type is None:
        model_type = 'llama'
    return f"{model_type.title()}_MoE"

@dataclass
class MoEConfig:
    """MoE 라우터 학습과 관련된 하이퍼파라미터 모음."""
    num_epochs: int = 1
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    load_balancing_loss_weight: float = 0.01
    logging_steps: int = 1
    eval_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    max_samples: int = 5000
    eval_max_samples: int = 500
    domains: List[str] = None
    top_k: int = 1  # Number of experts to select per token
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["medical", "law", "math", "code"]
        if self.eval_batch_size is not None:
            self.eval_batch_size = int(self.eval_batch_size)

@dataclass
class SystemConfig:
    """프로젝트 전역 시스템 설정(시드, 출력 디렉터리 등)."""
    seed: int = 42
    output_dir: str = "domain_models"
    gradient_checkpointing: bool = False

@dataclass
class GPUConfig:
    """사용할 GPU 장치 및 가시화 설정."""
    cuda_visible_devices: str = "6"
    device: str = "cuda:0"

@dataclass
class ModelConfig:
    """기반 언어모델 관련 설정."""
    model_type: str = "llama"  # 모델 타입 (llama, qwen 등)
    name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_length: int = 1800
    trust_remote_code: bool = True

@dataclass
class DataConfig:
    """데이터 경로 및 샘플 제한 설정."""
    max_samples: int = 5000
    eval_max_samples: int = 500
    domains: List[str] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["medical", "law", "math", "code"]

class MoEConfigManager:
    """YAML 구성 파일을 읽어 dataclass 인스턴스로 변환하는 관리자."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Get project root directory (assuming this file is in config/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "config", "config.yaml")
        self.config_path = config_path
        self._config = None
        self._runtime_overrides = {}  # Runtime overrides for config values
    
    def load_config(self) -> Dict[str, Any]:
        """구성 파일을 읽어 캐싱한다."""
        if self._config is None:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config
    
    def set_runtime_overrides(self, overrides: Dict[str, Any]) -> None:
        """Set runtime overrides for config values."""
        self._runtime_overrides.update(overrides)
    
    def clear_runtime_overrides(self) -> None:
        """Clear all runtime overrides."""
        self._runtime_overrides.clear()
    
    def get_moe_config(self) -> MoEConfig:
        """MoE 관련 설정을 dataclass로 반환한다."""
        config = self.load_config()
        moe_config = config.get('moe', {})
        data_config = config.get('data', {})
        
        # Apply runtime overrides
        overrides = self._runtime_overrides
        
        return MoEConfig(
            num_epochs=overrides.get('num_epochs', moe_config.get('num_epochs', 1)),
            batch_size=overrides.get('batch_size', moe_config.get('batch_size', 1)),
            eval_batch_size=overrides.get(
                'eval_batch_size',
                moe_config.get('eval_batch_size')
            ),
            gradient_accumulation_steps=overrides.get('gradient_accumulation_steps', moe_config.get('gradient_accumulation_steps', 16)),
            learning_rate=overrides.get('learning_rate', moe_config.get('learning_rate', 1e-4)),
            weight_decay=overrides.get('weight_decay', moe_config.get('weight_decay', 0.01)),
            warmup_ratio=overrides.get('warmup_ratio', moe_config.get('warmup_ratio', 0.1)),
            load_balancing_loss_weight=overrides.get('load_balancing_loss_weight', moe_config.get('load_balancing_loss_weight', 0.01)),
            logging_steps=overrides.get('logging_steps', moe_config.get('logging_steps', 1)),
            eval_steps=overrides.get('eval_steps', moe_config.get('eval_steps', 100)),
            bf16=overrides.get('bf16', moe_config.get('bf16', True)),
            gradient_checkpointing=overrides.get('gradient_checkpointing', moe_config.get('gradient_checkpointing', False)),
            max_grad_norm=overrides.get('max_grad_norm', moe_config.get('max_grad_norm', 1.0)),
            max_samples=overrides.get('max_samples', data_config.get('max_samples', 5000)),
            eval_max_samples=overrides.get('eval_max_samples', data_config.get('eval_max_samples', 500)),
            domains=overrides.get('domains', data_config.get('domains', ["medical", "law", "math", "code"])),
            top_k=overrides.get('top_k', moe_config.get('top_k', 1))
        )
    
    def get_system_config(self) -> SystemConfig:
        """시스템 설정을 dataclass로 반환한다."""
        config = self.load_config()
        system_config = config.get('system', {})
        
        return SystemConfig(
            seed=system_config.get('seed', 42),
            output_dir=system_config.get('output_dir', "domain_models"),
            gradient_checkpointing=system_config.get('gradient_checkpointing', False)
        )
    
    def get_gpu_config(self) -> GPUConfig:
        """GPU 설정을 dataclass로 반환한다."""
        config = self.load_config()
        gpu_config = config.get('gpu', {})
        
        return GPUConfig(
            cuda_visible_devices=gpu_config.get('cuda_visible_devices', "6"),
            device=gpu_config.get('device', "cuda:0")
        )
    
    def get_model_config(self) -> ModelConfig:
        """모델 설정을 dataclass로 반환한다. model.type에 따라 name 자동 결정."""
        config = self.load_config()
        model_config = config.get('model', {})
        
        # 모델 타입 결정
        model_type = model_config.get('type', 'llama')
        
        # name이 null이면 registry에서 자동 결정
        name = model_config.get('name')
        if name is None:
            registry_entry = MODEL_REGISTRY.get(model_type, MODEL_REGISTRY['llama'])
            name = registry_entry['name']
        
        return ModelConfig(
            model_type=model_type,
            name=name,
            max_length=model_config.get('max_length', 2048),
            trust_remote_code=model_config.get('trust_remote_code', True)
        )
    
    def get_data_config(self) -> DataConfig:
        """데이터 설정을 dataclass로 반환한다."""
        config = self.load_config()
        data_config = config.get('data', {})
        
        return DataConfig(
            max_samples=data_config.get('max_samples', 5000),
            eval_max_samples=data_config.get('eval_max_samples', 500),
            domains=data_config.get('domains', ["medical", "law", "math", "code"])
        )
    
# Global MoE config manager
_moe_config_manager = None

def get_moe_config_manager() -> MoEConfigManager:
    """Get global MoE config manager"""
    global _moe_config_manager
    if _moe_config_manager is None:
        _moe_config_manager = MoEConfigManager()
    return _moe_config_manager

def get_moe_config() -> MoEConfig:
    """Get MoE configuration"""
    manager = get_moe_config_manager()
    return manager.get_moe_config()

def get_system_config() -> SystemConfig:
    """Get system configuration"""
    manager = get_moe_config_manager()
    return manager.get_system_config()

def get_gpu_config() -> GPUConfig:
    """Get GPU configuration"""
    manager = get_moe_config_manager()
    return manager.get_gpu_config()

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    manager = get_moe_config_manager()
    return manager.get_model_config()

def get_data_config() -> DataConfig:
    """Get data configuration"""
    manager = get_moe_config_manager()
    return manager.get_data_config()
