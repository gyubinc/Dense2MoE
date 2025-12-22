#!/usr/bin/env python3
"""Llama-MoE 프로젝트 전역에서 사용하는 유틸리티 모음."""

import os
import yaml
import logging
import torch
import gc
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import random
import numpy as np

# Import from unified config
from config.domains import domain_manager

def setup_logging(log_file: str = None, level: str = "INFO"):
    """공통 로깅 설정을 초기화한다."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

def print_gpu_memory_summary(stage: str = ""):
    """GPU 메모리 사용량을 요약해 출력한다. -> 메모리 체킹용 안하면 끄기"""
    if not torch.cuda.is_available():
        print(f"[GPU] {stage}: CUDA not available")
        return
    
    try:
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            free = total - reserved
            
            print(f"[GPU{i}] {stage}: mem: alloc={allocated:.2f} GiB, "
                  f"reserved={reserved:.2f} GiB, max_alloc={max_allocated:.2f} GiB, "
                  f"free={free:.2f} GiB/ total={total:.2f} GiB")
    except Exception as e:
        print(f"[GPU] {stage}: Error getting memory info - {e}")

def clear_gpu_memory():
    """GPU 캐시와 파이썬 GC를 명시적으로 정리한다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def validate_environment() -> bool:
    """학습에 필요한 환경(CUDA, GPU)을 검증한다."""
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Check GPU memory
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("❌ No GPU devices found")
            return False
        
        print(f"✅ Found {gpu_count} GPU device(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        return True
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def setup_cuda_environment():
    """설정 파일을 참조해 CUDA 환경 변수를 구성한다."""
    from config.moe import get_gpu_config
    
    gpu_config = get_gpu_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_config.cuda_visible_devices
    
    print(f"🎮 CUDA_VISIBLE_DEVICES set to: {gpu_config.cuda_visible_devices}")

def setup_random_seed(seed: int = None):
    """재현성을 위해 랜덤 시드를 설정한다."""
    if seed is None:
        from config.moe import get_system_config
        system_config = get_system_config()
        seed = system_config.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to: {seed}")

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 구성 파일을 로드한다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load config from {config_path}: {e}")
        raise

def check_data_availability(domains: List[str] = None) -> Dict[str, bool]:
    """도메인별 데이터 존재 여부를 반환한다."""
    if domains is None:
        domains = domain_manager.get_available_domains()
    
    availability = {}
    for domain in domains:
        try:
            domain_availability = domain_manager.check_data_availability(domain)
            availability[domain] = domain_availability[domain]
        except Exception as e:
            print(f"⚠️ Error checking {domain} data: {e}")
            availability[domain] = False
    
    return availability

def save_config_to_output_dir(config_path: str, output_dir: str, config_name: str = "config.yaml"):
    """사용한 설정 파일을 출력 디렉터리에 복사한다."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy config file to output directory
        destination_path = os.path.join(output_dir, config_name)
        shutil.copy2(config_path, destination_path)
        
        print(f"📝 Config file saved to: {destination_path}")
        return destination_path
    except Exception as e:
        print(f"❌ Failed to save config file: {e}")
        return None
