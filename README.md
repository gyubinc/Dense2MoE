# Dense2MoE: Dense-to-MoE Transformation Pipeline

Dense 모델을 Mixture-of-Experts (MoE) 모델로 변환하는 통합 파이프라인

## TL;DR

Dense LLM 모델 기반 도메인별 LoRA 어댑터를 훈련하고, 이를 **Layer-wise MoE 아키텍처**로 통합하는 End-to-End 파이프라인

### Main Features
- **Multi-Model 지원**: Llama, Qwen 등 다양한 Dense 모델 지원
- **Layer-wise MoE**: 각 layer마다 독립적인 Router
- **N개 Expert**: N개 도메인 전문가 + 1개 zero 전문가
- **유연한 학습**: Router만 학습, Router+MLP 학습, Attention 학습 등 다양한 옵션
- **메모리 효율성**: LoRA를 통한 파라미터 효율적 구조

## MoE Architecture

### **Layer Structure**
```
Layer N:
├── Self-Attention (freeze/trainable)
├── Router N (trainable) → N+1개 중 Top-K selection
└── Expert Selection
    ├── Domain 1 Expert (LoRA MLP)
    ├── Domain 2 Expert (LoRA MLP)
    ├── ...
    └── Zero Expert (원본 FFN)

Output = Σ(Router Weight × Expert Output)
```

### **Overall Architecture**
- **Base Model**: 설정 가능 (Llama, Qwen 등)
- **Experts per Layer**: N+1개 (N개 도메인 + 1개 zero)
- **Gating Strategy**: Top-1 (Hard) / Top-2+ (Soft) 선택 가능

## File Structure

```
Dense2MoE/
├── src/                      # 🏗️ 핵심 소스 코드
│   ├── models/               # 🤖 MoE 모델 구현체
│   │   └── model.py          # MoEModel, LayerRouter, ExpertFFN
│   ├── core/                 # 🔧 핵심 컴포넌트
│   │   ├── trainer.py        # 도메인/라우터 훈련기
│   │   ├── evaluator.py      # 라우터 평가기
│   │   └── dataset.py        # 데이터셋 처리
│   └── utils/                # 🛠️ 유틸리티
│       ├── utils.py          # GPU, 로깅, 환경설정
│       └── wandb_utils.py    # WandB 연동
├── config/                   # ⚙️ 설정 관리
│   ├── config.yaml           # 메인 설정 (model.type으로 모델 전환)
│   ├── moe.py                # MoE/모델 설정 (MODEL_REGISTRY)
│   └── domains.py            # 도메인 설정
├── scripts/                  # 🚀 실행 스크립트
│   ├── training/
│   │   ├── train_domain.py   # 도메인별 LoRA 학습
│   │   └── train_router.py   # MoE 라우터 학습
│   └── evaluation/
│       ├── evaluate.py       # MoE 평가
│       └── evaluate_domain.py# 도메인 LoRA 평가
├── data/                     # 📊 데이터
├── domain_models/            # 🎯 훈련된 LoRA 어댑터
├── moe_models/               # 🤖 학습된 라우터 체크포인트
└── requirements.txt          # 📦 의존성
```

## Setting

### Model Switching (config.yaml)
```yaml
model:
  type: "llama"  # "llama" 또는 "qwen"
  name: null     # null이면 type에서 자동 결정
```

### Supported Models (config/moe.py)
```python
MODEL_REGISTRY = {
    "llama": {"name": "meta-llama/Llama-3.2-3B-Instruct", "num_layers": 28},
    "qwen": {"name": "Qwen/Qwen3-4B-Instruct-2507", "num_layers": 36},
}
```

## Usage

### 1. Train Domain Lora Adapter

```bash
# 의료 도메인 훈련
python scripts/training/train_domain.py --domain medical --max-samples 1000

# 법률 도메인 훈련
python scripts/training/train_domain.py --domain law --max-samples 1000
```

**Main arguments:**
- `--domain`: 학습 도메인 (medical, law, math, code)
- `--max-samples`: 최대 샘플 수
- `--epochs`: 에폭 수
- `--output-dir`: 저장 경로

### 2. Train MoE Router

```bash
# 라우터 학습 (Router만)
python scripts/training/train_router.py --output-dir moe_models/run1 --target router

# 라우터 + MLP 학습
python scripts/training/train_router.py --output-dir moe_models/run1 --target router_mlp

# Attention 포함 학습
python scripts/training/train_router.py --output-dir moe_models/run1 --target attention
```

**주요 옵션:**
- `--target`: 학습 대상 (`router`, `mlp`, `attention`, `router_mlp`)
- `--top-k`: Expert 선택 수 (1: Hard, 2+: Soft routing)
- `--load-balancing-loss-weight`: 로드 밸런싱 손실 가중치
- `--use-wandb`: WandB 로깅

### 3. Evaluation

```bash
# MoE 모델 평가
python scripts/evaluation/evaluate.py \
    --moe-model-path /data/disk5/internship_disk/gyubin/MoE_models/Llama_model/final_router/router_epoch3_12600_aux_0_5e-4_top1/final_model/pytorch_model.bin \
    --domain medical --max-samples 200

# 도메인 LoRA 평가
python scripts/evaluation/evaluate_domain.py --domain medical --max-samples 200
```

## Domain Datasets

| 도메인 | 데이터셋 | 선택지 | Train | Test |
|--------|----------|--------|-------|------|
| Medical | MedMCQA | 4 | 20,000 | 1,000 |
| Law | casehold | 5 | 20,000 | 1,000 |
| Math | mathqa | 5 | 20,000 | 1,000 |
| Code | coding-mcq-reasoning | 4 | 3,000 | 300 |
| MMLU | MMLU | 4 | - | 1,000 |

## Pipeline

```bash
# 1. 환경 활성화
conda activate moe

# 2. 모든 도메인 LoRA 훈련
for domain in medical law math code; do
    python scripts/training/train_domain.py --domain $domain --epochs 2
done

# 3. MoE 라우터 훈련
python scripts/training/train_router.py --output-dir moe_models/run1 --epochs 1

# 4. 평가
python scripts/evaluation/evaluate.py \
    --moe-model-path /data/disk5/internship_disk/gyubin/MoE_models/Llama_model/final_router/router_epoch3_12600_aux_0_5e-4_top1/final_model/pytorch_model.bin \
    --domain medical
```

## Results directory

- `domain_models/<domain>/`: 도메인별 LoRA 어댑터
- `moe_models/<run>/final_model/`: 학습된 라우터
- `*_training_summary.json`: 학습 요약

---


# Model 위치

전체 모델 디렉토리: /data/disk5/internship_disk/gyubin/MoE_models
Llama best model
top-1
/data/disk5/internship_disk/gyubin/MoE_models/Llama_model/final_router/router_epoch3_12600_aux_0_5e-4_top1/final_model/pytorch_model.bin
top-2
/data/disk5/internship_disk/gyubin/MoE_models/Llama_model/final_router/router_epoch3_12600_aux_0_2e-4_top2/final_model/pytorch_model.bin


Qwen best model
top-1
/data/disk5/internship_disk/gyubin/MoE_models/Qwen_MoE/final_router/router_epoch3_12600_noaux_5e4_top1/final_model/pytorch_model.bin
top-2
/data/disk5/internship_disk/gyubin/MoE_models/Qwen_MoE/final_router/router_epoch3_12600_noaux_2e4_top2/final_model/pytorch_model.bin





# Experiment Results

# Qwen
# Dense model

- base model
    - 학습하지 않은 base model
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | qwen-base | 69.33 | 66.5 | 40.5 | 61.6 | 72.8 | 62.146 |
- domain model
    - 각 도메인별로 학습한 모델
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | **eval-adapter-law** | 63 | 89.6 | 40.5 | 61.3 | 70.3 | **64.94** |
        | **eval-adapter-math** | 69 | 68.9 | 66 | 64.1 | 73.4 | **68.28** |
        | **eval-adapter-medical** | 69.33 | 68.6 | 46.1 | 68.8 | 73.5 | **65.266** |
        | **eval-adapter-code** | 75 | 69.1 | 45.1 | 63.5 | 73.7 | **65.28** |
    - 20%만 사용한 모델
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | **eval-qwen-law-only** | 67 | 77.6 | 42.4 | 63.1 | 74.8 | **64.98** |
        | **eval-qwen-math-only** | 68 | 67.7 | 49 | 62 | 74.1 | **64.16** |
        | **eval-qwen-medical-only** | 70 | 68.9 | 44.6 | 64.8 | 75.6 | **64.78** |
        | **eval-qwen-code-only** | 70.33 | 66.8 | 41.8 | 61.5 | 72.9 | **62.666** |
    
- general model
    
    모든 데이터셋을 학습한 모델
    
    - mlp만 학습
        
        
        | name | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | **eval-adapter-general** | 76.33 | 87.6 | 63.1 | 69.2 | 74.4 | **74.126** |
    - mlp + attention도 학습
        
        
        | lr | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | 2e-4 | 74.33 | 88.0 | 65.8 | 69.7 | 74.5 | 74.466 |
    - 12,600개로 학습
        
        모든 데이터셋을 학습한 모델
        
        - mlp만 학습
            
            
            | name | code | law | math | medical | mmlu | average |
            | --- | --- | --- | --- | --- | --- | --- |
            | **eval-adapter-general** | 76.33 | 87.6 | 63.1 | 69.2 | 74.4 | **74.126** |
        - mlp + attention도 학습
            
            
            | lr | code | law | math | medical | mmlu | average |
            | --- | --- | --- | --- | --- | --- | --- |
            | 2e-4 | 74.33 | 80.9 | 57.4 | 66.1 | 74.5 | **70.646** |
            | **2e-5** | 75 | 81.2 | 59.1 | 66.3 | 75.7 | **71.46** |
            | **2e-6** | 76 | 82.2 | 58.5 | 67.4 | 74.9 | **71.8** |

# MoE model

- moe-base model
    
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **eval-moe-base** | 69 | 77 | 53.6 | 67 | 74.2 | **68.16** | 1 |
    | **eval-moe-base** | 72 | 77.2 | 53.2 | 65.8 | 75.8 | 68.8 | 2 |
- D2H model
    
    domain model를 결합한 형태
    
    - Router만 학습
        
        
        | domain | code | law | math | medical | mmlu | average | top-k |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | **final_router_top1** | 78 | 90.1 | 66.3 | 68.3 | 75 | **75.54** | 1 |
        | **final_router_top2** | 75.67 | 86.2 | 65.1 | 68.7 | 76 | **74.334** | 2 |
    - 각자 한번에 학습
    
    | **domain** | **code** | **law** | **math** | **medical** | **mmlu** | **average** | **top-k** |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **attention-noaux-5e5-top1** | 75.67 | 87.8 | 65.5 | 69.3 | 73.8 | **74.414** | 1 |
    | **mlp-attention-noaux-5e5-top1** | 72 | 88.4 | 60.8 | 64 | 73.3 | **71.7** | 1 |
    | **mlp-noaux-5e5-top1** | 75 | 88.1 | 63.8 | 66.8 | 73.4 | **73.42** | 1 |
    |  |  |  |  |  |  |  |  |
    | **attention-0-5e5-top2** | **76** | **87.1** | **63** | **68.1** | **73.6** | **73.56** | 2 |
    | **attention+mlp-noaux-5e5-top2** | **72.67** | **86.5** | **60.4** | **63.3** | **72.4** | **71.054** | 2 |
    | **mlp-noaux-5e5-top2** | **74.67** | **86.1** | **60.6** | **65.6** | **74.4** | **72.274** | 2 |
    - router를 먼저 학습한 후 결합
        
        
        | domain | code | law | math | medical | mmlu | average | top-k |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | **trained_router_attention** | 78 | 90.1 | 66.3 | 68.2 | 74.5 | **75.42** | 1 |
        | **trained_router_mlp** | 78 | 90.2 | 66.4 | 68 | 74.7 | **75.46** | 1 |
        | **trained_router_attention+mlp** | 78 | 90.1 | 66.3 | 68.4 | 74.8 | **75.52** | 1 |
        | **trained_router_mlp** | 75.67 | 85.9 | 64.9 | 68.3 | 75.9 | **74.134** | 2 |
        | **trained_router_attention** | 76 | 86 | 64.7 | 68.7 | 75.8 | **74.24** | 2 |
        | **trained_router_attention+mlp** | 76 | 86.3 | 65.1 | 68.3 | 75.7 | **74.28** | 2 |
- zero adapter model
    - router+adapter만 학습
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **aux0_LR5e4** | 77.0 | 87.1 | 62.9 | 67.7 | 74.0 | 73.74 | 1 |
    | **aux0_LR5e4** | 76.67 | 88.5 | 66.9 | 69.9 | 73.4 | 75.074 | 2 |
    - router + adapter + attention 학습
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **noaux _lr5e5** | 77.0 | 84.4 | 61.5 | 66.9 | 75.2 | 73.0 | 1 |
    | **noaux_lr5e5** | 76.33 | 83.8 | 61.5 | 67.9 | 74.3 | 72.766 | 2 |



# Llama
# Dense model

- base model
    - 학습하지 않은 base model
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | eval-llama-base | 53 | 53.2 | 35.3 | 68.1 | 56.9 | 53.3 |
- domain model
    - 각 도메인별로 학습한 모델
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | code | 65.33 | 55.9 | 34.5 | 71 | 57.6 | 56.766 |
        | law | 47.33 | 88.2 | 33.9 | 62.2 | 51.3 | 56.586 |
        | math | 55.67 | 56.1 | 47.9 | 72.2 | 57.2 | 57.814 |
        | medical | 57.33 | 53.8 | 40 | 76.8 | 58 | 57.186 |
    - 20%만 사용한 모델
        
        
        | domain | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | **eval-code-only** | 59.67 | 55 | 35.8 | 72.7 | 56.9 | **56.014** |
        | **eval-math-only** | 56 | 55 | 38.8 | 72.9 | 57.4 | **56.02** |
        | **eval-medical-only** | 57.67 | 54.7 | 38.1 | 75.3 | 57.4 | **56.634** |
        | **eval-law-only** | 56 | 73.6 | 35.9 | 71.6 | 58.2 | **59.06** |
    
- general model
    
    모든 데이터셋을 학습한 모델
    
    - mlp만 학습
        
        
        | name | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | **eval-adapter-general** | 62.67 | 79.9 | 44.5 | 76.3 | 57.7 | **64.214** |
    - mlp + attention도 학습
        
        
        | lr | code | law | math | medical | mmlu | average |
        | --- | --- | --- | --- | --- | --- | --- |
        | 2e4 | 66.0 | 85.8 | 45.8 | 75.4 | 57.6 | **66.12** |

# MoE model

- moe-base model
    
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **eval-moe-base** | 58.33 | 73.6 | 37.1 | 73.2 | 57.7 | **59.986** | 1 |
    | **eval-moe-base** | 62.33 | 74.8 | 38.8 | 75 | 58.9 | **61.966** | 2 |
- D2H model
    
    domain model를 결합한 형태
    
    - Router만 학습
        
        
        | domain | code | law | math | medical | mmlu | average | top-k |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | **eval-epoch1_12600_noaux_5e4** | 63.33 | 87.6 | 46.3 | 72.4 | 57.9 | **65.506** | 1 |
        | **eval-epoch1_12600_noaux_2e4** | 65 | 85.4 | 46 | 73.4 | 58 | **65.56** | 2 |
    - 각자 한번에 학습
    
    | **domain** | **average** | **code** | **law** | **math** | **medical** | **mmlu** | **top-k** |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **eval-attention-aux-0-lr-5e5-top1** | **64.66** | 65 | 86.6 | 45.7 | 68.3 | 57.7 | 1 |
    | **eval-attention+mlp-aux-0-lr-5e5-top1** | **65.18** | 62 | 87.3 | 46.2 | 72.4 | 58 | 1 |
    | **eval-mlp-aux-0-lr-5e5-top1** | **65.4** | 63 | 87.6 | 46.5 | 72.5 | 57.4 | 1 |
    | **eval-attention-aux-0-lr-5e5-top2** | **65.58** | 65 | 85.5 | 45.6 | 74 | 57.8 | 2 |
    | **eval-attention+mlp-aux-0-lr-5e5-top2** | **63** | 60 | 85.3 | 44.6 | 66.9 | 58.2 | 2 |
    | **eval-mlp-aux-0-lr-5e5-top2** | **65.506** | 65.33 | 85.6 | 45.1 | 73.5 | 58 | 2 |
    
    - router를 먼저 학습한 후 결합
        
        
        | domain | average | code | law | math | medical | mmlu | top-k |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | **eval-trained-attention-aux-noaux-lr-2e5-top1** | **65.134** | 61.67 | 87.6 | 45.8 | 72.6 | 58 | 1 |
        | **eval-trained-attention+mlp-aux-noaux-lr-2e5-top1** | **65.186** | 63.33 | 88 | 44.8 | 72.5 | 57.3 | 1 |
        | **eval-trained-mlp-aux-noaux-lr-2e5-top1** | **65.306** | 63.33 | 87.9 | 45.3 | 72.3 | 57.7 | 1 |
        | **eval-trained-attention-aux-noaux-lr-2e5-top2** | **65.354** | 63.67 | 86.5 | 46.1 | 72.7 | 57.8 | 2 |
        | **eval-trained-attention+mlp-aux-noaux-lr-2e5-top2** | **65.374** | 63.67 | 87 | 45.9 | 72.2 | 58.1 | 2 |
        | **eval-trained-mlp-aux-noaux-lr-2e5-top2** | **65.074** | 63.67 | 86.5 | 45.3 | 72.3 | 57.6 | 2 |
- zero adapter model
    
    여기를 추가로 실험해햐 할듯
    
    - router+adapter만 학습 (0,1,2,6)으로 쓰자(5e4)
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **zero-start-router_mlp** | 56 | 51.7 | 35.4 | 70.9 | 57 | **54.2** | 1 |
    | **zero-start-router_mlp_attention** | 61 | 81.6 | 42.5 | 71.3 | 57 | **62.68** | 2 |
    - router + adapter + attention 학습(5e5)
    
    | domain | code | law | math | medical | mmlu | average | top-k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | **zero-start-router_mlp** | 56 | 51.7 | 35.4 | 70.9 | 57 | **54.2** | 1 |
    | **zero-start-router_mlp_attention** | 64.33 | 82.7 | 43.8 | 72.5 | 57.7 | **64.206** | 2 |



**Author**: Gyubin Choi    