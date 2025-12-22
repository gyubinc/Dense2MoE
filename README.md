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

## 🔧 Setting

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
    --moe-model-path moe_models/run1/final_model/pytorch_model.bin \
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
    --moe-model-path moe_models/run1/final_model/pytorch_model.bin \
    --domain medical
```

## Results directory

- `domain_models/<domain>/`: 도메인별 LoRA 어댑터
- `moe_models/<run>/final_model/`: 학습된 라우터
- `*_training_summary.json`: 학습 요약

---

**Author**: Gyubin Choi
