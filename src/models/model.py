#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Llama-MoE 모델 구성 요소를 정의한다.

핵심 개념
---------
- *동결된 기본 MLP + 학습 가능한 라우터* 조합으로 동작한다.
- 각 레이어는 `[code, medical, law, math]` 도메인 LoRA 전문가와 ZeroExpert를 포함한다.
- 전문가 모듈은 베이스 FFN 출력에 **LoRA 델타**만 더해 추가 Linear 레이어 생성을 피한다.
- LoRA 로딩 실패 혹은 형상 불일치 시 즉시 예외를 발생시켜 안정성을 보장한다.
- Forward 시 `outputs.router_stats`로 라우팅 통계를 수집해 분석에 활용한다.

주요 구현 포인트
-----------------
1. `ExpertFFN`은 저랭크 A/B 가중치와 스케일만을 관리하며, 베이스 FFN 출력(`original_mlp`)과 결합해 델타를 산출한다.
2. LoRA 파라미터는 bf16 nn.Parameter로 등록돼 장치 이동 시 자동 추적되며 dtype 캐스팅 오버헤드를 최소화한다.
3. 하이브리드 로딩: state_dict에 LoRA가 있으면 사용, 없으면 domain_models/의 초기 LoRA 유지 (기존 모델 호환).
4. target="lora" 설정 시 LoRA 파라미터도 학습 가능.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from config.moe import get_model_config, get_gpu_config, get_moe_config

logger = logging.getLogger(__name__)

# 환경 설정(경고 억제 최소화)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ------------------------------
# Router (fp32 softmax)
# ------------------------------
class LayerRouter(nn.Module):
    """토큰별 전문가 선택을 담당하는 라우터 모듈."""
    def __init__(self, hidden_size: int, num_adapters: int, dropout: float = 0.1):
        super().__init__()
        self.num_adapters = int(num_adapters)
        mid = max(1, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, self.num_adapters),
        )
        top_k = get_moe_config().top_k
        self.top_k = min(top_k, self.num_adapters)
        
        # Force all parameters to BFloat16
        for param in self.parameters():
            param.data = param.data.to(torch.bfloat16)

    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """소프트/하드 라우팅을 수행하고 부가 손실을 반환한다.

        Args:
            x: `[batch, seq, hidden]` 형태의 입력 텐서.
            train: 학습 모드 여부. 학습 시에는 소프트 라우팅(STRAIGHT-THROUGH), 추론 시에는 하드 라우팅.

        Returns:
            router_mask: `[batch, seq, num_experts]` 토큰별 전문가 선택 확률/원핫 텐서.
            aux_loss: Switch Transformer 스타일의 부하 균형 손실.
        """
        logits = self.net(x)  # [B, S, E]
        probs = F.softmax(logits.to(torch.float32), dim=-1)  # [B, S, E]
        probs = probs.to(x.dtype)  # Use input tensor dtype instead of logits dtype
        
        if train:
            topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
            soft_topk = torch.zeros_like(probs)
            soft_topk.scatter_(-1, topk_idx, topk_vals)
            denom = soft_topk.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            soft_mask = soft_topk / denom
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            if self.top_k == 1:
                # router_mask = hard_mask + soft_mask - soft_mask.detach()
                router_mask = hard_mask + (probs - probs.detach())
                aux_loss = self._compute_load_balancing_loss(probs, hard_mask)
            else:
                router_mask = soft_mask
                aux_loss = self._compute_load_balancing_loss(soft_mask, hard_mask)
            # aux_loss = self._compute_load_balancing_loss(soft_mask, hard_mask)
        else:
            topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
            soft_topk = torch.zeros_like(probs)
            soft_topk.scatter_(-1, topk_idx, topk_vals)
            denom = soft_topk.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            soft_mask = soft_topk / denom
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            if self.top_k == 1:
                router_mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                router_mask = soft_mask
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        
        return router_mask, aux_loss
    
    def _compute_load_balancing_loss(self, probs: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        """Switch Transformer 스타일 부하 균형 손실을 계산한다."""
        """이 과정에서 self.topk가 2이상일 경우에는 glam-style로 진행함"""
        expert_usage = (hard_mask > 0).to(probs.dtype).mean(dim=(0, 1))  # [E]
        router_prob_mass = probs.mean(dim=(0, 1))  # [E]
        aux_loss = self.num_adapters * torch.sum(expert_usage * router_prob_mass)
        return aux_loss.to(probs.dtype)


# ------------------------------
# Expert FFN (LoRA 델타만 계산)
# ------------------------------
class ExpertFFN(nn.Module):
    """LoRA 델타만 계산하는 전문가 모듈.

    - 베이스 FFN Linear를 새로 생성하지 않고 기존 모듈을 참조한다.
    - 레이어별 LoRA 키를 탐색해 해당 레이어의 A/B/scale만 로딩한다.
    - bf16 nn.Parameter로 등록해 장치와 dtype 변경을 자동으로 추적한다.
    - 기본 requires_grad=False, target="lora" 시 학습 가능.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, domain: str, adapter_path: str, layer_idx: Optional[int] = None, original_mlp: Optional[nn.Module] = None):
        super().__init__()
        self.domain = domain
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.layer_idx = layer_idx  # <-- 레이어 인덱스 보관

        # 로드 & 검증
        self._load_lora_or_fail(adapter_path)
        
        # 원본 MLP 가중치 저장
        if original_mlp is not None:
            self._store_original_weights(original_mlp)

        # 학습 파라미터 없음(라우터만 학습). 안전상 requires_grad False 보장
        for p in self.parameters():
            p.requires_grad = False

    # --- 내부 유틸: 키 선택 ---
    def _find_layer_keys(self, state: Dict[str, torch.Tensor], layer_idx: int, name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        대상 레이어(layer_idx)와 모듈명(name: gate_proj/up_proj/down_proj)에 대해
        lora_A, lora_B, scaling/alpha/lora_alpha 키를 탐색하여 키 문자열을 반환.
        (없을 수 있으므로 Optional)
        """
        # 반드시 모두 포함되어야 할 부분 문자열
        must_have = [f".layers.{layer_idx}.", f".mlp.{name}."]
        # 후보 접미 형태
        sfx_A = ".lora_A.weight"
        sfx_B = ".lora_B.weight"
        sfx_scaling = ".scaling"
        sfx_alpha = ".alpha"
        sfx_lora_alpha = ".lora_alpha"

        def pick_one(suffix: str) -> Optional[str]:
            cands = [k for k in state.keys() if all(x in k for x in must_have) and k.endswith(suffix)]
            if not cands:
                return None
            # 가장 구체적인(접두가 긴) 키를 선택
            cands.sort(key=len, reverse=True)
            return cands[0]

        kA = pick_one(sfx_A)
        kB = pick_one(sfx_B)
        kScaling = pick_one(sfx_scaling)
        kAlpha = pick_one(sfx_alpha)
        kLoraAlpha = pick_one(sfx_lora_alpha)
        return kA, kB, kScaling, kAlpha, kLoraAlpha

    def _extract_triplet_for_layer(self, state: Dict[str, torch.Tensor], layer_idx: int, name: str) -> Dict[str, torch.Tensor]:
        """
        레이어별 triplet(A,B,scale)을 추출. scale은 scaling 우선, 없으면 alpha/r, 둘 다 없으면 1.0.
        실패 시 RuntimeError.
        """
        kA, kB, kScaling, kAlpha, kLoraAlpha = self._find_layer_keys(state, layer_idx, name)

        if kA is None or kB is None:
            raise RuntimeError(f"[LoRA] missing lora_A/lora_B for layer={layer_idx} {name}")

        A = state[kA]
        B = state[kB]

        # scale 결정: scaling > alpha/r > lora_alpha/r > 1.0
        scale = None
        if kScaling is not None:
            scale = state[kScaling]
        else:
            alpha = None
            if kAlpha is not None:
                alpha = state[kAlpha]
            elif kLoraAlpha is not None:
                alpha = state[kLoraAlpha]
            if alpha is not None:
                # r은 A.shape[0] 또는 B.shape[1] 로 해석 가능
                r = A.shape[0]
                # alpha가 스칼라 텐서/정수일 수 있음
                alpha_val = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
                # Use the same dtype as A tensor for consistency
                scale = torch.tensor(alpha_val / max(1, r), dtype=A.dtype)

        if scale is None:
            scale = torch.tensor(2.0, dtype=A.dtype)  # lora_alpha=64, r=32 → scale=2.0

        return {"A": A, "B": B, "scale": scale}

    def _extract_triplet_global(self, state: Dict[str, torch.Tensor], name: str) -> Dict[str, torch.Tensor]:
        """
        레이어 정보가 없는(구형) 어댑터용 전역 키 추출. 기존 코드와 최대한 동일한 기준을 사용.
        """
        A = B = scale = None
        for k, v in state.items():
            if name not in k:
                continue
            if "lora_A" in k:
                A = v
            elif "lora_B" in k:
                B = v
            elif "scaling" in k:
                scale = v
            elif k.endswith(".alpha") or k.endswith(".lora_alpha"):
                # alpha는 scale 후보로 보관 후 A/B 랭크로 보정
                alpha = float(v.item() if isinstance(v, torch.Tensor) else v)
                # A/B가 아직 없을 수 있으므로 scale은 잠정 처리. 최종 반환 시 보정
                scale = ("_ALPHA_", alpha)
        if A is None or B is None:
            raise RuntimeError(f"[LoRA] missing global A/B for {self.domain}:{name}")

        # scale 처리
        if isinstance(scale, tuple) and scale[0] == "_ALPHA_":
            alpha_val = scale[1]
            r = A.shape[0]
            scale = torch.tensor(alpha_val / max(1, r), dtype=A.dtype)
        if scale is None:
            scale = torch.tensor(2.0, dtype=A.dtype)  # lora_alpha=64, r=32 → scale=2.0

        return {"A": A, "B": B, "scale": scale}

    def _load_state_from_file(self, adapter_path: str) -> Dict[str, torch.Tensor]:
        if not os.path.isdir(adapter_path):
            raise RuntimeError(f"[LoRA] adapter dir not found: {adapter_path}")

        model_file = None
        for fn in ("adapter_model.safetensors", "adapter_model.bin"):
            p = os.path.join(adapter_path, fn)
            if os.path.exists(p):
                model_file = p
                break
        if model_file is None:
            raise RuntimeError(f"[LoRA] adapter weights not found in: {adapter_path}")

        if model_file.endswith(".safetensors"):
            from safetensors import safe_open
            state = {}
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
        else:
            state = torch.load(model_file, map_location="cpu")
        return state

    def _assert_shapes_or_fail(self, lw: Dict[str, torch.Tensor], name: str):
        A, B = lw["A"], lw["B"]
        if name in ("gate_proj", "up_proj"):
            exp_A = (None, self.hidden_size)
            exp_B = (self.intermediate_size, None)
        elif name == "down_proj":
            exp_A = (None, self.intermediate_size)
            exp_B = (self.hidden_size, None)
        else:
            raise RuntimeError(f"[LoRA] unknown layer name: {name}")

        ok = (
            A.dim() == 2 and B.dim() == 2 and
            A.shape[1] == exp_A[1] and
            B.shape[0] == exp_B[0] and
            A.shape[0] == B.shape[1]
        )
        if not ok:
            raise RuntimeError(
                f"[LoRA] shape mismatch in {name}: A{tuple(A.shape)} B{tuple(B.shape)} "
                f"(expect hidden={self.hidden_size}, inter={self.intermediate_size}, rank free)"
            )

    def _register_triplet_params(self, triplet: Dict[str, torch.Tensor], prefix: str):
        """LoRA triplet을 nn.Parameter로 등록한다 (학습 가능하도록 변경).
        
        기본적으로 requires_grad=False로 설정하며, 
        MoEModel._enable_router_training에서 필요시 활성화한다.
        """
        def to_bf16_param(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, torch.Tensor) and t.dim() > 0:
                return t.clone().detach().to(torch.bfloat16)
            return torch.tensor(float(t), dtype=torch.bfloat16) if not isinstance(t, torch.Tensor) else t

        # Force all tensors to BFloat16 with explicit conversion
        A = to_bf16_param(triplet["A"]).contiguous()
        B = to_bf16_param(triplet["B"]).contiguous()
        s = triplet.get("scale", torch.tensor(2.0, dtype=torch.bfloat16))
        s = s.clone().detach().to(torch.bfloat16).contiguous() if isinstance(s, torch.Tensor) else torch.tensor(float(s), dtype=torch.bfloat16)
        
        # Scale이 0차원일 경우 1차원으로 변환 (nn.Parameter 호환성)
        if s.dim() == 0:
            s = s.unsqueeze(0)

        # Verify dtype before registration
        assert A.dtype == torch.bfloat16, f"A tensor dtype is {A.dtype}, expected bfloat16"
        assert B.dtype == torch.bfloat16, f"B tensor dtype is {B.dtype}, expected bfloat16"
        assert s.dtype == torch.bfloat16, f"Scale tensor dtype is {s.dtype}, expected bfloat16"

        # Register as nn.Parameter (requires_grad=False by default, enabled via _enable_router_training)
        setattr(self, f"{prefix}A", nn.Parameter(A, requires_grad=False))
        setattr(self, f"{prefix}B", nn.Parameter(B, requires_grad=False))
        setattr(self, f"{prefix}Scale", nn.Parameter(s, requires_grad=False))

    def _load_lora_or_fail(self, adapter_path: str) -> None:
        state = self._load_state_from_file(adapter_path)

        def load_for(name: str) -> Dict[str, torch.Tensor]:
            # 우선 레이어별 키 시도
            if self.layer_idx is not None:
                try:
                    return self._extract_triplet_for_layer(state, self.layer_idx, name)
                except RuntimeError as e:
                    logger.error(f"[LoRA] Failed to extract layer-specific triplet for {name} (layer_idx={self.layer_idx}): {e}")
                    logger.error("[LoRA] This indicates a critical LoRA loading failure. Training cannot proceed safely.")
                    raise RuntimeError(f"[LoRA] Critical failure: Cannot load layer-specific LoRA weights for {name}") from e

            # 전역(구형) 키 폴백
            triplet = self._extract_triplet_global(state, name)
            import warnings
            warnings.warn(f"[LoRA][compat] No per-layer keys for {name} (layer_idx={self.layer_idx}); using global adapter slices.")
            return triplet

        gate = load_for("gate_proj")
        up   = load_for("up_proj")
        down = load_for("down_proj")

        # 형상 검증
        self._assert_shapes_or_fail(gate, "gate_proj")
        self._assert_shapes_or_fail(up,   "up_proj")
        self._assert_shapes_or_fail(down, "down_proj")

        # bf16 Parameter로 등록 (기본 requires_grad=False, 필요시 활성화)
        self._register_triplet_params(gate, "gate")
        self._register_triplet_params(up,   "up")
        self._register_triplet_params(down, "down")

    def _store_original_weights(self, original_mlp: nn.Module):
        """원본 MLP 가중치를 참조만 저장해 메모리를 절약한다."""
        # 원본 MLP의 가중치를 직접 참조 (복사하지 않음)
        self.original_gate_proj = original_mlp.gate_proj
        self.original_up_proj = original_mlp.up_proj
        self.original_down_proj = original_mlp.down_proj

    # --- 내부: LoRA 델타 계산 ---
    @staticmethod
    def _lora_delta(x: torch.Tensor, A: torch.Tensor, B_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # 불필요한 텐서 생성을 피하고 기존 버퍼를 활용해 메모리 누수를 최소화한다.
        # 모든 텐서는 동일한 장치/ dtype에 존재해야 한다.
        if A.device != x.device or B_tensor.device != x.device or scale.device != x.device:
            raise RuntimeError(f"Device mismatch: x={x.device}, A={A.device}, B={B_tensor.device}, scale={scale.device}")
        
        # Scale이 1차원인 경우 스칼라로 변환
        scale_val = scale.squeeze() if scale.dim() > 0 else scale
        
        # LoRA 델타는 `(x @ A^T) @ B^T * scale`로 계산된다.
        # 배치 차원이 존재하면 `torch.bmm`이 아닌 reshape+행렬 곱을 사용해 가독성을 유지한다.
        if x.dim() == 3:  # [B, S, H]
            # Reshape for batch matrix multiplication
            batch_size, seq_len, hidden_size = x.shape
            x_flat = x.view(-1, hidden_size)  # [B*S, H]
            result = (x_flat @ A.transpose(-2, -1)) @ B_tensor.transpose(-2, -1) * scale_val
            return result.view(batch_size, seq_len, -1)  # [B, S, H]
        else:  # [S, H] or [H]
            return (x @ A.transpose(-2, -1)) @ B_tensor.transpose(-2, -1) * scale_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All parameters are already BFloat16, just ensure they're on the same device
        input_device = x.device
        
        # 입력 장치가 변경되면 파라미터를 한 번만 이동해 메모리 재할당을 방지한다.
        if not hasattr(self, '_last_device') or self._last_device != input_device:
            self._last_device = input_device
            
            # Move parameters only if necessary
            params_to_move = [
                (self.gateA, self.gateB, self.gateScale),
                (self.upA, self.upB, self.upScale),
                (self.downA, self.downB, self.downScale)
            ]
            
            for param_group in params_to_move:
                if param_group[0].device != input_device:
                    # 파라미터의 data를 제자리 이동
                    for param in param_group:
                        param.data = param.data.to(device=input_device, non_blocking=True)
            
            # Original weights are now referenced directly, no need to move
        
        # 도메인 전문가의 경우 원본 가중치를 사용해 전체 출력과 델타를 계산한다.
        if hasattr(self, 'original_gate_proj'):
            # 원본 프로젝션 연산을 위해 입력을 bf16으로 맞춘다.
            x_typed = x.to(torch.bfloat16)
            
            # LoRA 델타를 더한 전체 FFN 출력을 구성한다.
            gate_full = F.silu(self.original_gate_proj(x_typed) + self._lora_delta(x_typed, self.gateA, self.gateB, self.gateScale))
            up_full = self.original_up_proj(x_typed) + self._lora_delta(x_typed, self.upA, self.upB, self.upScale)
            intermediate_full = gate_full * up_full
            full_output = self.original_down_proj(intermediate_full) + self._lora_delta(intermediate_full, self.downA, self.downB, self.downScale)
            

            cached = getattr(self, '_cached_base_out', None)
            if cached is not None:
                base_output = cached.to(full_output.dtype).to(full_output.device)

            # 기준(델타 없는) FFN 출력을 재사용하거나 새로 계산한다.
            else:
                gate_base = F.silu(self.original_gate_proj(x_typed))
                up_base = self.original_up_proj(x_typed)
                intermediate_base = gate_base * up_base
                base_output = self.original_down_proj(intermediate_base)
            
            # 최종적으로 델타만 반환해 MoE 조합 시에만 델타가 더해지도록 한다.
            return full_output - base_output
        else:
            # Fallback to original implementation (for ZeroExpert or compatibility)
            gate = F.silu(self._lora_delta(x, self.gateA, self.gateB, self.gateScale))
            up   = self._lora_delta(x, self.upA, self.upB, self.upScale)
            return self._lora_delta(gate * up, self.downA, self.downB, self.downScale)

# ------------------------------
# Zero Expert (Zero-initialized 학습 가능한 LoRA)
# ------------------------------
class ZeroExpert(nn.Module):
    """Zero로 초기화된 학습 가능한 LoRA adapter 전문가.
    
    초기 상태에서는 0 델타를 반환하지만, 학습을 통해 유의미한 델타를 출력할 수 있다.
    """

    def __init__(
        self, 
        name: str = "zero",
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        original_mlp: Optional[nn.Module] = None,
        lora_r: int = 32,
        lora_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.domain = name
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.lora_r = lora_r
        self.lora_scale = lora_scale
        
        # original_mlp가 없으면 학습 불가 (기존 호환성 유지)
        if hidden_size is None or intermediate_size is None:
            self._trainable = False
            return
        
        self._trainable = True
        
        # 원본 MLP 참조 저장
        if original_mlp is not None:
            self.original_gate_proj = original_mlp.gate_proj
            self.original_up_proj = original_mlp.up_proj
            self.original_down_proj = original_mlp.down_proj
        
        # LoRA 파라미터 생성 (Kaiming 초기화)
        # A 행렬: Kaiming uniform 초기화 (gradient 흐름에 최적)
        # B 행렬: 0 초기화 (LoRA 논문 권장, 초기에 delta=0)
        
        # gate_proj: [hidden_size → intermediate_size]
        gateA_tensor = torch.empty(lora_r, hidden_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(gateA_tensor, a=math.sqrt(5))
        self.gateA = nn.Parameter(gateA_tensor.to(torch.bfloat16), requires_grad=False)
        self.gateB = nn.Parameter(torch.zeros(intermediate_size, lora_r, dtype=torch.bfloat16), requires_grad=False)
        self.gateScale = nn.Parameter(torch.tensor([lora_scale], dtype=torch.bfloat16), requires_grad=False)
        
        # up_proj: [hidden_size → intermediate_size]
        upA_tensor = torch.empty(lora_r, hidden_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(upA_tensor, a=math.sqrt(5))
        self.upA = nn.Parameter(upA_tensor.to(torch.bfloat16), requires_grad=False)
        self.upB = nn.Parameter(torch.zeros(intermediate_size, lora_r, dtype=torch.bfloat16), requires_grad=False)
        self.upScale = nn.Parameter(torch.tensor([lora_scale], dtype=torch.bfloat16), requires_grad=False)
        
        # down_proj: [intermediate_size → hidden_size]
        downA_tensor = torch.empty(lora_r, intermediate_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(downA_tensor, a=math.sqrt(5))
        self.downA = nn.Parameter(downA_tensor.to(torch.bfloat16), requires_grad=False)
        self.downB = nn.Parameter(torch.zeros(hidden_size, lora_r, dtype=torch.bfloat16), requires_grad=False)
        self.downScale = nn.Parameter(torch.tensor([lora_scale], dtype=torch.bfloat16), requires_grad=False)

    @staticmethod
    def _lora_delta(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """LoRA 델타를 계산한다: (x @ A^T) @ B^T * scale"""
        scale_val = scale.squeeze() if scale.dim() > 0 else scale
        
        if x.dim() == 3:  # [B, S, H]
            batch_size, seq_len, hidden_size = x.shape
            x_flat = x.view(-1, hidden_size)
            result = (x_flat @ A.transpose(-2, -1)) @ B.transpose(-2, -1) * scale_val
            return result.view(batch_size, seq_len, -1)
        else:
            return (x @ A.transpose(-2, -1)) @ B.transpose(-2, -1) * scale_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 학습 불가 모드면 0 반환 (기존 호환성)
        if not self._trainable:
            return torch.zeros_like(x)
        
        # 파라미터 장치 이동
        input_device = x.device
        if self.gateA.device != input_device:
            for param_name in ['gateA', 'gateB', 'gateScale', 'upA', 'upB', 'upScale', 'downA', 'downB', 'downScale']:
                param = getattr(self, param_name)
                param.data = param.data.to(device=input_device, non_blocking=True)
        
        x_typed = x.to(torch.bfloat16)
        
        # original_mlp가 있으면 full output - base output = delta 방식
        if hasattr(self, 'original_gate_proj'):
            # LoRA 델타를 더한 전체 FFN 출력
            gate_full = F.silu(self.original_gate_proj(x_typed) + self._lora_delta(x_typed, self.gateA, self.gateB, self.gateScale))
            up_full = self.original_up_proj(x_typed) + self._lora_delta(x_typed, self.upA, self.upB, self.upScale)
            intermediate_full = gate_full * up_full
            full_output = self.original_down_proj(intermediate_full) + self._lora_delta(intermediate_full, self.downA, self.downB, self.downScale)
            
            # 캐시된 base output 사용 또는 새로 계산
            cached = getattr(self, '_cached_base_out', None)
            if cached is not None:
                base_output = cached.to(full_output.dtype).to(full_output.device)
            else:
                gate_base = F.silu(self.original_gate_proj(x_typed))
                up_base = self.original_up_proj(x_typed)
                intermediate_base = gate_base * up_base
                base_output = self.original_down_proj(intermediate_base)
            
            # 델타만 반환
            return full_output - base_output
        else:
            # original_mlp 없으면 순수 LoRA 델타만 반환
            gate = F.silu(self._lora_delta(x_typed, self.gateA, self.gateB, self.gateScale))
            up = self._lora_delta(x_typed, self.upA, self.upB, self.upScale)
            return self._lora_delta(gate * up, self.downA, self.downB, self.downScale)


# ------------------------------
# MLP with Experts
# ------------------------------
class MLPWithExperts(nn.Module):
    """동결된 베이스 MLP에 전문가 델타를 더하는 모듈.

    `forward` 호출 시 다음 정보를 `self._last_stats`에 저장한다.
    - `usage_counts`: 전문가별 토큰 선택 횟수
    - `usage_frac`: 전문가별 토큰 선택 비율
    """
    def __init__(self, original_mlp: nn.Module, hidden_size: int, adapter_root: str, layer_idx: Optional[int] = None):
        super().__init__()
        self.original_mlp = original_mlp
        for p in self.original_mlp.parameters():
            p.requires_grad = False

        self.layer_idx = layer_idx  # <-- 보관




        # 이 부분에서 단순히 adapter를 부착한다고 끝나진 않고 
        """
        config/domains.py에 새 DomainConfig + 데이터 파일 준비.
        config/moe.py·config/config.yaml의 기본 도메인 목록 확장.
        통합 데이터(data/processed/<domain>, data/processed/total/*.json)에 새 도메인 샘플과 domain 필드 추가.
        src/models/model.py의 domains 리스트에 새 이름 추가, 해당 LoRA 어댑터 디렉터리 생성.
        라우터 통계용 expert_names 목록(src/core/evaluator.py, scripts/evaluation/generation_check.py) 갱신.
        """

        domains = ["zero", "zero", "zero", "zero", "zero"]
        # domains = ["code", "medical", "law", "math", "zero"]
        # domains = ["medical", 'zero', 'zero', 'zero', 'zero']
        # domains = ['math', 'zero', 'zero', 'zero', 'zero']
        inter = original_mlp.gate_proj.out_features
        self.expert_names: List[str] = []

        experts: List[nn.Module] = []
        for d in domains:
            if isinstance(d, str) and d.lower().startswith("zero"):
                # Zero-initialized 학습 가능한 LoRA expert 생성
                zero_expert = ZeroExpert(
                    name=d,
                    hidden_size=hidden_size,
                    intermediate_size=inter,
                    original_mlp=self.original_mlp,
                )
                experts.append(zero_expert)
                self.expert_names.append(zero_expert.domain)
                continue
            pth = os.path.join(adapter_root, d, "final_adapter")
            experts.append(ExpertFFN(hidden_size, inter, d, pth, layer_idx=self.layer_idx, original_mlp=self.original_mlp))
            self.expert_names.append(d)



        # 재현을 위한 코드 (동일 도메인 adapter를 5개 사용해 봤을 때의 성능이 단일 domain과 같은지 평가)

        # dummy_domain = 'code'
        # domains = [dummy_domain, dummy_domain, dummy_domain, dummy_domain, dummy_domain]
        # inter = original_mlp.gate_proj.out_features
        # experts: List[nn.Module] = []
        # for d in domains:
        #     pth = os.path.join(adapter_root, d, "final_adapter")
        #     # 레이어 인덱스와 원본 MLP를 ExpertFFN에 전달
        #     experts.append(ExpertFFN(hidden_size, inter, d, pth, layer_idx=self.layer_idx, original_mlp=self.original_mlp))



        self.experts = nn.ModuleList(experts)
        self.router  = LayerRouter(hidden_size, num_adapters=len(self.experts))
        
        


        # 라우터 통계와 손실 버퍼는 재사용해 메모리 할당을 피한다.
        self._last_stats: Dict[str, torch.Tensor] = {}
        self._last_aux_loss = None

    def get_expert_names(self) -> List[str]:
        """현재 레이어에 등록된 전문가 이름을 반환한다."""
        return list(self.expert_names)

    def forward(self, x: torch.Tensor, train: bool = None) -> torch.Tensor:
        """소프트/하드 라우팅을 수행해 최종 출력을 생성한다."""
        # 학습 모드 여부를 인자로 받되, 명시되지 않으면 모듈의 상태를 사용한다.
        if train is None:
            train = self.training
        
        # 이전에 할당한 보조 손실 텐서를 재사용해 메모리 증가를 방지한다.
        if self._last_aux_loss is None:
            self._last_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            self._last_aux_loss.zero_()
        
        # 1. 동결된 베이스 MLP 출력 계산.
        base = self.original_mlp(x.to(torch.bfloat16))  # [B, S, H]

        for _expert in self.experts:
            setattr(_expert, '_cached_base_out', base)

        # 2. 라우터로 토큰별 전문가 선택.
        router_mask, aux_loss = self.router(x, train=train)  # [B, S, E], scalar

        # 3. 전문가별 델타 출력 계산 (추론 시 최적화 적용)
        if not train:
            # 추론 최적화: 선택된 top-k expert만 계산
            B, S, E = router_mask.shape
            H = x.shape[-1]
            top_k = self.router.top_k
            
            # top-k expert 인덱스와 가중치 추출
            topk_weights, topk_indices = torch.topk(router_mask, k=top_k, dim=-1)  # [B, S, k]
            # 가중치 정규화
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            # 선택된 expert만 계산
            combined_expert_out = torch.zeros(B, S, H, device=x.device, dtype=x.dtype)
            
            # 어떤 expert가 선택되었는지 파악
            selected_experts = set(topk_indices.reshape(-1).tolist())
            
            for expert_idx in selected_experts:
                expert = self.experts[expert_idx]
                # 이 expert가 선택된 위치 마스크 (top-k 중 어디에서든 선택됨)
                mask = (topk_indices == expert_idx).any(dim=-1)  # [B, S]
                
                if mask.any():
                    expert_out = expert(x)  # [B, S, H]
                    
                    # 해당 expert의 가중치 추출 (선택된 위치에서만)
                    # topk_indices에서 expert_idx와 일치하는 위치의 가중치
                    weight_mask = (topk_indices == expert_idx).to(x.dtype)  # [B, S, k]
                    expert_weight = (topk_weights * weight_mask).sum(dim=-1)  # [B, S]
                    
                    combined_expert_out = combined_expert_out + expert_out * expert_weight.unsqueeze(-1)
        else:
            # 학습: 모든 expert 계산 (gradient flow 유지)
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))  # [B, S, H]
            expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, S, H, E]
            
            # 선택 확률(또는 원핫)로 전문가 출력을 가중합
            router_mask_exp = router_mask.unsqueeze(2)  # [B, S, 1, E]
            combined_expert_out = torch.sum(expert_outputs * router_mask_exp, dim=-1)  # [B, S, H]

        # 캐시를 제거해 이후 호출에서 누적되지 않도록 한다.
        for _expert in self.experts:
            if hasattr(_expert, '_cached_base_out'):
                delattr(_expert, '_cached_base_out')

        # 5. 베이스 출력에 델타를 더해 최종 출력 구성.
        final_output = base + combined_expert_out

        # 6. MoEModel에서 접근할 수 있도록 부가 손실 저장.
        self._last_aux_loss = aux_loss

        # 7. 전문가 사용 통계를 업데이트해 디버깅 시각화 등에 활용한다.
        with torch.no_grad():
            E = len(self.experts)
            topk_vals, topk_idx = torch.topk(router_mask, k=self.router.top_k, dim=-1)
            flat_idx = topk_idx.reshape(-1)
            flat_w = topk_vals.reshape(-1).to(x.dtype)
            counts = torch.zeros(E, device=x.device, dtype=x.dtype)
            counts.scatter_add_(0, flat_idx, flat_w)
            total = counts.sum().clamp_min(torch.tensor(1.0, device=x.device, dtype=x.dtype))
            frac = counts / total
            
            if "usage_counts" not in self._last_stats:
                self._last_stats["usage_counts"] = counts.clone().detach()
                self._last_stats["usage_frac"] = frac.clone().detach()
            else:
                self._last_stats["usage_counts"].copy_(counts)
                self._last_stats["usage_frac"].copy_(frac)
            
            del counts, total, frac, flat_idx, flat_w, topk_vals, topk_idx

        return final_output
    
    def clear_router_stats(self):
        """라우터 통계를 초기화해 메모리 누적을 방지한다."""
        with torch.no_grad():
            # Clear stats dictionary but keep the structure
            for key in list(self._last_stats.keys()):
                if isinstance(self._last_stats[key], torch.Tensor):
                    del self._last_stats[key]
            self._last_stats.clear()
            
            # Reset auxiliary loss
            if self._last_aux_loss is not None:
                del self._last_aux_loss
                self._last_aux_loss = None


# ------------------------------
# MoE Model (라우터만 학습)
# ------------------------------
class MoEModel(nn.Module):
    """라우터만 학습하는 LLaMA 기반 MoE 래퍼."""
    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_root: str = "domain_models",
        target: Optional[str] = "router",
    ):
        super().__init__()
        model_cfg = get_model_config()
        moe_cfg = get_moe_config()

        self.base_model_name = base_model_name or model_cfg.name
        self.target = self._normalize_target(target)
        
        # 설정 파일에서 부하 균형 계수를 가져와 하드코딩을 피한다.
        self.aux_loss_coef = moe_cfg.load_balancing_loss_weight


        # 라우터 통계와 손실을 재사용 가능한 버퍼에 저장해 메모리 사용을 안정화한다.
        self._last_router_stats = []
        self._last_router_loss = None

        # GPU 설정에 따라 단일 장치에 모델을 로드한다.
        gpu_cfg = get_gpu_config()
        target_device = gpu_cfg.device if hasattr(gpu_cfg, 'device') else "cuda:0"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=target_device,  # Use specific device instead of "auto"
            trust_remote_code=model_cfg.trust_remote_code,
        )
        self.base_model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=model_cfg.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.base_model, "generation_config", None) is not None:
            self.base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if not hasattr(self.base_model, "model") or not hasattr(self.base_model.model, "layers"):
            raise ValueError("Requires LLaMA-style model exposing `.model.layers`.")

        hidden = int(self.base_model.config.hidden_size)
        self.layer_expert_names: List[List[str]] = []
        for i, block in enumerate(self.base_model.model.layers):
            if not hasattr(block, "mlp"):
                raise ValueError(f"Layer {i} has no .mlp; unsupported architecture.")
            # 레이어 인덱스를 전달
            block.mlp = MLPWithExperts(block.mlp, hidden, adapter_root, layer_idx=i)
            if hasattr(block.mlp, "get_expert_names"):
                self.layer_expert_names.append(block.mlp.get_expert_names())
            else:
                self.layer_expert_names.append([])

        # 모델이 GPU에 로드된 이후 모든 구성 요소가 동일한 장치에 있도록 보정한다.
        self._ensure_lora_params_on_gpu()
        self._force_device_consistency()
        
        # 그라디언트 체크포인팅 활성화 여부를 추적한다.
        self._gradient_checkpointing_enabled = False

        # 기본적으로 모든 파라미터를 동결한다.
        self._freeze_all_parameters()
        
        # 라우터 파라미터만 학습하도록 설정한다.
        self._enable_router_training(self.target)

    def get_expert_names(self, layer_idx: Optional[int] = None) -> Union[List[str], List[List[str]]]:
        """
        레이어별 전문가 이름을 반환한다.

        Args:
            layer_idx: 지정 시 해당 레이어의 전문가 이름 리스트를 반환한다.

        Returns:
            layer_idx가 None이면 전체 레이어의 전문가 이름 목록(List[List[str]])을,
            아니면 특정 레이어의 이름 목록(List[str])을 반환한다.
        """
        if layer_idx is None:
            return [list(names) for names in self.layer_expert_names]
        if layer_idx < 0 or layer_idx >= len(self.layer_expert_names):
            raise IndexError(f"Invalid layer index: {layer_idx}")
        return list(self.layer_expert_names[layer_idx])

    def _freeze_all_parameters(self):
        """모델의 모든 파라미터를 동결한다."""
        for p in self.parameters():
            p.requires_grad = False

    def _normalize_target(self, target: Optional[str]) -> str:
        if isinstance(target, str):
            cleaned = target.strip().lower()
            if cleaned:
                return cleaned
        return "router"

    def _enable_router_training(self, target: Optional[str] = None):
        """라우터 및 지정된 컴포넌트의 학습을 활성화한다.
        
        Args:
            target: 학습 대상 지정. 조합 가능:
                - "router": 라우터만 학습 (기본값)
                - "lora": 라우터 + LoRA 파라미터 학습
                - "attention": 라우터 + Attention proj 학습
                - "mlp": 라우터 + 베이스 MLP 학습
                - "router+lora": 라우터 + LoRA 학습 (명시적)
        """
        resolved_target = self._normalize_target(target if target is not None else self.target)
        self.target = resolved_target

        for layer in self.base_model.model.layers:
            # 라우터는 항상 학습 가능하도록 설정한다.
            if "router" in resolved_target:
                if hasattr(layer.mlp, "router"):
                    for p in layer.mlp.router.parameters():
                        p.requires_grad = True

            # LoRA 파라미터 학습 활성화
            if "mlp" in resolved_target:
                if hasattr(layer.mlp, "experts"):
                    for expert in layer.mlp.experts:
                        if hasattr(expert, 'gateA'):  # ExpertFFN인 경우
                            for param_name in ['gateA', 'gateB', 'gateScale', 
                                              'upA', 'upB', 'upScale',
                                              'downA', 'downB', 'downScale']:
                                if hasattr(expert, param_name):
                                    param = getattr(expert, param_name)
                                    if isinstance(param, nn.Parameter):
                                        param.requires_grad = True

            if "attention" in resolved_target:
                for name, p in layer.self_attn.named_parameters():
                    p.requires_grad = "proj" in name  # proj만 학습

            # if "mlp" in resolved_target:
            #     if hasattr(layer.mlp, 'original_mlp'):
            #         for p in layer.mlp.original_mlp.parameters():
            #             p.requires_grad = True
            
    def setup_training_parameters(self, target: Optional[str] = None):
        """라우터 + 선택된 서브모듈만 학습 가능하도록 파라미터를 설정한다."""
        resolved_target = self._normalize_target(target if target is not None else self.target)
        self._freeze_all_parameters()
        self._enable_router_training(resolved_target)
        logger.info("✅ Trainable components: %s", resolved_target)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Parameter summary: {trainable_params:,}/{total_params:,} trainable")

    def _ensure_lora_params_on_gpu(self):
        """모델과 모든 LoRA Parameter가 동일한 장치에 있도록 강제한다."""
        model_device = next(self.parameters()).device
        
        for layer in self.base_model.model.layers:
            if hasattr(layer.mlp, 'experts'):
                for expert in layer.mlp.experts:
                    if hasattr(expert, 'gateA'):  # Check if it's an ExpertFFN
                        # Move all LoRA parameters to the same device as the model
                        for param_name in ['gateA', 'gateB', 'gateScale', 
                                          'upA', 'upB', 'upScale',
                                          'downA', 'downB', 'downScale']:
                            if hasattr(expert, param_name):
                                param = getattr(expert, param_name)
                                if param.device != model_device:
                                    # Parameter의 data를 직접 이동
                                    param.data = param.data.to(model_device)

    def _force_device_consistency(self):
        """모든 파라미터와 버퍼를 강제로 동일한 장치로 이동시킨다."""
        model_device = next(self.parameters()).device
        
        # Move all parameters to the same device
        for param in self.parameters():
            if param.device != model_device:
                param.data = param.data.to(model_device)
        
        # Move all buffers to the same device
        for buffer in self.buffers():
            if buffer.device != model_device:
                buffer.data = buffer.data.to(model_device)
        
        # Recursively move all submodule buffers
        for module in self.modules():
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None and buffer.device != model_device:
                    module._buffers[buffer_name] = buffer.to(model_device)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """모델 전체에 대해 그라디언트 체크포인팅을 활성화한다."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        
        self._gradient_checkpointing_enabled = True
        
        # Enable gradient checkpointing for the base model if it supports it
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
        # Enable gradient checkpointing for each layer's MLP
        for layer in self.base_model.model.layers:
            if hasattr(layer.mlp, 'gradient_checkpointing_enable'):
                layer.mlp.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """모델 전체의 그라디언트 체크포인팅을 비활성화한다."""
        self._gradient_checkpointing_enabled = False
        
        # Disable gradient checkpointing for the base model if it supports it
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
        
        # Disable gradient checkpointing for each layer's MLP
        for layer in self.base_model.model.layers:
            if hasattr(layer.mlp, 'gradient_checkpointing_disable'):
                layer.mlp.gradient_checkpointing_disable()
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_router_loss=False):
        """소프트/하드 라우팅을 적용하며 라우터 통계를 포함한 출력을 반환한다."""
        # Simplified approach: always return a proper CausalLMOutputWithPast object
        try:
        # 베이스 모델의 표준 forward를 호출한다.
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            # Extract logits and other components
            if isinstance(base_outputs, dict):
                logits = base_outputs['logits']
                past_key_values = base_outputs.get('past_key_values')
                hidden_states = base_outputs.get('hidden_states')
                attentions = base_outputs.get('attentions')
                base_loss = base_outputs.get('loss')
            else:
                logits = base_outputs.logits
                past_key_values = base_outputs.past_key_values
                hidden_states = base_outputs.hidden_states
                attentions = base_outputs.attentions
                base_loss = getattr(base_outputs, 'loss', None)
            
            # Ensure we have a proper main loss tensor
            if base_loss is not None and isinstance(base_loss, torch.Tensor):
                main_loss = base_loss
            else:
                device = logits.device
                main_loss = torch.tensor(0.0, device=device, requires_grad=True)

            total_aux = None
            for layer in self.base_model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "_last_aux_loss") and layer.mlp._last_aux_loss is not None:
                    aux = layer.mlp._last_aux_loss
                    total_aux = aux if total_aux is None else (total_aux + aux)

            if total_aux is not None:
                router_loss_tensor = total_aux.to(main_loss.dtype).to(main_loss.device)
            else:
                router_loss_tensor = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)

            total_loss_tensor = main_loss + self.aux_loss_coef * router_loss_tensor

            # Create proper CausalLMOutputWithPast object
            outputs = CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=attentions,
                loss=main_loss
            )
            
            # Set total_loss as well
            outputs.total_loss = total_loss_tensor
            outputs.router_loss = router_loss_tensor
            
            # 필요한 경우 레이어별 라우터 통계를 모아 후속 분석에 활용한다.
            if return_router_loss:
                router_stats = []
                for layer_idx, layer in enumerate(self.base_model.model.layers):
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "_last_stats"):
                        stats = layer.mlp._last_stats
                        if stats is not None:
                            router_stats.append({
                                "layer": layer_idx,
                                "usage_counts": stats["usage_counts"].detach(),
                                "usage_frac": stats["usage_frac"].detach()
                            })
                outputs.router_stats = router_stats
            
            # Reuse existing storage to prevent memory leaks
            self._last_router_stats.clear()
            if self._last_router_loss is None or self._last_router_loss.dtype != router_loss_tensor.dtype or self._last_router_loss.device != router_loss_tensor.device:
                if self._last_router_loss is not None:
                    del self._last_router_loss
                self._last_router_loss = router_loss_tensor.detach().clone()
            else:
                self._last_router_loss.copy_(router_loss_tensor.detach())
            
            # Validate outputs before returning
            if not isinstance(outputs.loss, torch.Tensor):
                logger.error(f"[CRITICAL] outputs.loss is not a tensor: {type(outputs.loss)}")
                raise RuntimeError(f"[CRITICAL] outputs.loss is not a tensor: {type(outputs.loss)}")
            
            return outputs
                
        except Exception as e:
            # Critical failure: Do not use fallback, force termination
            logger.error(f"[MoE] Critical failure in forward pass: {e}")
            logger.error("[MoE] This indicates a fundamental problem with the MoE model.")
            logger.error("[MoE] Training cannot proceed safely with fallback outputs.")
            logger.error("[MoE] Please check: device consistency, LoRA loading, router configuration")
            
            # Log detailed error information
            import traceback
            logger.error(f"[MoE] Full traceback:\n{traceback.format_exc()}")
            
            # Force termination - do not return fallback outputs
            raise RuntimeError(f"[MoE] Critical forward pass failure: {e}") from e
    
    def clear_all_router_stats(self):
        """Clear router statistics from all layers to prevent memory accumulation"""
        with torch.no_grad():
            # Clear model-level stats
            if hasattr(self, '_last_router_stats'):
                self._last_router_stats.clear()
            if hasattr(self, '_last_router_loss') and self._last_router_loss is not None:
                del self._last_router_loss
                self._last_router_loss = None
            
            # Clear layer-level stats
            for layer in self.base_model.model.layers:
                if hasattr(layer.mlp, 'clear_router_stats'):
                    layer.mlp.clear_router_stats()
                if hasattr(layer.mlp, '_last_aux_loss'):
                    layer.mlp._last_aux_loss = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """하이브리드 로딩: state_dict에 LoRA가 있으면 사용, 없으면 domain_models 유지.
        
        이 방식으로 기존 학습된 모델(LoRA 없는 state_dict)과 
        새로운 모델(LoRA 포함 state_dict) 모두 호환된다.
        """
        # LoRA 관련 키 탐지
        lora_key_patterns = ['gateA', 'gateB', 'gateScale', 'upA', 'upB', 'upScale', 'downA', 'downB', 'downScale']
        lora_keys = [k for k in state_dict.keys() 
                     if any(pattern in k for pattern in lora_key_patterns)]
        
        if lora_keys:
            logger.info(f"[MoEModel] LoRA weights detected in state_dict ({len(lora_keys)} keys). Using state_dict LoRA.")
            # state_dict에 LoRA가 있으면 그대로 로드 (덮어쓰기)
            missing, unexpected = [], []
            try:
                # strict=False로 로드하여 누락된 키를 허용
                result = super().load_state_dict(state_dict, strict=False)
                missing = result.missing_keys if hasattr(result, 'missing_keys') else []
                unexpected = result.unexpected_keys if hasattr(result, 'unexpected_keys') else []
            except Exception as e:
                logger.warning(f"[MoEModel] load_state_dict warning: {e}")
            
            if missing:
                logger.debug(f"[MoEModel] Missing keys (expected for non-LoRA params): {len(missing)}")
            if unexpected:
                logger.debug(f"[MoEModel] Unexpected keys: {len(unexpected)}")
        else:
            logger.info("[MoEModel] No LoRA weights in state_dict. Using domain_models/ LoRA (already loaded).")
            # state_dict에 LoRA가 없으면 LoRA 키를 제외하고 로드
            # (이미 domain_models에서 로드된 LoRA 유지)
            filtered_state = {}
            
            for k, v in state_dict.items():
                # LoRA 키가 아닌 경우만 포함
                if not any(pattern in k for pattern in lora_key_patterns):
                    filtered_state[k] = v
            
            try:
                super().load_state_dict(filtered_state, strict=False)
            except Exception as e:
                logger.warning(f"[MoEModel] load_state_dict warning: {e}")
        
        # 장치 일관성 보정
        self._ensure_lora_params_on_gpu()
        self._force_device_consistency()
        
        logger.info("[MoEModel] State dict loaded successfully.")

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # HF가 모르는 인자는 사전에 제거하여 에러 방지
        for k in ("return_router_stats", "router_stats", "collect_router_stats", "return_router_loss"):
            if k in kwargs:
                kwargs.pop(k, None)
        for k in ("temperature", "top_p", "top_k"):
            kwargs.pop(k, None)
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

