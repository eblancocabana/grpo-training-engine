from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportUnknownMemberType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportRedeclaration=false
# pyright: reportDeprecated=false
# pyright: reportConstantRedefinition=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportImplicitOverride=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnreachable=false
# pyright: reportArgumentType=false

from typing import Optional, Tuple, cast, TYPE_CHECKING, Protocol, Callable, Sequence

import torch

class _QuantStateLike(Protocol):
    absmax: torch.Tensor
    code: torch.Tensor
    blocksize: int
    shape: Sequence[int]
    nested: bool
    offset: Optional[float]
    state2: Optional["_QuantStateLike"]


class _Linear4BitLike(Protocol):
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class _LoRALayerLike(Protocol):
    base_layer: _Linear4BitLike
    lora_A: torch.nn.Module
    lora_B: torch.nn.Module
    scaling: float
    lora_dropout: Optional[torch.nn.Module]


class _TritonKernel(Protocol):
    def __getitem__(self, grid: Tuple[int, ...]) -> Callable[..., None]:
        ...


class _LoRAContext(Protocol):
    saved_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    quant_state: _QuantStateLike
    scaling: float
    output_shape: Tuple[int, ...]
    has_bias: bool

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        ...



try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

if TYPE_CHECKING:
    import triton as triton  # noqa: F401
    import triton.language as tl  # noqa: F401


if TRITON_AVAILABLE:

    LORA_FORWARD_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_R": 16},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_R": 16},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "BLOCK_R": 16},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_R": 16},
            num_warps=8,
            num_stages=2,
        ),
    ]

    @triton.autotune(configs=LORA_FORWARD_CONFIGS, key=["M", "N", "K", "R"])
    @triton.jit
    def _lora_forward_kernel(  # noqa: PLR0913
        x_ptr,
        w_ptr,
        absmax_ptr,
        code_ptr,
        state2_absmax_ptr,
        state2_code_ptr,
        bias_ptr,
        lora_A_ptr,
        lora_B_ptr,
        out_ptr,
        stride_xm,
        stride_xk,
        stride_ar,
        stride_ak,
        stride_bn,
        stride_br,
        stride_om,
        stride_on,
        M,
        N,
        K,
        R,
        blocksize,
        state2_blocksize,
        offset,
        scaling,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        NESTED: tl.constexpr,
        LORA_DTYPE: tl.constexpr,
        LORA_DOT_DTYPE: tl.constexpr,
        X_DTYPE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        if HAS_BIAS:
            bias_ptr = bias_ptr

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.multiple_of(offs_m, BLOCK_M)
        tl.multiple_of(offs_n, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            tl.multiple_of(offs_k, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_block = tl.load(
                x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )

            element_idx = offs_n[None, :] * K + offs_k[:, None]
            mask_w = mask_n[None, :] & mask_k[:, None]
            packed_idx = element_idx // 2
            packed = tl.load(w_ptr + packed_idx, mask=mask_w, other=0)
            packed = tl.cast(packed, tl.int32)
            is_low = element_idx & 1
            is_low = tl.cast(is_low, tl.int32)

            low_val = packed & 0xF
            high_val = packed >> 4
            q = tl.where(is_low == 0, high_val, low_val)

            code_val = tl.load(code_ptr + q, mask=mask_w, other=0.0)
            code_val = tl.cast(code_val, tl.float32)

            block_idx = element_idx // blocksize
            if NESTED:
                qabsmax = tl.load(absmax_ptr + block_idx, mask=mask_w, other=0)
                qabsmax = tl.cast(qabsmax, tl.int32)
                state2_block_idx = block_idx // state2_blocksize
                state2_absmax = tl.load(
                    state2_absmax_ptr + state2_block_idx, mask=mask_w, other=0.0
                )
                state2_absmax = tl.cast(state2_absmax, tl.float32)
                state2_code_val = tl.load(
                    state2_code_ptr + qabsmax, mask=mask_w, other=0.0
                )
                state2_code_val = tl.cast(state2_code_val, tl.float32)
                absmax_val = state2_code_val * state2_absmax + offset
            else:
                absmax_val = tl.load(absmax_ptr + block_idx, mask=mask_w, other=0.0)
                absmax_val = tl.cast(absmax_val, tl.float32)

            w_block = code_val * absmax_val
            w_block = tl.where(mask_w, w_block, 0.0)
            w_block = tl.cast(w_block, LORA_DTYPE)
            x_block = tl.cast(x_block, LORA_DTYPE)
            acc += tl.dot(x_block, w_block, out_dtype=LORA_DOT_DTYPE)

        lora_out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        scale = tl.full((), scaling, LORA_DTYPE)
        lora_x_ptrs = x_ptr + offs_m[:, None] * stride_xm
        for r_start in range(0, R, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < R

            lora_acc = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K

                x_ptrs = lora_x_ptrs + offs_k[None, :] * stride_xk
                x_lora_block = tl.load(
                    x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
                )
                x_lora_block = tl.cast(x_lora_block, LORA_DTYPE)
                a_ptrs = (
                    lora_A_ptr
                    + offs_r[None, :] * stride_ar
                    + offs_k[:, None] * stride_ak
                )
                a_block = tl.load(
                    a_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0
                )
                a_block = tl.cast(a_block, LORA_DTYPE)
                lora_acc += tl.dot(x_lora_block, a_block, out_dtype=LORA_DOT_DTYPE)

            b_ptrs = lora_B_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
            b_block = tl.load(
                b_ptrs, mask=mask_n[None, :] & mask_r[:, None], other=0.0
            )
            b_block = tl.cast(b_block, LORA_DTYPE)
            lora_acc = tl.cast(lora_acc, LORA_DTYPE)
            lora_out += tl.dot(lora_acc, b_block, out_dtype=LORA_DOT_DTYPE)

        acc = tl.cast(acc, X_DTYPE)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_n
            bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
            bias = tl.cast(bias, X_DTYPE)
            acc = acc + bias[None, :]

        lora_out = tl.cast(lora_out, LORA_DTYPE)
        lora_out = lora_out * scale
        if acc.dtype != lora_out.dtype:
            lora_out = tl.cast(lora_out, acc.dtype)
        acc = acc + lora_out

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _lora_grad_x_base_kernel(
        grad_ptr,
        w_ptr,
        absmax_ptr,
        code_ptr,
        state2_absmax_ptr,
        state2_code_ptr,
        out_ptr,
        stride_gm,
        stride_gn,
        stride_om,
        stride_ok,
        M,
        N,
        K,
        blocksize,
        state2_blocksize,
        offset,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NESTED: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        tl.multiple_of(offs_m, BLOCK_M)
        tl.multiple_of(offs_k, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K

        acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)

        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            tl.multiple_of(offs_n, BLOCK_N)
            mask_n = offs_n < N

            grad_ptrs = grad_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
            grad_block = tl.load(
                grad_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )

            element_idx = offs_n[:, None] * K + offs_k[None, :]
            mask_w = mask_n[:, None] & mask_k[None, :]
            packed_idx = element_idx // 2
            packed = tl.load(w_ptr + packed_idx, mask=mask_w, other=0)
            packed = tl.cast(packed, tl.int32)
            is_low = element_idx & 1
            is_low = tl.cast(is_low, tl.int32)

            low_val = packed & 0xF
            high_val = packed >> 4
            q = tl.where(is_low == 0, high_val, low_val)

            code_val = tl.load(code_ptr + q, mask=mask_w, other=0.0)
            code_val = tl.cast(code_val, tl.float32)

            block_idx = element_idx // blocksize
            if NESTED:
                qabsmax = tl.load(absmax_ptr + block_idx, mask=mask_w, other=0)
                qabsmax = tl.cast(qabsmax, tl.int32)
                state2_block_idx = block_idx // state2_blocksize
                state2_absmax = tl.load(
                    state2_absmax_ptr + state2_block_idx, mask=mask_w, other=0.0
                )
                state2_absmax = tl.cast(state2_absmax, tl.float32)
                state2_code_val = tl.load(
                    state2_code_ptr + qabsmax, mask=mask_w, other=0.0
                )
                state2_code_val = tl.cast(state2_code_val, tl.float32)
                absmax_val = state2_code_val * state2_absmax + offset
            else:
                absmax_val = tl.load(absmax_ptr + block_idx, mask=mask_w, other=0.0)
                absmax_val = tl.cast(absmax_val, tl.float32)

            w_block = code_val * absmax_val
            w_block = tl.where(mask_w, w_block, 0.0)
            w_block = tl.cast(w_block, grad_block.dtype)

            acc += tl.dot(grad_block, w_block)

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

            a_block = tl.load(
                a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            b_block = tl.load(
                b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )

            acc += tl.dot(a_block, b_block)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
else:

    def _lora_forward_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for lora_fused_forward.")

    def _lora_grad_x_base_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for lora_fused_forward.")

    def _matmul_kernel(*_args, **_kwargs):
        raise ImportError("Triton is not available for lora_fused_forward.")


def _select_block_size(value: int, options: Tuple[int, ...]) -> int:
    for option in options:
        if value <= option:
            return option
    return options[-1]


def _select_matmul_config_forward(m: int, n: int, k: int) -> Tuple[int, int, int, int]:
    block_m = _select_block_size(m, (16, 32, 64))
    block_n = _select_block_size(n, (64, 128))
    block_k = _select_block_size(k, (32, 64))
    num_warps = 8 if block_m * block_n >= 4096 else 4
    return block_m, block_n, block_k, num_warps


def _select_matmul_config_backward(m: int, n: int, k: int) -> Tuple[int, int, int, int]:
    block_m = _select_block_size(m, (16, 32))
    block_n = _select_block_size(n, (32, 64))
    block_k = _select_block_size(k, (16, 32))
    num_warps = 4 if block_m * block_n >= 1024 else 2
    return block_m, block_n, block_k, num_warps


def _select_rank_block(rank: int) -> int:
    return _select_block_size(rank, (16, 32, 64))


def _validate_inputs(
    x: torch.Tensor,
    weight_4bit: torch.Tensor,
    quant_state: _QuantStateLike,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must be [batch, seq_len, hidden], got {x.shape}.")
    if lora_A.ndim != 2 or lora_B.ndim != 2:
        raise ValueError("lora_A and lora_B must be 2D weight matrices.")

    if not x.is_cuda:
        raise ValueError("x must be on CUDA device.")
    if not weight_4bit.is_cuda:
        raise ValueError("weight_4bit must be on CUDA device.")
    if not lora_A.is_cuda or not lora_B.is_cuda:
        raise ValueError("lora_A and lora_B must be on CUDA device.")

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x must be float16/bfloat16, got {x.dtype}.")

    if lora_A.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_A must be float16/bfloat16, got {lora_A.dtype}.")
    if lora_B.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_B must be float16/bfloat16, got {lora_B.dtype}.")
    if lora_A.dtype != lora_B.dtype:
        raise ValueError(
            "lora_A and lora_B must have the same dtype, "
            f"got {lora_A.dtype} and {lora_B.dtype}."
        )

    batch, seq_len, in_features = x.shape
    out_features, in_features_w = tuple(quant_state.shape)
    if in_features != in_features_w:
        raise ValueError(
            "Input hidden size does not match quantized weight shape: "
            f"{in_features} vs {in_features_w}."
        )

    rank, in_features_a = lora_A.shape
    out_features_b, rank_b = lora_B.shape
    if in_features_a != in_features:
        raise ValueError(
            f"lora_A input features mismatch: {in_features_a} vs {in_features}."
        )
    if out_features_b != out_features or rank_b != rank:
        raise ValueError(
            "lora_B shape mismatch: expected "
            f"[{out_features}, {rank}], got {lora_B.shape}."
        )

    return batch * seq_len, out_features, in_features, rank


def _validate_adapter_inputs(
    x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor
) -> Tuple[int, int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must be [batch, seq_len, hidden], got {x.shape}.")
    if lora_A.ndim != 2 or lora_B.ndim != 2:
        raise ValueError("lora_A and lora_B must be 2D weight matrices.")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA device.")
    if not lora_A.is_cuda or not lora_B.is_cuda:
        raise ValueError("lora_A and lora_B must be on CUDA device.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x must be float16/bfloat16, got {x.dtype}.")
    if lora_A.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_A must be float16/bfloat16, got {lora_A.dtype}.")
    if lora_B.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"lora_B must be float16/bfloat16, got {lora_B.dtype}.")
    if lora_A.dtype != lora_B.dtype:
        raise ValueError(
            "lora_A and lora_B must have the same dtype, "
            f"got {lora_A.dtype} and {lora_B.dtype}."
        )

    batch, seq_len, in_features = x.shape
    rank, in_features_a = lora_A.shape
    out_features_b, rank_b = lora_B.shape
    if in_features_a != in_features:
        raise ValueError(
            f"lora_A input features mismatch: {in_features_a} vs {in_features}."
        )
    if rank_b != rank:
        raise ValueError(
            f"lora_B rank mismatch: {rank_b} vs {rank}."
        )
    return batch * seq_len, out_features_b, in_features, rank


def _extract_quant_state(
    weight_4bit: torch.Tensor, quant_state: _QuantStateLike
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    bool,
    float,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    int,
]:
    absmax = quant_state.absmax
    code = quant_state.code
    blocksize = int(quant_state.blocksize)
    nested = bool(getattr(quant_state, "nested", False))

    absmax_t = absmax
    code_t = code

    device = weight_4bit.device
    absmax_t = absmax_t.to(device=device, non_blocking=True).contiguous()
    code_t = code_t.to(device=device, non_blocking=True).contiguous()

    offset = 0.0
    state2_absmax: Optional[torch.Tensor] = None
    state2_code: Optional[torch.Tensor] = None
    state2_blocksize = 0
    if nested:
        offset_val = quant_state.offset
        offset = float(offset_val) if offset_val is not None else 0.0
        state2 = quant_state.state2
        if state2 is None:
            raise ValueError("quant_state.state2 required for nested quantization.")
        state2_absmax = state2.absmax
        state2_code = state2.code
        state2_absmax = state2_absmax.to(device=device, non_blocking=True).contiguous()
        state2_code = state2_code.to(device=device, non_blocking=True).contiguous()
        state2_blocksize = int(state2.blocksize)

    return (
        absmax_t,
        code_t,
        blocksize,
        nested,
        offset,
        state2_absmax,
        state2_code,
        state2_blocksize,
    )


def _launch_matmul_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
) -> None:
    block_m, block_n, block_k, num_warps = _select_matmul_config_forward(m, n, k)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    cast(_TritonKernel, _matmul_kernel)[grid](
        a,
        b,
        out,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )


def _launch_matmul_backward(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
) -> None:
    block_m, block_n, block_k, num_warps = _select_matmul_config_backward(m, n, k)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    cast(_TritonKernel, _matmul_kernel)[grid](
        a,
        b,
        out,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )


class FusedLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        weight_4bit: torch.Tensor,
        quant_state: _QuantStateLike,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise ImportError("Triton is not available for lora_fused_forward.")

        m, n, k, r = _validate_inputs(x, weight_4bit, quant_state, lora_A, lora_B)

        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}.")

        x_contig = x.contiguous()
        x_flat = x_contig.view(m, k)

        weight_contig = weight_4bit.contiguous()
        if weight_contig.dtype != torch.uint8:
            weight_contig = weight_contig.to(torch.uint8)
        weight_flat = weight_contig.view(-1)

        lora_A_contig = lora_A.contiguous()
        lora_B_contig = lora_B.contiguous()

        (
            absmax,
            code,
            blocksize,
            nested,
            offset,
            state2_absmax,
            state2_code,
            state2_blocksize,
        ) = _extract_quant_state(weight_flat, quant_state)

        ctx_typed: _LoRAContext = cast(_LoRAContext, cast(object, ctx))

        bias_contig: Optional[torch.Tensor] = None
        if bias is not None:
            bias_contig = bias.to(dtype=x.dtype).contiguous()

        if m == 0 or n == 0 or k == 0:
            out = torch.empty((m, n), device=x.device, dtype=x.dtype)
            ctx.save_for_backward(x_flat, lora_A_contig, lora_B_contig, weight_flat)
            ctx_typed.quant_state = quant_state
            ctx_typed.scaling = float(scaling)
            ctx_typed.output_shape = x.shape[:-1] + (n,)
            ctx_typed.has_bias = bias_contig is not None
            output = out.view(ctx_typed.output_shape)
            return output

        out = torch.empty((m, n), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
        cast(_TritonKernel, _lora_forward_kernel)[grid](
            x_flat,
            weight_flat,
            absmax,
            code,
            state2_absmax if state2_absmax is not None else absmax,
            state2_code if state2_code is not None else code,
            bias_contig if bias_contig is not None else x_flat,
            lora_A_contig,
            lora_B_contig,
            out,
            x_flat.stride(0),
            x_flat.stride(1),
            lora_A_contig.stride(0),
            lora_A_contig.stride(1),
            lora_B_contig.stride(0),
            lora_B_contig.stride(1),
            out.stride(0),
            out.stride(1),
            m,
            n,
            k,
            r,
            blocksize,
            state2_blocksize if nested else 1,
            offset,
            float(scaling),
            NESTED=nested,
            LORA_DTYPE=(
                tl.float16 if lora_A_contig.dtype == torch.float16 else tl.bfloat16
            ),
            LORA_DOT_DTYPE=tl.float32,
            X_DTYPE=(tl.float16 if x_flat.dtype == torch.float16 else tl.bfloat16),
            HAS_BIAS=bias_contig is not None,
        )

        ctx.save_for_backward(x_flat, lora_A_contig, lora_B_contig, weight_flat)
        ctx_typed.quant_state = quant_state
        ctx_typed.scaling = float(scaling)
        ctx_typed.output_shape = x.shape[:-1] + (n,)
        ctx_typed.has_bias = bias_contig is not None
        output = out.view(ctx_typed.output_shape)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        Optional[torch.Tensor],
    ]:
        if not TRITON_AVAILABLE:
            raise ImportError("Triton is not available for lora_fused_forward.")

        ctx_typed = cast(_LoRAContext, cast(object, ctx))
        x_flat, lora_A, lora_B, weight_flat = ctx_typed.saved_tensors
        quant_state = ctx_typed.quant_state
        scaling = ctx_typed.scaling

        m, k = x_flat.shape
        n, r = lora_B.shape

        if len(grad_outputs) != 1:
            raise ValueError("Expected a single grad output for LoRA fused backward.")
        grad_out = grad_outputs[0].contiguous().view(m, n)

        grad_bias: Optional[torch.Tensor] = None
        if ctx_typed.has_bias:
            grad_bias = grad_out.sum(dim=0)

        (
            absmax,
            code,
            blocksize,
            nested,
            offset,
            state2_absmax,
            state2_code,
            state2_blocksize,
        ) = _extract_quant_state(weight_flat, quant_state)

        grad_x_base = torch.empty((m, k), device=x_flat.device, dtype=x_flat.dtype)
        block_m, block_n, block_k, num_warps = _select_matmul_config_backward(m, n, k)
        grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
        cast(_TritonKernel, _lora_grad_x_base_kernel)[grid](
            grad_out,
            weight_flat,
            absmax,
            code,
            state2_absmax if state2_absmax is not None else absmax,
            state2_code if state2_code is not None else code,
            grad_x_base,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_x_base.stride(0),
            grad_x_base.stride(1),
            m,
            n,
            k,
            blocksize,
            state2_blocksize if nested else 1,
            offset,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            NESTED=nested,
            num_warps=num_warps,
            num_stages=1,
        )

        if grad_out.dtype != lora_A.dtype:
            grad_out_lora = grad_out.to(dtype=lora_A.dtype)
        else:
            grad_out_lora = grad_out
        scale = grad_out_lora.new_tensor(float(scaling))
        scaled_grad = (grad_out_lora * scale).contiguous()

        grad_intermediate = torch.empty(
            (m, r), device=x_flat.device, dtype=lora_A.dtype
        )
        _launch_matmul_backward(
            scaled_grad,
            lora_B,
            grad_intermediate,
            m,
            r,
            n,
            scaled_grad.stride(0),
            scaled_grad.stride(1),
            lora_B.stride(0),
            lora_B.stride(1),
            grad_intermediate.stride(0),
            grad_intermediate.stride(1),
        )

        grad_x_lora = torch.empty((m, k), device=x_flat.device, dtype=x_flat.dtype)
        _launch_matmul_backward(
            grad_intermediate,
            lora_A,
            grad_x_lora,
            m,
            k,
            r,
            grad_intermediate.stride(0),
            grad_intermediate.stride(1),
            lora_A.stride(0),
            lora_A.stride(1),
            grad_x_lora.stride(0),
            grad_x_lora.stride(1),
        )

        grad_x = grad_x_base + grad_x_lora
        grad_x = grad_x.view(ctx_typed.output_shape[:-1] + (k,))

        if x_flat.dtype != lora_A.dtype:
            x_lora = x_flat.to(dtype=lora_A.dtype)
        else:
            x_lora = x_flat
        x_lora = x_lora.contiguous()

        xA = torch.empty((m, r), device=x_flat.device, dtype=lora_A.dtype)
        _launch_matmul_backward(
            x_lora,
            lora_A,
            xA,
            m,
            r,
            k,
            x_lora.stride(0),
            x_lora.stride(1),
            lora_A.stride(1),
            lora_A.stride(0),
            xA.stride(0),
            xA.stride(1),
        )

        grad_B = torch.empty((n, r), device=x_flat.device, dtype=lora_B.dtype)
        _launch_matmul_backward(
            scaled_grad,
            xA,
            grad_B,
            n,
            r,
            m,
            scaled_grad.stride(1),
            scaled_grad.stride(0),
            xA.stride(0),
            xA.stride(1),
            grad_B.stride(0),
            grad_B.stride(1),
        )

        grad_A = torch.empty((r, k), device=x_flat.device, dtype=lora_A.dtype)
        _launch_matmul_backward(
            grad_intermediate,
            x_lora,
            grad_A,
            r,
            k,
            m,
            grad_intermediate.stride(1),
            grad_intermediate.stride(0),
            x_lora.stride(0),
            x_lora.stride(1),
            grad_A.stride(0),
            grad_A.stride(1),
        )

        return grad_x, None, None, grad_A, grad_B, None, grad_bias


def lora_fused_forward(
    x: torch.Tensor,
    base_layer: Optional[_Linear4BitLike] = None,
    lora_A: Optional[torch.nn.Module] = None,
    lora_B: Optional[torch.nn.Module] = None,
    scaling: Optional[float] = None,
    dropout: Optional[torch.nn.Module] = None,
    lora_layer: Optional[_LoRALayerLike] = None,
    prefer_base_layer: bool = False,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available for lora_fused_forward.")

    if lora_layer is not None:
        base_layer = getattr(lora_layer, "base_layer", None)
        lora_A = getattr(lora_layer, "lora_A", None)
        lora_B = getattr(lora_layer, "lora_B", None)
        scaling = getattr(lora_layer, "scaling", None)
        dropout = getattr(lora_layer, "lora_dropout", None)

    if base_layer is None or lora_A is None or lora_B is None or scaling is None:
        raise ValueError(
            "Provide lora_layer or base_layer/lora_A/lora_B/scaling for fused LoRA."
        )

    dropout_p = getattr(dropout, "p", 0.0) if dropout is not None else 0.0
    if dropout_p and dropout_p > 0.0:
        raise ValueError("Dropout is not supported in lora_fused_forward.")

    weight_4bit = cast(Optional[torch.Tensor], getattr(base_layer, "weight", None))
    lora_A_weight = getattr(lora_A, "weight", lora_A)
    lora_B_weight = getattr(lora_B, "weight", lora_B)

    if (
        not prefer_base_layer
        and weight_4bit is not None
        and getattr(weight_4bit, "quant_state", None) is not None
    ):
        quant_state = cast(_QuantStateLike, getattr(weight_4bit, "quant_state"))
        output = cast(
            torch.Tensor,
            FusedLoRAFunction.apply(
                x,
                weight_4bit,
                quant_state,
                lora_A_weight,
                lora_B_weight,
                float(scaling),
                getattr(base_layer, "bias", None),
            ),
        )
        return output

    m, n, k, r = _validate_adapter_inputs(x, lora_A_weight, lora_B_weight)
    if r <= 0:
        raise ValueError(f"LoRA rank must be > 0, got {r}.")

    x_contig = x.contiguous()
    x_flat = x_contig.view(m, k)
    lora_A_contig = cast(torch.Tensor, lora_A_weight).contiguous()
    lora_B_contig = cast(torch.Tensor, lora_B_weight).contiguous()

    base_output = base_layer(x_contig)
    if getattr(base_output, "requires_grad", False):
        base_output = base_output.clone()

    x_lora = x_flat.to(dtype=lora_A_contig.dtype)
    lora_intermediate = torch.empty((m, r), device=x.device, dtype=lora_A_contig.dtype)
    _launch_matmul_forward(
        x_lora,
        lora_A_contig,
        lora_intermediate,
        m,
        r,
        k,
        x_lora.stride(0),
        x_lora.stride(1),
        lora_A_contig.stride(1),
        lora_A_contig.stride(0),
        lora_intermediate.stride(0),
        lora_intermediate.stride(1),
    )

    lora_out = torch.empty((m, n), device=x.device, dtype=lora_A_contig.dtype)
    _launch_matmul_forward(
        lora_intermediate,
        lora_B_contig,
        lora_out,
        m,
        n,
        r,
        lora_intermediate.stride(0),
        lora_intermediate.stride(1),
        lora_B_contig.stride(1),
        lora_B_contig.stride(0),
        lora_out.stride(0),
        lora_out.stride(1),
    )

    lora_out = lora_out * float(scaling)
    if lora_out.dtype != base_output.dtype:
        lora_out = lora_out.to(base_output.dtype)

    output = base_output + lora_out.view(base_output.shape)
    return output


__all__ = ["lora_fused_forward", "FusedLoRAFunction"]
