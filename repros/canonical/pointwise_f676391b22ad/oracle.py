"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes interleaved RoPE written as a stack of two lanes, each cast to bf16 before the stack (real = a*cos - b*sin, imag = a*sin + b*cos for the pair a = x[2k], b = x[2k+1]), loading each bf16 pair as one 32-bit word and storing the rotated pair as one 32-bit word (up to 128-bit vector loads and stores per thread; TorchTitan hand-writes this int32 form for bf16 inputs (bf16 split views read the two halves but still write int32 pairs) and keeps the stack form for fp32/fp16), with explicit tl.fma in Inductor's contraction order so the output is bitwise equal to Inductor's (the int32 unpack changes the IR, so with contraction left to LLVM it fuses the other product), whereas Inductor lowers the stack as a pointwise cat whose two lane branches each load x as masked 16-bit elements and cos/sin as scalar 32-bit values (8 scalar loads per output element in its PTX; its stores do vectorize), 1.24-1.27x off this kernel at 64k tokens on GB300 with coordinate descent (1.36x without); Inductor cannot do this today because its codegen never pairs two adjacent 16-bit accesses of a stack/cat into one wider access; the fix is NEW_PATTERN: load the adjacent 16-bit pair x[2k], x[2k+1] that both lanes read once, as one 32-bit word of a 4-byte-aligned buffer, instead of masked 16-bit loads in each lane branch."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl

CONFIGS = [triton.Config({"XBLOCK": xb}, num_warps=w) for xb in (512, 1024, 2048) for w in (4, 8)]


@triton.autotune(CONFIGS, key=["N", "H", "P"])
@triton.jit
def _rope_pairs_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, N, H: tl.constexpr, P: tl.constexpr,
                       XBLOCK: tl.constexpr):
    # one bf16 pair per int32 word: k indexes pairs of x [T, H, 2P]; cos/sin are [T, 1, P]
    k = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    m = k < N
    w = tl.load(x_ptr.to(tl.pointer_type(tl.int32)) + k, m)
    a = (w << 16).to(tl.float32, bitcast=True)
    b = (w & -65536).to(tl.float32, bitcast=True)
    row = (k // (H * P)) * P + k % P
    c = tl.load(cos_ptr + row, m, eviction_policy="evict_last")
    s = tl.load(sin_ptr + row, m, eviction_policy="evict_last")
    # Inductor's contraction: the first product goes into the FMA, the second is rounded
    lo = tl.fma(a, c, -(b * s)).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    hi = tl.fma(a, s, b * c).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
    tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + k, lo | hi, m)


@oracle_impl(hardware="GB300")
def oracle_forward(inputs):
    x, cos, sin, _pair_shape, _out_shape = inputs
    T, H, D = x.shape
    # 32-bit word access needs a contiguous, 4-byte-aligned bf16 input
    assert x.is_contiguous() and x.data_ptr() % 4 == 0 and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty_like(x)
    N = T * H * (D // 2)
    _rope_pairs_kernel[lambda meta: (triton.cdiv(N, meta["XBLOCK"]),)](x, cos, sin, out, N, H=H, P=D // 2)
    return out
