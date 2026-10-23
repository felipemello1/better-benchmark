"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the backward of interleaved RoPE (for the output-grad pair g_r = g[2k], g_i = g[2k+1]: grad_a = g_i*sin + g_r*cos, grad_b = g_i*cos + (-g_r)*sin, in fp32, cast to bf16), loading each bf16 grad pair as one 32-bit word and storing the result pair as one 32-bit word (the backward of TorchTitan's hand-written int32 form), with explicit tl.fma in Inductor's contraction order so the output is bitwise equal to Inductor's (the int32 unpack changes the IR, so with contraction left to LLVM it fuses the other product), whereas Inductor lowers the stacked fp32 lanes as a pointwise cat whose two lane branches each load g as masked 16-bit elements and cos/sin as scalar 32-bit values (8 scalar loads per output element in its PTX), 2.0-2.4x off this kernel at 4k-16k tokens on GB300 with coordinate descent (2.1-2.6x without); Inductor cannot do this today because its codegen never pairs two adjacent 16-bit accesses of a stack/cat into one wider access; the fix is NEW_PATTERN: load the adjacent 16-bit pair x[2k], x[2k+1] that both lanes read once, as one 32-bit word of a 4-byte-aligned buffer, instead of masked 16-bit loads in each lane branch."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl

CONFIGS = [triton.Config({"XBLOCK": xb}, num_warps=w) for xb in (512, 1024, 2048) for w in (4, 8)]


@triton.autotune(CONFIGS, key=["N", "H", "P"])
@triton.jit
def _rope_pairs_bwd_kernel(g_ptr, cos_ptr, sin_ptr, out_ptr, N, H: tl.constexpr, P: tl.constexpr,
                           XBLOCK: tl.constexpr):
    # one bf16 pair per int32 word: k indexes pairs of g [T, H, 2P]; cos/sin are [T, 1, P]
    k = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    m = k < N
    w = tl.load(g_ptr.to(tl.pointer_type(tl.int32)) + k, m)
    g_r = (w << 16).to(tl.float32, bitcast=True)
    g_i = (w & -65536).to(tl.float32, bitcast=True)
    row = (k // (H * P)) * P + k % P
    c = tl.load(cos_ptr + row, m, eviction_policy="evict_last")
    s = tl.load(sin_ptr + row, m, eviction_policy="evict_last")
    # Inductor's contraction: the first product goes into the FMA, the second is rounded
    lo = tl.fma(g_i, s, g_r * c).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    hi = tl.fma(g_i, c, (-g_r) * s).to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
    tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + k, lo | hi, m)


@oracle_impl(hardware="GB300")
def oracle_forward(inputs):
    g, cos, sin, _pair_shape, _out_shape = inputs
    T, H, D = g.shape
    # 32-bit word access needs a contiguous, 4-byte-aligned bf16 input
    assert g.is_contiguous() and g.data_ptr() % 4 == 0 and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty_like(g)
    N = T * H * (D // 2)
    _rope_pairs_bwd_kernel[lambda meta: (triton.cdiv(N, meta["XBLOCK"]),)](g, cos, sin, out, N, H=H, P=D // 2)
    return out
