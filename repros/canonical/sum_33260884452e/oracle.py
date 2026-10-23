"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the training forward of the MoE combine (each token sums its K routed expert rows, weighted by the router scores, then casts to bf16; the flattened index is returned as a view, as in the repro) in registers, one output element per thread with K coalesced loads of the routed rows (the kernel Inductor itself emits when the reduction is unrolled with unroll_reductions_threshold=K+1, transcribed shape-general, with unmasked loads when the grid tiles the output exactly as in Inductor's static kernel, and bitwise equal to it at every point under either of Inductor's two pointwise configs), whereas Inductor lowers the K = 8-32 element sum over the non-innermost dim as a persistent reduction, 1.8x (K=8, D=4096) and 2.8x (K=16, D=7168) off this kernel at 16k tokens on GB300 with coordinate descent (5.6x and 5.7x without; idx a permutation); Inductor cannot do this today because it unrolls only reductions with fewer than unroll_reductions_threshold (8) elements, and for 8 <= K <= 32 num_splits returns before looking at strides and the persistent XBLOCK stops at 128; the fix is NEW_PATTERN: unroll small reductions whose reads are strided or gathered along the reduced dim."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl

CONFIGS = [triton.Config({"XBLOCK": xb}, num_warps=w) for xb in (512, 1024, 2048) for w in (4, 8)]


@triton.autotune(CONFIGS, key=["xnumel", "K", "D"])
@triton.heuristics({"EVEN": lambda a: a["xnumel"] % a["XBLOCK"] == 0})
@triton.jit
def _gather_combine_kernel(idx_ptr, rows_ptr, scores_ptr, out_ptr, xnumel, n_rows,
                           K: tl.constexpr, D: tl.constexpr, XBLOCK: tl.constexpr, EVEN: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    # no mask when the grid tiles xnumel exactly, like Inductor's static kernel
    xmask = None if EVEN else xindex < xnumel
    x1 = xindex // D
    x0 = xindex % D
    acc = tl.zeros([XBLOCK], tl.float32)
    for k in tl.static_range(K):
        i = tl.load(idx_ptr + (k + K * x1), xmask, eviction_policy="evict_last")
        i = tl.where(i < 0, i + n_rows, i)
        s = tl.load(scores_ptr + (k + K * x1), xmask, eviction_policy="evict_last")
        v = tl.load(rows_ptr + (x0 + D * i), xmask).to(tl.float32) * s
        if k == 0:
            acc = v
        else:
            acc = acc + v
    tl.store(out_ptr + xindex, acc.to(tl.bfloat16), xmask)


def _combine(idx, rows, scores):
    # the kernel indexes idx/scores as row-major [T, K] and rows as row-major [n_rows, D]
    assert idx.is_contiguous() and rows.is_contiguous() and scores.is_contiguous()
    T, K = idx.shape
    D = rows.shape[1]
    out = torch.empty(T, D, device=rows.device, dtype=rows.dtype)
    xnumel = T * D
    # FMA contraction on, as Inductor compiles its kernels (Triton's default is off)
    _gather_combine_kernel[lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]),)](
        idx, rows, scores, out, xnumel, rows.shape[0], K=K, D=D, enable_fp_fusion=True)
    return out


@oracle_impl(hardware="GB300")
def oracle_forward(inputs):
    idx, rows, scores, flat_idx_shape, _view_shape = inputs
    return idx.view(tuple(int(d) for d in flat_idx_shape)), _combine(idx, rows, scores)
