#!/usr/bin/env python3
"""Capture TorchTitan MoE and RoPE microbenchmark graphs into the corpus (genai suite).

Each case is a small function compiled at several token counts T with the
capture hook installed and merged into the canonical corpus under
models/genai/<mode>/<name>:

- moe_gather_combine_*: MoE combine as a gather and a sum. Each token sums its
  K routed expert outputs, weighted by the router scores:
  (rows[idx].view(T, K, D).float() * scores[..., None]).sum(1).to(bf16)
  TorchTitan's main combines with a deterministic scatter-add; this is the
  gather-sum form proposed for its MoE.
- moe_dispatch_bwd_sum_*: the backward of a gather dispatch. Each token sums
  the gradients of its K routed copies: rows[idx].view(T, K, D).float().sum(1).
- *_interleaved_rope: interleaved RoPE as a stack of two lanes, each cast to
  the input dtype first. TorchTitan keeps this form for fp32/fp16 only; for
  bf16 it writes each pair as one int32 word (and reads one, if contiguous),
  because Inductor's lowering of the stack is 1.2-1.5x slower forward and
  2.0-2.4x backward (that int32 kernel is the RoPE oracle here).

idx is a route map: every (token, k) owns its own row. After each merge its
generator is set to a permutation (see routes_as_permutations).

infer = no_grad forward; train = forward + a real backward.

Usage:
    python scripts/capture_torchtitan.py --corpus-root repros \
        [--cases moe_gather_combine_k8_d4096,...] [--modes infer,train] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MOE_INFER_T = (16384, 2048, 64)
MOE_TRAIN_T = (16384, 4096)
ROPE_INFER_T = (65536, 16384, 2048, 64)
ROPE_TRAIN_T = (16384, 4096)


def gather_combine(rows, idx, scores):
    T, K = idx.shape
    x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
    return (x * scores[..., None]).sum(1).to(rows.dtype)


def dispatch_bwd_sum(rows, idx):
    T, K = idx.shape
    x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
    return x.sum(1).to(rows.dtype)


def rope_castfirst(x, cos, sin):
    a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    real = (a * cos - b * sin).to(x.dtype)
    imag = (a * sin + b * cos).to(x.dtype)
    return torch.stack((real, imag), -1).flatten(-2)


def moe_inputs(K, D, with_scores):
    def make(T, requires_grad):
        g = torch.Generator(device="cuda").manual_seed(0)
        rows = torch.randn(T * K, D, device="cuda", generator=g).bfloat16()
        idx = torch.randperm(T * K, device="cuda", generator=g).view(T, K)
        args = [rows.requires_grad_(requires_grad), idx]
        if with_scores:
            scores = torch.rand(T, K, device="cuda", generator=g)
            args.append(scores.requires_grad_(requires_grad))
        return args
    return make


def rope_inputs(H, D):
    def make(T, requires_grad):
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(T, H, D, device="cuda", generator=g).bfloat16()
        freqs = torch.rand(T, 1, D // 2, device="cuda", generator=g) * 6.28
        return [x.requires_grad_(requires_grad), freqs.cos(), freqs.sin()]
    return make


# name: (fn, make_inputs, infer T, train T)
CASES = {
    "moe_gather_combine_k8_d4096": (gather_combine, lambda: moe_inputs(8, 4096, True), MOE_INFER_T, MOE_TRAIN_T),
    "moe_gather_combine_k16_d7168": (gather_combine, lambda: moe_inputs(16, 7168, True), MOE_INFER_T, MOE_TRAIN_T),
    "moe_dispatch_bwd_sum_k8_d4096": (dispatch_bwd_sum, lambda: moe_inputs(8, 4096, False), MOE_INFER_T, MOE_TRAIN_T),
    "moe_dispatch_bwd_sum_k16_d7168": (dispatch_bwd_sum, lambda: moe_inputs(16, 7168, False), MOE_INFER_T, MOE_TRAIN_T),
    "llama3_8b_interleaved_rope": (rope_castfirst, lambda: rope_inputs(32, 128), ROPE_INFER_T, ROPE_TRAIN_T),
    "muse_spark_interleaved_rope": (rope_castfirst, lambda: rope_inputs(128, 64), ROPE_INFER_T, ROPE_TRAIN_T),
}


def routes_as_permutations(corpus_root: Path, model_key: str) -> int:
    """Draw this model's route maps as permutations, as TorchTitan's are.

    The capture records idx as uniform ids drawn with replacement
    (["index", 0, N]), which adds duplicate rows to the gathers and atomic
    conflicts to the backward index_put. An integer input whose id range equals
    its element count becomes ["perm", N]. Later merges keep "perm" over
    "index" (merge_captures._upgrade_runs_gens). A point shared with another
    model becomes a permutation for every model on it, the same optimistic
    rule _upgrade_runs_gens uses. Returns the inputs changed.
    """
    from full_graph_harness import _OneLine, dumps_with_onelines
    from merge_captures import _atomic_write_text

    changed = 0
    for path in sorted((corpus_root / "canonical").glob("*/shapes.json")):
        data = json.loads(path.read_text())
        hits = 0
        for point in data.get("points", []):
            if model_key not in point.get("models", {}):
                continue
            for entry in point.get("inputs") or []:
                gen = entry[2].get("gen") if len(entry) > 2 and isinstance(entry[2], dict) else None
                if gen and gen[0] == "index" and gen[1] == 0 and gen[2] == math.prod(entry[0]):
                    entry[2]["gen"] = ["perm", gen[2]]
                    hits += 1
        if hits:
            for point in data["points"]:
                if isinstance(point.get("inputs"), list):
                    point["inputs"] = [_OneLine(e) for e in point["inputs"]]
            _atomic_write_text(path, dumps_with_onelines(data) + "\n")
            changed += hits
    return changed


def _raise_timeout(signum, frame):
    raise TimeoutError("capture timed out")


def capture_one(name: str, mode: str, corpus_root: Path, timeout_s: int) -> dict:
    from torch._inductor.utils import fresh_inductor_cache
    from capture_hook import install_capture_hook, uninstall_capture_hook
    from run_recapture import write_run_log_entry, _count_roundtrips
    from merge_captures import temporary_capture_for_merge

    fn, make_factory, infer_t, train_t = CASES[name]
    make_inputs = make_factory()
    model_name = f"torchtitan_{name}"
    t0 = time.time()
    entry = {
        "model_key": f"genai/{mode}/{model_name}",
        "suite": "genai",
        "model": model_name,
        "mode": mode,
        "status": "failed",
        "wall_time_s": 0.0,
        "region_count": 0,
        "roundtrip_summary": {"ok": 0, "failed": 0},
        "error_tail": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    model_dir = corpus_root / "models" / "genai" / mode / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_s)
    try:
        with temporary_capture_for_merge(
                corpus_root, model_name, suite="genai", mode=mode) as capture:
            install_capture_hook(
                output_dir=str(capture.capture_dir),
                label=f"genai_{model_name}_{mode}",
                graph_dir=str(model_dir),
                validate=True,
            )
            try:
                with fresh_inductor_cache():
                    for T in infer_t if mode == "infer" else train_t:
                        torch._dynamo.reset()
                        args = make_inputs(T, requires_grad=mode == "train")
                        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
                        if mode == "train":
                            out = compiled(*args)
                            out.backward(torch.randn_like(out))
                        else:
                            with torch.no_grad():
                                compiled(*args)
                        torch.cuda.synchronize()
                        del args
                        torch.cuda.empty_cache()
            finally:
                uninstall_capture_hook()

            # fail-hard on drops, same as run_recapture
            index_path = capture.capture_dir / "index.json"
            if index_path.exists():
                idx = json.loads(index_path.read_text())
                dropped = (idx.get("dropped", [])
                           if isinstance(idx, dict) else [])
                if dropped:
                    raise RuntimeError(
                        f"{len(dropped)} region(s) DROPPED: "
                        f"{dropped[0].get('reason', '?')[:200]}")

            n = capture.merge()
        routes_as_permutations(corpus_root, entry["model_key"])

        rt_ok, rt_failed = _count_roundtrips(model_dir)
        entry["status"] = "ok"
        entry["region_count"] = n
        entry["roundtrip_summary"] = {"ok": rt_ok, "failed": rt_failed}
    except Exception:  # noqa: BLE001 -- per-case isolation
        import traceback
        entry["error_tail"] = traceback.format_exc()[-1500:]
    finally:
        signal.alarm(0)
    entry["wall_time_s"] = time.time() - t0
    write_run_log_entry(corpus_root, entry)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--modes", default="infer,train")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--timeout", type=int, default=900, help="per case and mode, seconds")
    args = ap.parse_args()

    import os
    # safe after `import torch`: CUDA isn't initialized until the first CUDA call
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    names = list(CASES) if args.cases == "all" else [c for c in args.cases.split(",") if c]
    modes = [m for m in args.modes.split(",") if m]
    print(f"torchtitan capture: {len(names)} cases x {modes}")
    ok = failed = 0
    for name in names:
        for mode in modes:
            e = capture_one(name, mode, args.corpus_root, args.timeout)
            ok += e["status"] == "ok"
            failed += e["status"] != "ok"
            print(f"  {name} {mode}: {e['status']} ({e['region_count']} regions, "
                  f"{e['wall_time_s']:.0f}s)"
                  + (f" — {e['error_tail'][-300:]}" if e["error_tail"] else ""))
    print(f"torchtitan: {ok} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
