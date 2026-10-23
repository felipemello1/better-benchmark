"""
Standalone repro captured via capture_hook.
Label: genai_torchtitan_llama3_8b_interleaved_rope_train
Pattern hash: 1e2489a4a85a
Shape hash: f65ccca0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16384, 32, 128]", arg1_1: "f32[16384, 1, 64]", arg2_1: "f32[16384, 1, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[16384, 32, 64, 2]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        select: "bf16[16384, 32, 64]" = torch.ops.aten.select.int(view, -1, 0)
        select_1: "bf16[16384, 32, 64]" = torch.ops.aten.select.int(view, -1, 1);  view = None
        convert_element_type: "f32[16384, 32, 64]" = torch.ops.prims.convert_element_type.default(select_1, torch.float32);  select_1 = None
        mul: "f32[16384, 32, 64]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1)
        mul_1: "f32[16384, 32, 64]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = None
        convert_element_type_1: "f32[16384, 32, 64]" = torch.ops.prims.convert_element_type.default(select, torch.float32);  select = None
        neg: "f32[16384, 32, 64]" = torch.ops.aten.neg.default(convert_element_type_1)
        mul_2: "f32[16384, 32, 64]" = torch.ops.aten.mul.Tensor(neg, arg2_1);  neg = arg2_1 = None
        add: "f32[16384, 32, 64]" = torch.ops.aten.add.Tensor(mul, mul_2);  mul = mul_2 = None
        mul_3: "f32[16384, 32, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg1_1);  convert_element_type_1 = arg1_1 = None
        add_1: "f32[16384, 32, 64]" = torch.ops.aten.add.Tensor(mul_1, mul_3);  mul_1 = mul_3 = None
        unsqueeze: "f32[16384, 32, 64, 1]" = torch.ops.aten.unsqueeze.default(add_1, 3);  add_1 = None
        unsqueeze_1: "f32[16384, 32, 64, 1]" = torch.ops.aten.unsqueeze.default(add, 3);  add = None
        cat: "f32[16384, 32, 64, 2]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        view_1: "f32[16384, 32, 128]" = torch.ops.aten.view.default(cat, _shape_param_1);  cat = _shape_param_1 = None
        convert_element_type_2: "bf16[16384, 32, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        return convert_element_type_2



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
