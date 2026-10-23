"""
Standalone repro captured via capture_hook.
Label: genai_torchtitan_llama3_8b_interleaved_rope_infer
Pattern hash: f676391b22ad
Shape hash: ce0bf204
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
    def forward(self, arg0_1: "bf16[65536, 32, 128]", arg1_1: "f32[65536, 1, 64]", arg2_1: "f32[65536, 1, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "f32[65536, 32, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[65536, 32, 64, 2]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        unbind = torch.ops.aten.unbind.int(view, -1);  view = None
        getitem: "f32[65536, 32, 64]" = unbind[0]
        getitem_1: "f32[65536, 32, 64]" = unbind[1];  unbind = None
        mul: "f32[65536, 32, 64]" = torch.ops.aten.mul.Tensor(getitem, arg1_1)
        mul_1: "f32[65536, 32, 64]" = torch.ops.aten.mul.Tensor(getitem_1, arg2_1)
        sub: "f32[65536, 32, 64]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_1: "bf16[65536, 32, 64]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        unsqueeze: "bf16[65536, 32, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3);  convert_element_type_1 = None
        mul_2: "f32[65536, 32, 64]" = torch.ops.aten.mul.Tensor(getitem, arg2_1);  getitem = arg2_1 = None
        mul_3: "f32[65536, 32, 64]" = torch.ops.aten.mul.Tensor(getitem_1, arg1_1);  getitem_1 = arg1_1 = None
        add: "f32[65536, 32, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        convert_element_type_2: "bf16[65536, 32, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_1: "bf16[65536, 32, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        cat: "bf16[65536, 32, 64, 2]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        view_1: "bf16[65536, 32, 128]" = torch.ops.aten.view.default(cat, _shape_param_1);  cat = _shape_param_1 = None
        return view_1



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
