"""
Standalone repro captured via capture_hook.
Label: genai_torchtitan_moe_gather_combine_k8_d4096_train
Pattern hash: 6c10527cdd41
Shape hash: 48f91b5d
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
    def forward(self, arg0_1: "bf16[16384, 4096]", arg1_1: "bf16[131072, 4096]", arg2_1: "i64[131072]", arg3_1: "f32[16384, 8]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze: "f32[16384, 1, 4096]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        expand: "f32[16384, 8, 4096]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        index: "bf16[131072, 4096]" = torch.ops.aten.index.Tensor(arg1_1, [arg2_1]);  arg1_1 = None
        view: "bf16[16384, 8, 4096]" = torch.ops.aten.view.default(index, _shape_param_1);  index = _shape_param_1 = None
        convert_element_type_1: "f32[16384, 8, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[16384, 8, 4096]" = torch.ops.aten.mul.Tensor(expand, convert_element_type_1);  convert_element_type_1 = None
        unsqueeze_1: "f32[16384, 8, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
        mul_1: "f32[16384, 8, 4096]" = torch.ops.aten.mul.Tensor(expand, unsqueeze_1);  expand = unsqueeze_1 = None
        sum_1: "f32[16384, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True);  mul = None
        squeeze: "f32[16384, 8]" = torch.ops.aten.squeeze.dim(sum_1, 2);  sum_1 = None
        convert_element_type_2: "bf16[16384, 8, 4096]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view_1: "bf16[131072, 4096]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        full: "bf16[131072, 4096]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        index_put: "bf16[131072, 4096]" = torch.ops.aten.index_put.default(full, [arg2_1], view_1, True);  full = arg2_1 = view_1 = None
        return (squeeze, index_put)



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
