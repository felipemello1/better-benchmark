class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[65536, 128, 64][8192, 64, 1]cuda:0", arg1_1: "f32[65536, 1, 32][32, 32, 1]cuda:0", arg2_1: "f32[65536, 1, 32][32, 32, 1]cuda:0"):
        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:65 in rope_castfirst, code: a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        convert_element_type: "f32[65536, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[65536, 128, 32, 2][8192, 64, 2, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type, [65536, 128, 32, 2]);  convert_element_type = None
        unbind = torch.ops.aten.unbind.int(view, -1);  view = None
        getitem: "f32[65536, 128, 32][8192, 64, 2]cuda:0" = unbind[0]
        getitem_1: "f32[65536, 128, 32][8192, 64, 2]cuda:0" = unbind[1];  unbind = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:66 in rope_castfirst, code: real = (a * cos - b * sin).to(x.dtype)
        mul: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem, arg1_1)
        mul_1: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_1, arg2_1)
        sub: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_1: "bf16[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:68 in rope_castfirst, code: return torch.stack((real, imag), -1).flatten(-2)
        unsqueeze: "bf16[65536, 128, 32, 1][4096, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3);  convert_element_type_1 = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:67 in rope_castfirst, code: imag = (a * sin + b * cos).to(x.dtype)
        mul_2: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem, arg2_1);  getitem = arg2_1 = None
        mul_3: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(getitem_1, arg1_1);  getitem_1 = arg1_1 = None
        add: "f32[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        convert_element_type_2: "bf16[65536, 128, 32][4096, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:68 in rope_castfirst, code: return torch.stack((real, imag), -1).flatten(-2)
        unsqueeze_1: "bf16[65536, 128, 32, 1][4096, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        cat: "bf16[65536, 128, 32, 2][8192, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        view_1: "bf16[65536, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [65536, 128, 64]);  cat = None
        return (view_1,)
