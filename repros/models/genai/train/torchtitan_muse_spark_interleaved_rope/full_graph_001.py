class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[16384, 1, 32][32, 32, 1]cuda:0", primals_3: "f32[16384, 1, 32][32, 32, 1]cuda:0", tangents_1: "bf16[16384, 128, 64][8192, 64, 1]cuda:0"):
        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:68 in rope_castfirst, code: return torch.stack((real, imag), -1).flatten(-2)
        view_2: "bf16[16384, 128, 32, 2][8192, 64, 2, 1]cuda:0" = torch.ops.aten.reshape.default(tangents_1, [16384, 128, 32, 2]);  tangents_1 = None
        select: "bf16[16384, 128, 32][8192, 64, 2]cuda:0" = torch.ops.aten.select.int(view_2, -1, 0)
        select_1: "bf16[16384, 128, 32][8192, 64, 2]cuda:0" = torch.ops.aten.select.int(view_2, -1, 1);  view_2 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:67 in rope_castfirst, code: imag = (a * sin + b * cos).to(x.dtype)
        convert_element_type_3: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select_1, torch.float32);  select_1 = None
        mul_4: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, primals_2)
        mul_5: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, primals_3);  convert_element_type_3 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:66 in rope_castfirst, code: real = (a * cos - b * sin).to(x.dtype)
        convert_element_type_4: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select, torch.float32);  select = None
        neg: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_4)
        mul_6: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg, primals_3);  neg = primals_3 = None
        add_1: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_6);  mul_4 = mul_6 = None
        mul_7: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, primals_2);  convert_element_type_4 = primals_2 = None
        add_2: "f32[16384, 128, 32][4096, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_5, mul_7);  mul_5 = mul_7 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:65 in rope_castfirst, code: a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        unsqueeze_2: "f32[16384, 128, 32, 1][4096, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_2, 3);  add_2 = None
        unsqueeze_3: "f32[16384, 128, 32, 1][4096, 32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_1, 3);  add_1 = None
        cat_1: "f32[16384, 128, 32, 2][8192, 64, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        view_3: "f32[16384, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [16384, 128, 64]);  cat_1 = None
        convert_element_type_5: "bf16[16384, 128, 64][8192, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        return (convert_element_type_5, None, None)
