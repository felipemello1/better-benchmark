class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "bf16[32768, 4096][4096, 1]cuda:0", primals_3: "f32[4096, 8][8, 1]cuda:0", view: "i64[32768][1]cuda:0", tangents_1: "bf16[4096, 4096][4096, 1]cuda:0"):
        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:55 in gather_combine, code: return (x * scores[..., None]).sum(1).to(rows.dtype)
        convert_element_type_2: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        unsqueeze_1: "f32[4096, 1, 4096][4096, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        expand: "f32[4096, 8, 4096][4096, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_1, [4096, 8, 4096]);  unsqueeze_1 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:54 in gather_combine, code: x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
        index: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_2, [view]);  primals_2 = None
        view_1: "bf16[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(index, [4096, 8, 4096]);  index = None
        convert_element_type: "f32[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:55 in gather_combine, code: return (x * scores[..., None]).sum(1).to(rows.dtype)
        mul_1: "f32[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(expand, convert_element_type);  convert_element_type = None
        unsqueeze: "f32[4096, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_3, 2);  primals_3 = None
        mul_2: "f32[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(expand, unsqueeze);  expand = unsqueeze = None
        sum_2: "f32[4096, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1, [2], True);  mul_1 = None
        squeeze: "f32[4096, 8][8, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_2, 2);  sum_2 = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:54 in gather_combine, code: x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
        convert_element_type_3: "bf16[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view_2: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [32768, 4096]);  convert_element_type_3 = None
        full_default: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.full.default([32768, 4096], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.index_put.default(full_default, [view], view_2, True);  full_default = view = view_2 = None
        return (None, index_put, squeeze)
