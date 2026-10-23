class GraphModule(torch.nn.Module):
    def forward(self, view: "i64[32768][1]cuda:0", tangents_1: "bf16[4096, 4096][4096, 1]cuda:0"):
        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:61 in dispatch_bwd_sum, code: return x.sum(1).to(rows.dtype)
        convert_element_type_2: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        unsqueeze: "f32[4096, 1, 4096][4096, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        expand: "f32[4096, 8, 4096][4096, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze, [4096, 8, 4096]);  unsqueeze = None

        # Backward of forward node: File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:60 in dispatch_bwd_sum, code: x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
        convert_element_type_3: "bf16[4096, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.bfloat16);  expand = None
        view_2: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [32768, 4096]);  convert_element_type_3 = None
        full_default: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.full.default([32768, 4096], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "bf16[32768, 4096][4096, 1]cuda:0" = torch.ops.aten.index_put.default(full_default, [view], view_2, True);  full_default = view = view_2 = None
        return (None, index_put)
