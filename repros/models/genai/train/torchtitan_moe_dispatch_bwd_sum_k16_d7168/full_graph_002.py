class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4096, 16][16, 1]cuda:0", primals_2: "bf16[65536, 7168][7168, 1]cuda:0"):
        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:60 in dispatch_bwd_sum, code: x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
        view: "i64[65536][1]cuda:0" = torch.ops.aten.reshape.default(primals_1, [65536]);  primals_1 = None
        index: "bf16[65536, 7168][7168, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_2, [view]);  primals_2 = None
        view_1: "bf16[4096, 16, 7168][114688, 7168, 1]cuda:0" = torch.ops.aten.reshape.default(index, [4096, 16, 7168]);  index = None
        convert_element_type: "f32[4096, 16, 7168][114688, 7168, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:61 in dispatch_bwd_sum, code: return x.sum(1).to(rows.dtype)
        sum_1: "f32[4096, 7168][7168, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type, [1]);  convert_element_type = None
        convert_element_type_1: "bf16[4096, 7168][7168, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        return (convert_element_type_1, view)
