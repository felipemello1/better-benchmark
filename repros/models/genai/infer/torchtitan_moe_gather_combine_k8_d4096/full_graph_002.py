class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[64, 8][8, 1]cuda:0", arg1_1: "bf16[512, 4096][4096, 1]cuda:0", arg2_1: "f32[64, 8][8, 1]cuda:0"):
        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:54 in gather_combine, code: x = rows[idx.flatten()].view(T, K, rows.shape[-1]).float()
        view: "i64[512][1]cuda:0" = torch.ops.aten.reshape.default(arg0_1, [512]);  arg0_1 = None
        index: "bf16[512, 4096][4096, 1]cuda:0" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "bf16[64, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(index, [64, 8, 4096]);  index = None
        convert_element_type: "f32[64, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # File: /home/felipemello/refs/bb-bb1/scripts/capture_torchtitan.py:55 in gather_combine, code: return (x * scores[..., None]).sum(1).to(rows.dtype)
        unsqueeze: "f32[64, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
        mul: "f32[64, 8, 4096][32768, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, unsqueeze);  convert_element_type = unsqueeze = None
        sum_1: "f32[64, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [1]);  mul = None
        convert_element_type_1: "bf16[64, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        return (convert_element_type_1,)
