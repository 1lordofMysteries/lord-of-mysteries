import torch
import torch.nn as nn
from thop import profile, clever_format
from mamba_ssm import Mamba

class MambaWrapper(nn.Module):
    def __init__(self, d_model=48, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.core = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):
        return self.core(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 1, 3, 256, 256
    seq_len = H * W
    d_model = 48  # 与 BiGMa 的 chunk_dim 一致
    x = torch.randn(B, seq_len, d_model).to(device)

    model = MambaWrapper(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device).eval()

    macs, params = profile(model, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"Input: {B}x{C}x{H}x{W}  (flatten→L={seq_len}, d_model={d_model})")
    print(f"MACs: {macs} | Params: {params}")
