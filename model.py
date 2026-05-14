import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from gcn_layers_pyg import GraphConvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MSDP: Multi-scale Decomposition Perception ----------
class MSDP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.path1 = nn.Sequential(
            conv_block(in_channels, out_channels),
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(out_channels, out_channels),
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(out_channels, out_channels)
        )

        self.path2 = nn.Sequential(
            conv_block(in_channels, out_channels),
            nn.AvgPool2d(3, stride=1, padding=1),
            conv_block(out_channels, out_channels)
        )

        self.path3 = conv_block(in_channels, out_channels)

        self.gate_conv = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.gate_sigmoid = nn.Sigmoid()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.path1(x)
        f2 = self.path2(x)
        f3 = self.path3(x)

        cat = torch.cat([f1, f2, f3], dim=1)
        gate = self.gate_sigmoid(self.gate_conv(cat))

        out = f1 * gate + f2 * gate + f3 * (1 - gate)
        out = self.fusion_conv(out)
        return out


# ---------- GCN Branch (two layers + gated residual) ----------
class GCNBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        h1 = self.conv1(x, edge_index, edge_weight)
        h1 = self.bn1(h1)
        h1 = self.relu(h1)

        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = self.bn2(h2)
        gate = torch.sigmoid(h1)
        h2 = self.relu(h2 + h1 * gate)
        return h2


# ---------- NIA: Node Interaction Attention ----------
class NIA(nn.Module):
    def __init__(self, in_channels, num_heads=8, pool_ratio=8):
        super().__init__()
        assert in_channels % num_heads == 0
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.pool_ratio = pool_ratio
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(in_channels, num_heads, bias=False)
        self.k_proj = nn.Linear(in_channels, num_heads, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, h, w):
        B = 1
        N, C = x.shape
        pad_len = h * w - N
        if pad_len > 0:
            x_pad = torch.cat([x, torch.zeros(pad_len, C, device=x.device)], dim=0)
        else:
            x_pad = x[:h*w]
        x_grid = x_pad.view(1, h, w, C)

        x_pooled = x_grid.permute(0, 3, 1, 2)
        x_pooled = self.pool(x_pooled)
        x_pooled = self.conv(x_pooled)
        _, _, hp, wp = x_pooled.shape
        x_pooled = x_pooled.permute(0, 2, 3, 1)
        x_pooled = self.norm(x_pooled)
        x_pooled = self.act(x_pooled)

        Q = self.q_proj(x_grid)
        K = self.k_proj(x_pooled)
        V = self.v_proj(x_pooled)

        Q = Q.permute(0, 3, 1, 2).reshape(1, self.num_heads, -1, 1)
        K = K.permute(0, 3, 1, 2).reshape(1, self.num_heads, -1, 1)
        V = V.permute(0, 3, 1, 2).reshape(1, self.num_heads, -1, C // self.num_heads)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(1, h*w, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out[0, :N]
        return out


# ---------- Local Window Attention (used in MSAF) ----------
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, H, W, C = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape
        x = x.view(B, Hp // self.window_size, self.window_size,
                   Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.num_heads, -1).transpose(1, 2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out = self.proj(out)
        out = self.dropout(out)

        out = out.view(B, Hp // self.window_size, Wp // self.window_size,
                      self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]
        return out


# ---------- MSAF: Multi-scale Adaptive Fusion ----------
class MSAF(nn.Module):
    def __init__(self, channels, down_rate=2, min_size=16, window_size=8, num_heads=4):
        super().__init__()
        self.channels = channels
        self.down_rate = down_rate
        self.min_size = min_size
        self.window_size = window_size
        self.num_heads = num_heads

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, cnn_feat, gcn_feat):
        B, C, H, W = cnn_feat.shape
        cat = torch.cat([cnn_feat, gcn_feat], dim=1)
        cat = cat.permute(0, 2, 3, 1)

        levels = 1
        h, w = H, W
        while h // self.down_rate >= self.min_size and w // self.down_rate >= self.min_size:
            levels += 1
            h //= self.down_rate
            w //= self.down_rate

        feats = []
        cur = cat
        for l in range(levels):
            attn = LocalWindowAttention(cur.shape[-1], self.num_heads, self.window_size)(cur)
            feats.append(attn)
            if l < levels - 1:
                cur = cur.permute(0, 3, 1, 2)
                cur = F.avg_pool2d(cur, self.down_rate)
                cur = cur.permute(0, 2, 3, 1)

        fused = feats[-1]
        for l in range(levels - 2, -1, -1):
            _, Hl, Wl, _ = feats[l].shape
            fused = fused.permute(0, 3, 1, 2)
            fused = F.interpolate(fused, size=(Hl, Wl), mode='bilinear', align_corners=True)
            fused = fused.permute(0, 2, 3, 1)
            fused = feats[l] + (1.0 / (levels - l)) * fused

        fused = fused.permute(0, 3, 1, 2)
        out = self.out_conv(fused)
        return out


# ---------- M²FNet: Main Network ----------
class M2FNet(nn.Module):
    def __init__(self, height, width, channel, class_count,
                 Q1, edge_index1, edge_weight1,
                 Q2, edge_index2, edge_weight2):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.class_count = class_count

        self.Q1 = Q1
        self.Q2 = Q2
        self.edge_index1 = edge_index1
        self.edge_weight1 = edge_weight1
        self.edge_index2 = edge_index2
        self.edge_weight2 = edge_weight2

        self.msdp1 = MSDP(channel, 64)
        self.msdp2 = MSDP(channel, 64)

        self.gcn_branch1 = GCNBranch(64, 64)
        self.gcn_branch2 = GCNBranch(64, 64)

        self.nia1 = NIA(64, num_heads=8, pool_ratio=8)
        self.nia2 = NIA(64, num_heads=8, pool_ratio=8)

        self.msaf = MSAF(channels=64, down_rate=2, min_size=16, window_size=8, num_heads=4)

        self.classifier = nn.Linear(128, class_count)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        H, W, C = x1.shape

        # CNN features via MSDP
        x1_cnn = x1.permute(2, 0, 1).unsqueeze(0)
        x2_cnn = x2.permute(2, 0, 1).unsqueeze(0)
        cnn_feat1 = self.msdp1(x1_cnn)
        cnn_feat2 = self.msdp2(x2_cnn)

        # GCN features
        x1_flat = x1.view(-1, C)
        x2_flat = x2.view(-1, C)
        proj = nn.Linear(C, 64).to(x1.device)
        x1_proj = proj(x1_flat)
        x2_proj = proj(x2_flat)

        sp_feat1 = torch.mm(self.Q1.t(), x1_proj) / (torch.sum(self.Q1, 0, keepdim=True).t() + 1e-8)
        sp_feat2 = torch.mm(self.Q2.t(), x2_proj) / (torch.sum(self.Q2, 0, keepdim=True).t() + 1e-8)

        gcn_out1 = self.gcn_branch1(sp_feat1, self.edge_index1, self.edge_weight1)
        gcn_out2 = self.gcn_branch2(sp_feat2, self.edge_index2, self.edge_weight2)

        N_sp = gcn_out1.shape[0]
        grid_size = int(math.ceil(math.sqrt(N_sp)))
        gcn_out1 = self.nia1(gcn_out1, grid_size, grid_size)
        gcn_out2 = self.nia2(gcn_out2, grid_size, grid_size)

        gcn_pixel1 = torch.mm(self.Q1, gcn_out1)
        gcn_pixel2 = torch.mm(self.Q2, gcn_out2)

        gcn_pixel1 = gcn_pixel1.view(1, H, W, 64).permute(0, 3, 1, 2)
        gcn_pixel2 = gcn_pixel2.view(1, H, W, 64).permute(0, 3, 1, 2)

        # Fusion
        fused1 = self.msaf(cnn_feat1, gcn_pixel1)
        fused2 = self.msaf(cnn_feat2, gcn_pixel2)

        fused = torch.cat([fused1, fused2], dim=1)
        fused = fused.permute(0, 2, 3, 1).reshape(-1, 128)

        logits = self.classifier(fused)
        out = F.softmax(logits, dim=-1)
        return out

ImprovedNet = M2FNet