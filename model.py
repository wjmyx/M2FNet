import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from gcn_layers_pyg import GraphConvolution


# ==================== Node Interaction Attention (NIA) Module ====================
class NodeInteractionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class AttentionBlock(nn.Module):
    """Global multi-head self-attention block (used inside local window attention)"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LocalWindowAttention(nn.Module):
    """Local window attention for 2D grids (used in MSAF's multi-scale processing)"""
    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.Attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, "B H W C -> B C H W")

        patches = self.unfold(x)
        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        patches = self.norm(patches)
        out = self.Attention(patches)

        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )

        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)

        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))
        norm = fold(norm)
        out = out / norm

        out = rearrange(out, "B C H W -> B H W C")
        return out


class AdaptiveMultiScaleAttention(nn.Module):

    def __init__(
            self,
            in_channels,
            local_attention_kernel_size=3,
            local_attention_stride=1,
            downsampling="conv",
            upsampling="conv",
            sampling_rate=2,
            heads=4,
            dim_head=16,
            dropout=0.1,
            min_size=8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.min_size = min_size

        self.levels = None
        self.current_size = None

        self.attention = LocalWindowAttention(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=in_channels,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        if downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AdaptiveAvgPool2d((None, None)),
                Rearrange("B C H W -> B H W C"),
            )

        if upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode='bilinear', align_corners=True),
                Rearrange("B C H W -> B H W C"),
            )

    def _calculate_levels(self, H, W):
        levels = 0
        h, w = H, W
        while h >= self.min_size and w >= self.min_size:
            levels += 1
            h = h // self.sampling_rate
            w = w // self.sampling_rate
        return max(1, levels)

    def _adaptive_downsample(self, x, target_size):
        B, H, W, C = x.shape
        if (H, W) == target_size:
            return x
        x = rearrange(x, "B H W C -> B C H W")
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x = rearrange(x, "B C H W -> B H W C")
        return x

    def _adaptive_upsample(self, x, target_size):
        B, H, W, C = x.shape
        if (H, W) == target_size:
            return x
        x = rearrange(x, "B H W C -> B C H W")
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x = rearrange(x, "B C H W -> B H W C")
        return x

    def forward(self, x):
        B, H, W, C = x.shape

        if self.levels is None or self.current_size != (H, W):
            self.levels = self._calculate_levels(H, W)
            self.current_size = (H, W)

        level_sizes = []
        h, w = H, W
        for l in range(self.levels):
            level_sizes.append((h, w))
            h = h // self.sampling_rate
            w = w // self.sampling_rate

        x_out = []
        x_current = x
        x_out.append(self.attention(x_current))

        for l in range(1, self.levels):
            target_size = level_sizes[l]
            x_current = self._adaptive_downsample(x_current, target_size)
            x_out_down = self.attention(x_current)
            x_out.append(x_out_down)

        res = x_out[-1]
        for l in range(self.levels - 2, -1, -1):
            target_size = level_sizes[l]
            res = self._adaptive_upsample(res, target_size)
            res = x_out[l] + (1 / (self.levels - l)) * res

        return res


# ==================== Multi-scale Adaptive Fusion (MSAF) Module ====================
class MSAF(nn.Module):
    """Multi-scale Adaptive Fusion module for CNN and GCN features"""
    def __init__(self, channels=64):
        super(MSAF, self).__init__()

        self.multi_scale_attn = AdaptiveMultiScaleAttention(
            in_channels=channels * 2,
            local_attention_kernel_size=3,
            local_attention_stride=1,
            downsampling="conv",
            upsampling="bilinear",
            sampling_rate=2,
            heads=2,
            dim_head=32,
            dropout=0.1,
            min_size=16
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        combined = torch.cat([x1, x2], dim=1)
        combined_permuted = combined.permute(0, 2, 3, 1)
        fused = self.multi_scale_attn(combined_permuted)
        fused = fused.permute(0, 3, 1, 2)
        output = self.fusion_conv(fused)
        return output


# ==================== GCN Branch with NIA ====================
class GCNWithNIA(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=8, pool_ratio=8):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvolution(hidden_channels, hidden_channels))

        self.nia = NodeInteractionAttention(
            dim1=hidden_channels,
            num_heads=num_heads,
            pool_ratio=pool_ratio
        )

        self.relu = nn.ReLU()
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_weight=None):
        H_prev = x
        for i, conv in enumerate(self.convs):
            H = conv(H_prev, edge_index, edge_weight)
            H = self.bn_layers[i](H)
            H = self.relu(H)
            if i > 0:
                H = H + H_prev * torch.sigmoid(H_prev)
            H_prev = H

        num_nodes = H_prev.shape[0]
        grid_size = int(math.sqrt(num_nodes))
        if grid_size * grid_size < num_nodes:
            grid_size += 1

        if grid_size * grid_size > num_nodes:
            padding_size = grid_size * grid_size - num_nodes
            H_padded = torch.cat([
                H_prev,
                torch.zeros(padding_size, H_prev.shape[1], device=H_prev.device)
            ], dim=0)
        else:
            H_padded = H_prev[:grid_size * grid_size]

        H_reshaped = H_padded.unsqueeze(0)
        H_enhanced = self.nia(H_reshaped, grid_size, grid_size)

        if grid_size * grid_size > num_nodes:
            H_final = H_enhanced.squeeze(0)[:num_nodes]
        else:
            H_final = H_enhanced.squeeze(0)

        return H_final


# ==================== Multi-scale Decomposition Perception (MSDP) Module ====================
class MSDP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.gate_conv = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.gate_sigmoid = nn.Sigmoid()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.path1(x)
        feat2 = self.path2(x)
        feat3 = self.path3(x)

        feat_concat = torch.cat([feat1, feat2, feat3], dim=1)
        gate = self.gate_sigmoid(self.gate_conv(feat_concat))

        out = feat1 * gate + feat2 * gate + feat3 * (1 - gate)
        out = self.fusion_conv(out)
        return out


# ==================== M²FNet: Main Network ====================
class M2FNet(nn.Module):

    def __init__(self, height, width, channel, class_count, Q1, edge_index1, edge_weight1, Q2, edge_index2, edge_weight2):
        super(M2FNet, self).__init__()

        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.edge_index1 = edge_index1
        self.edge_weight1 = edge_weight1
        self.Q2 = Q2
        self.edge_index2 = edge_index2
        self.edge_weight2 = edge_weight2

        self.feat = nn.Linear(self.channel, 64)
        self.BN_GCN = nn.BatchNorm1d(64)

        self.gcn_branch = GCNWithNIA(64, 64, num_layers=2, num_heads=8, pool_ratio=8)

        self.cnn_branch1 = MSDP(self.channel, 64)
        self.cnn_branch2 = MSDP(self.channel, 64)

        self.msaf = MSAF(channels=64)

        self.classifier = nn.Linear(128, self.class_count)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_gcn_features(self, x, Q, edge_index, edge_weight):
        h, w, c = x.shape
        x_flatten = x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(Q.t(), x_flatten)
        norm_col_Q = torch.sum(Q, 0, keepdim=True)
        superpixels_flatten = superpixels_flatten / norm_col_Q.t()
        superpixels_flatten = self.feat(superpixels_flatten)
        superpixels_flatten = self.BN_GCN(superpixels_flatten)
        gcn_features = self.gcn_branch(superpixels_flatten, edge_index, edge_weight)
        return gcn_features

    def forward(self, x1, x2):
        # GCN features
        gcn_features1 = self._get_gcn_features(x1, self.Q1, self.edge_index1, self.edge_weight1)
        gcn_features2 = self._get_gcn_features(x2, self.Q2, self.edge_index2, self.edge_weight2)

        GCNout1 = torch.matmul(self.Q1, gcn_features1)
        GCNout2 = torch.matmul(self.Q2, gcn_features2)

        # CNN features (MSDP)
        CNNin1 = torch.unsqueeze(x1.permute([2, 0, 1]), 0)
        CNNin2 = torch.unsqueeze(x2.permute([2, 0, 1]), 0)

        CNNout1 = self.cnn_branch1(CNNin1)
        CNNout2 = self.cnn_branch2(CNNin2)

        CNNout1 = torch.squeeze(CNNout1, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        CNNout2 = torch.squeeze(CNNout2, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        
        CNNout1_4d = CNNout1.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)
        CNNout2_4d = CNNout2.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)
        GCNout1_4d = GCNout1.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)
        GCNout2_4d = GCNout2.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)

        out1 = self.msaf(CNNout1_4d, GCNout1_4d)
        out2 = self.msaf(CNNout2_4d, GCNout2_4d)

        out1 = torch.squeeze(out1, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        out2 = torch.squeeze(out2, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        out = torch.cat([out1, out2], dim=-1)
        out = self.classifier(out)
        out = F.softmax(out, dim=-1)

        return out