import torch
import torch.nn as nn
import torch.nn.functional as F


#Input: [32, 64, 71, 27]

# 아까 만든 MultiHeadAttention 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: [batch_size, seq_len, embed_dim]
        B, T, C = x.size()

        # Linear projections and reshape to [B, num_heads, T, head_dim]
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V  # [B, heads, T, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (T, V) → (1, 1)
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: [B, C, T, V] → pooled → [B, C] → regression → [B, 1]
        """
        x = self.pool(x)             # [B, C, 1, 1]
        x = x.view(x.size(0), -1)    # [B, C]
        out = self.regressor(x)      # [B, 1]
        return out


class TA_Wraper(nn.Module):
    def __init__(self,
                 in_channels=64,
                 embed_dim=64,
                 num_heads=8,
                 hidden_dim=64):
        super().__init__()
        self.reg_head = RegressionHead(in_channels=in_channels, hidden_dim=hidden_dim)
        self.temporal_attention= MultiHeadAttention(embed_dim=in_channels, num_heads=num_heads)

    def forward(self, x):
        B, C, T, V = x.shape

        # 1. reshape: [B, C, T, V] → [B, V, T, C]
        x = x.permute(0, 3, 2, 1).contiguous()
        # 2. reshape for attention: [B * V, T, C]
        x = x.view(B * V, T, C)

        x_TA = self.temporal_attention(x)
        x_TA = x_TA.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()  # [B, C, T, V]
        attention_score = self.reg_head(x_TA)

        return attention_score