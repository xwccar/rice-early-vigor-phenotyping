import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, input_dim=3, use_time=True, use_attention=False,
                 use_residual=False, activation="relu", use_feat_norm=False):
        super(PointNet, self).__init__()
        self.use_time = use_time
        self.use_attention = use_attention
        self.use_residual = use_residual

        # ===== 激活函数 =====
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # ===== 特征归一化方式 =====
        def norm_layer(out_dim):
            return nn.LayerNorm(out_dim) if use_feat_norm else nn.Identity()

        # ===== 点云特征提取层 =====
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            norm_layer(64),
            self.activation_fn
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            norm_layer(128),
            self.activation_fn
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            norm_layer(1024),
            self.activation_fn
        )

        # ===== 时间特征：放大 + 升维到 64 维 =====
        if self.use_time:
            self.time_scale = 10.0          # 放大时间特征影响力，可根据需要调整
            self.time_proj = nn.Linear(1, 64)
            time_feat_dim = 64
        else:
            self.time_scale = 1.0
            self.time_proj = None
            time_feat_dim = 0

        # ===== 注意力层（可选），输入：1024 + 时间嵌入 =====
        if self.use_attention:
            self.attn_fc = nn.Sequential(
                nn.Linear(1024 + time_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # ===== 回归器：输入 1024 + 时间嵌入 =====
        regressor_input_dim = 1024 + time_feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, time=None):
        # x: [B, N, D] -> [B, D, N]
        x = x.transpose(2, 1)  # [B, D, N]

        # ===== 点云特征提取 =====
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)

        # ===== 可选残差连接 =====
        if self.use_residual:
            if x2.shape[1] != x3.shape[1]:
                # 通道不一致，用 1x1 卷积投影
                proj = nn.Conv1d(x2.shape[1], x3.shape[1], 1).to(x.device)
                x2_proj = proj(x2)
                x3 = x3 + x2_proj
            else:
                x3 = x3 + x2

        # 全局池化 -> [B, 1024]
        x_global = torch.max(x3, 2)[0]

        # ===== 时间特征处理：放大 + 升维到 64 维 =====
        if self.use_time:
            # 确保 shape 为 [B, 1]
            if time is None:
                raise ValueError("use_time=True 但 forward 没有传入 time 张量")
            if time.dim() == 1:
                time = time.unsqueeze(1)  # [B] -> [B, 1]

            # 放大时间特征数值尺度
            t_scaled = time * self.time_scale        # [B, 1]

            # 升维到 64 维嵌入
            t_emb = self.time_proj(t_scaled)         # [B, 64]
        else:
            t_emb = None

        # ===== 注意力：用时间嵌入一起调节全局点云特征 =====
        if self.use_attention:
            if self.use_time:
                x_aug = torch.cat([x_global, t_emb], dim=1)  # [B, 1024+64]
            else:
                x_aug = x_global                             # [B, 1024]
            attn_weight = torch.sigmoid(self.attn_fc(x_aug))  # [B, 1]
            x_global = x_global * attn_weight

        # ===== 最终拼接：全局点云特征 + 时间嵌入 =====
        if self.use_time:
            x_final = torch.cat([x_global, t_emb], dim=1)   # [B, 1024+64]
        else:
            x_final = x_global                              # [B, 1024]

        # ===== 回归输出 =====
        out = self.regressor(x_final)  # [B, 1]
        return out
