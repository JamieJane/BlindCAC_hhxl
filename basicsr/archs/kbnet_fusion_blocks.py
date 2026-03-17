# Started by Cursor 10356102 20250307153000000
"""
KBNet 融合块模块，包含多种 KBBlock 变体：
- KBBlock_s_mobile: 原始轻量版 KBBlock
- KBBlock_v5: 融合 MobileNetV5 设计理念的 KBBlock
- KBBlock_v5_Light: 轻量版 KBBlock_v5

不依赖 basicsr.archs.common，用于 SwinUnet 模块级替换。

更新记录:
- 20250307: 初始版本，包含 KBBlock_s_mobile
- 20250309: 添加 KBBlock_v5, KBBlock_v5_Light, RmsNorm2d, UniversalInvertedResidual_simplified
"""
import torch
import torch.nn as nn
from typing import Optional
# Ended by Cursor 10356102 20250307153000000


# ==================== 归一化层 ====================

# Started by Cursor 10356102 20250307153000000
class LayerNorm2d(nn.Module):
    """层归一化层 - 用于 KBBlock_s_mobile
    
    Args:
        channels: 输入通道数
        eps: 数值稳定性常数
        requires_grad: 权重和偏置是否可训练
    """
    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.c = channels
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, self.c, 1, 1) * y + self.bias.view(1, self.c, 1, 1)
        return y


class RmsNorm2d(nn.Module):
    """RMS 归一化层 - 从 MobileNetV5 引入，用于 KBBlock_v5
    
    相比 LayerNorm2d 的优势：
    - 计算量更低（仅计算方差，不计算均值）
    - 训练稳定性更好
    - 推理速度更快
    
    Args:
        channels: 输入通道数
        eps: 数值稳定性常数
    """
    
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        variance = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = x / (variance + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


# ==================== 激活与门控 ====================

class SimpleGate(nn.Module):
    """简单门控模块
    
    将输入在通道维度分成两部分，返回它们的逐元素乘积。
    这是一种高效的非线性激活方式，常用于 FFN 中。
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ==================== UIR 模块 ====================

class UniversalInvertedResidual_simplified(nn.Module):
    """简化版通用倒残差块 (UIR) - 适配 KBBlock_v5
    
    基于 MobileNetV5 的 UniversalInvertedResidual 设计，用于增强 FFN 部分。
    
    结构：
        输入 → Expand(1x1) → Norm → Act → DWConv → Norm → Act → Project(1x1) → Norm → Layer Scale → 残差连接
    
    Args:
        in_chs: 输入通道数
        out_chs: 输出通道数
        dw_kernel_size: 深度卷积核大小，0 表示不使用深度卷积
        exp_ratio: 扩展比例
        layer_scale_init_value: Layer Scale 初始值，None 表示不使用
        norm_layer: 归一化层类型
        act_layer: 激活层类型
    """
    
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        exp_ratio: float = 2.0,
        layer_scale_init_value: Optional[float] = None,
        norm_layer: Optional[nn.Module] = None,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        mid_chs = int(in_chs * exp_ratio)
        
        # 默认使用 RmsNorm2d
        if norm_layer is None:
            norm_layer = RmsNorm2d
        
        # 扩展层 (Expansion)
        self.expand_conv = nn.Conv2d(in_chs, mid_chs, kernel_size=1, bias=True)
        self.expand_norm = norm_layer(mid_chs)
        self.expand_act = act_layer()
        
        # 深度卷积 (Depthwise) - 可选
        if dw_kernel_size > 0:
            self.dw_conv = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size,
                padding=dw_kernel_size // 2, groups=mid_chs, bias=True
            )
            self.dw_norm = norm_layer(mid_chs)
            self.dw_act = act_layer()
        else:
            self.dw_conv = None
        
        # 投影层 (Projection)
        self.project_conv = nn.Conv2d(mid_chs, out_chs, kernel_size=1, bias=True)
        self.project_norm = norm_layer(out_chs)
        
        # Layer Scale
        if layer_scale_init_value is not None:
            self.layer_scale = nn.Parameter(
                torch.ones(out_chs, 1, 1) * layer_scale_init_value
            )
        else:
            self.layer_scale = None
        
        # 残差连接条件
        self.use_residual = (in_chs == out_chs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 扩展
        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)
        
        # 深度卷积
        if self.dw_conv is not None:
            x = self.dw_conv(x)
            x = self.dw_norm(x)
            x = self.dw_act(x)
        
        # 投影
        x = self.project_conv(x)
        x = self.project_norm(x)
        
        # Layer Scale
        if self.layer_scale is not None:
            x = x * self.layer_scale
        
        # 残差连接
        if self.use_residual:
            x = x + identity
        
        return x


# ==================== KBBlock 变体 ====================

class KBBlock_s_mobile(nn.Module):
    """KBNet 轻量块 - 原始版本
    
    仅用 PyTorch 与 LayerNorm2d/SimpleGate，不依赖 common。
    
    Args:
        c: 输入通道数
        DW_Expand: 深度卷积扩展比例
        FFN_Expand: FFN 扩展比例
        nset: 注意力集合数
        lightweight: 是否使用轻量模式
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, lightweight=False):
        super(KBBlock_s_mobile, self).__init__()
        self.c = c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        self.kba = nn.Sequential(
                nn.Conv2d(in_channels=c + self.nset, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 * sca
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class KBBlock_v5(nn.Module):
    """融合 MobileNetV5 设计理念的 KBBlock
    
    主要改进：
    1. RmsNorm2d 替代 LayerNorm2d - 提升训练稳定性
    2. UIR 结构增强 FFN 部分 - 更高效的特征变换
    3. Layer Scale 提升训练稳定性
    4. 保留 KBA 核心注意力机制
    
    架构流程：
        输入 (inp) [B, C, H, W]
            │
            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ RmsNorm2d (norm1)                                           │
        │     │                                                       │
        │     ├──► SCA 分支                                           │
        │     │    AdaptiveAvgPool2d(1) → Conv1x1                    │
        │     │                                                       │
        │     ├──► Conv11 分支                                        │
        │     │    Conv1x1 → DWConv5x5 (groups=c//4)                  │
        │     │                                                       │
        │     └──► KBA 模块                                           │
        │         att = conv2(x) * attgamma + conv211(x)              │
        │         uf = conv21(conv1(x))                               │
        │         x = kba(cat[att, uf]) * ga1 + uf                    │
        │                                                               │
        │     x = x * x1 * sca                                         │
        │     x = conv3(x)                                             │
        │     y = inp + x * beta                                       │
        └─────────────────────────────────────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ RmsNorm2d (norm2)                                           │
        │     │                                                       │
        │     └──► UIR-FFN (新设计)                                    │
        │         │                                                   │
        │         ├── Expand: Conv1x1 (c → ffn_ch)                    │
        │         ├── RmsNorm2d + GELU                                │
        │         ├── DWConv (可选)                                    │
        │         ├── Project: Conv1x1 (ffn_ch//2 → c)               │
        │         └── Layer Scale                                     │
        │                                                               │
        │     return y + x (残差连接)                                  │
        └─────────────────────────────────────────────────────────────┘
    
    Args:
        c: 输入通道数
        DW_Expand: 深度卷积扩展比例，默认 2
        FFN_Expand: FFN 扩展比例，默认 2.0
        nset: 注意力集合数，默认 32
        lightweight: 轻量模式标志，默认 False
        layer_scale_init_value: Layer Scale 初始值，默认 1e-5
        use_uir_ffn: 是否使用 UIR 结构作为 FFN，默认 True
    """
    
    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        FFN_Expand: float = 2.0,
        nset: int = 32,
        lightweight: bool = False,
        layer_scale_init_value: float = 1e-5,
        use_uir_ffn: bool = True,
    ):
        super().__init__()
        self.c = c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)
        
        # ========== 归一化层 (使用 RmsNorm2d) ==========
        self.norm1 = RmsNorm2d(c)
        self.norm2 = RmsNorm2d(c)
        
        # ========== SCA 模块 (保留原有设计) ==========
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
        )
        
        # ========== Conv11 分支 ==========
        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c // 4, bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=True),
                nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True),
            )
        
        # ========== KBA 模块 (保留核心设计) ==========
        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.conv21 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        
        self.kba = nn.Sequential(
            nn.Conv2d(c + nset, c, kernel_size=1, bias=True),
            nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c // 4, bias=True),
        )
        
        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, interc, kernel_size=3, padding=1, groups=interc, bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, nset, kernel_size=1, bias=True),
        )
        self.conv211 = nn.Conv2d(c, nset, kernel_size=1, bias=True)
        
        # ========== 输出投影 ==========
        # conv3 输入通道始终为 c，因为 kba 输出是 c 通道
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        
        # ========== FFN 部分 (使用 UIR 结构) ==========
        if use_uir_ffn:
            self.ffn = UniversalInvertedResidual_simplified(
                in_chs=c,
                out_chs=c,
                dw_kernel_size=3,
                exp_ratio=FFN_Expand,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            # 传统 FFN
            self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
            self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
            self.sg = SimpleGate()
        
        # ========== 可学习参数 ==========
        self.ga1 = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)
        self.attgamma = nn.Parameter(torch.zeros(1, nset, 1, 1) + 1e-2)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1) + 1e-2)
        
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
    
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.norm1(x)
        
        # SCA 分支
        sca = self.sca(x)
        x1 = self.conv11(x)
        
        # KBA 模块
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 * sca
        
        # 输出投影
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        # FFN
        x = self.norm2(y)
        if hasattr(self, 'ffn'):
            x = self.ffn(x)
        else:
            x = self.conv4(x)
            x = self.sg(x)
            x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class KBBlock_v5_Light(KBBlock_v5):
    """轻量版 KBBlock_v5
    
    使用 lightweight=True 和较小的扩展比例，适合移动端部署。
    
    Args:
        c: 输入通道数
        DW_Expand: 深度卷积扩展比例，默认 1
        FFN_Expand: FFN 扩展比例，默认 1.5
        nset: 注意力集合数，默认 16
        layer_scale_init_value: Layer Scale 初始值，默认 1e-5
        use_uir_ffn: 是否使用 UIR 结构作为 FFN，默认 True
    """
    
    def __init__(
        self,
        c: int,
        DW_Expand: int = 1,
        FFN_Expand: float = 1.5,
        nset: int = 16,
        layer_scale_init_value: float = 1e-5,
        use_uir_ffn: bool = True,
    ):
        super().__init__(
            c=c,
            DW_Expand=DW_Expand,
            FFN_Expand=FFN_Expand,
            nset=nset,
            lightweight=True,
            layer_scale_init_value=layer_scale_init_value,
            use_uir_ffn=use_uir_ffn,
        )


# ==================== 工厂函数 ====================

def create_kbblock(
    block_type: str,
    c: int,
    **kwargs
) -> nn.Module:
    """创建 KBBlock 实例的工厂函数
    
    Args:
        block_type: 块类型，可选 'mobile', 'v5', 'v5_light'
        c: 输入通道数
        **kwargs: 其他参数传递给对应的 KBBlock
    
    Returns:
        对应的 KBBlock 实例
    
    Raises:
        ValueError: 当 block_type 不是有效值时
    """
    if block_type == 'mobile':
        return KBBlock_s_mobile(c=c, **kwargs)
    elif block_type == 'v5':
        return KBBlock_v5(c=c, **kwargs)
    elif block_type == 'v5_light':
        return KBBlock_v5_Light(c=c, **kwargs)
    else:
        raise ValueError(f"Unknown block_type: {block_type}. "
                        f"Available options: 'mobile', 'v5', 'v5_light'")


# ==================== 导出列表 ====================

__all__ = [
    # 归一化层
    'LayerNorm2d',
    'RmsNorm2d',
    # 激活与门控
    'SimpleGate',
    # UIR 模块
    'UniversalInvertedResidual_simplified',
    # KBBlock 变体
    'KBBlock_s_mobile',
    'KBBlock_v5',
    'KBBlock_v5_Light',
    # 工厂函数
    'create_kbblock',
]
# Ended by Cursor 10356102 20250309154000000
