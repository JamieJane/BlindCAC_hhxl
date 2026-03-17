import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .network_swinir import RSTB, window_partition, window_reverse
from .fema_utils import ResBlock, CombineQuantBlock, ResBlock_wz 
from .vgg_arch import VGGFeatureExtractor
import numbers
from .arch_util import default_init_weights, make_layer, pixel_unshuffle
from .swinir_arch import *
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, patch_shuffle, patch_unshuffle
# Started by Cursor 10356102 20250307143000000
from .kbnet_arch import KBBlock_s_mobile
# Ended by Cursor 10356102 20250307143000000

# Started by Cursor 10356102 20250311143000000
# ========== NAFBlock Implementation ==========
class SimpleGate(nn.Module):
    """NAFNet中的SimpleGate，替代传统的激活函数"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """NAFNet的基础模块 (Nonlinear Activation Free Block)"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Spatial Mixing
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # Channel Mixing (FFN)
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma
# =============================================
# Ended by Cursor 10356102 20250311143000000

# Import necessary classes from vqcac_arch

class MultiScaleEncoder_light(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                # ResBlock(out_ch, out_ch, norm_type, act_type),
                # ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage: 
            self.blocks.append(SwinLayers(**swin_opts))
            # self.blocks.append(PSALayers(**swin_opts))
            # self.blocks.append(RCABLayers(**swin_opts))
            # upsampler = nn.ModuleList()
            # # for i in range(2):
            #     in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            #     upsampler.append(nn.Sequential(
                #     nn.Upsample(scale_factor=2),
                #     nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     )
                # )
            #     res = res * 2
        
            # self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 use_nafblock=False,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            # Started by Cursor 10356102 20250311143000000
            if use_nafblock:
                tmp_down_block = [
                    nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                    NAFBlock(c=out_ch),
                    NAFBlock(c=out_ch),
                ]
            else:
                tmp_down_block = [
                    nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                ]
            # Ended by Cursor 10356102 20250311143000000
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        # if LQ_stage: 
            # self.blocks.append(SwinLayers(**swin_opts))
            # self.blocks.append(NAFLayers(**swin_opts))
            # self.blocks.append(RRDBLayers(**swin_opts))
            # self.blocks.append(PSALayers(**swin_opts))
            # self.blocks.append(RCABLayers(**swin_opts))
            # upsampler = nn.ModuleList()
            # # for i in range(2):
            #     in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            #     upsampler.append(nn.Sequential(
                #     nn.Upsample(scale_factor=2),
                #     nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     )
                # )
            #     res = res * 2
        
            # self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 use_nafblock=False,
                 ):
        super().__init__()
        # self.use_warp = False
        self.upsampler = nn.ModuleList()
        
        # self.warp = nn.ModuleList()
        res =  input_res // (2 ** max_depth)
        self.maxdepth = max_depth
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            # Started by Cursor 10356102 20250311143000000
            if use_nafblock:
                self.upsampler.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                    NAFBlock(c=out_channel),
                    NAFBlock(c=out_channel),
                ))
            else:
                self.upsampler.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                ))
            # Ended by Cursor 10356102 20250311143000000
            res = res * 2

    def forward(self, input, enc_feature=None):
        x = input
        for idx, m in enumerate(self.upsampler):
            if enc_feature is not None:
                if idx == (self.maxdepth-1):
                    x = m(x)
                else:
                    x = m(x) + enc_feature[idx+1]
            else:
                x = m(x)
        return x

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                blk_depth=6,
                num_heads=8,
                window_size=8,
                num_blk = 4,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(num_blk):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class ChannelAttention(nn.Module):
    """方案5: 通道注意力模块
    
    在SwinLayers基础上添加通道注意力，增强PSF特征与图像特征的交互。
    使用SE-Net风格的通道注意力机制。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # 通道注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # 通道加权
        return x * y.expand_as(x)


class EnhancedSwinLayers(nn.Module):
    """方案5: 增强型Swin Layers
    
    封装SwinLayers + ChannelAttention，提供可选的注意力增强功能。
    """
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                blk_depth=6, num_heads=8, window_size=8, num_blk=4,
                use_attention=True, attention_reduction=16, **kwargs):
        super().__init__()
        self.swin_blks = SwinLayers(
            input_resolution=input_resolution,
            embed_dim=embed_dim,
            blk_depth=blk_depth,
            num_heads=num_heads,
            window_size=window_size,
            num_blk=num_blk,
            **kwargs
        )
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(embed_dim, reduction=attention_reduction)
    
    def forward(self, x):
        x = self.swin_blks(x)
        if self.use_attention:
            x = self.channel_attention(x)
        return x


# Started by Cursor 10356102 20250303143000000
class MultiScaleFusion(nn.Module):
    """方案4: 多尺度特征融合模块
    
    当前架构仅在1/8分辨率进行深度融合，此模块引入多尺度特征融合提升性能。
    通过上采样和卷积实现不同尺度特征的融合。
    注意：输入为多尺度 [cac; psf] 拼接特征，故每尺度通道为 2*单路通道（如 512,512,256）。
    """
    def __init__(self, channels_list, fusion_mode='concat'):
        """
        Args:
            channels_list: 各尺度输入通道数列表（拼接后），如 [512, 512, 256] 表示 3 个尺度分别为 cac+psf 的通道
            fusion_mode: 融合模式，'concat' 或 'add'
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        self.fuse_convs = nn.ModuleList()
        self.channels_list = list(channels_list)
        
        for i, ch in enumerate(self.channels_list):
            if i == 0:
                # 第一个尺度不需要跨尺度融合
                self.fuse_convs.append(nn.Identity())
            else:
                if fusion_mode == 'concat':
                    # 上一尺度输出通道 + 当前尺度通道 -> 当前尺度输出通道（与当前尺度通道一致以便后续使用）
                    prev_ch = self.channels_list[i - 1]
                    out_ch = ch
                    self.fuse_convs.append(
                        nn.Sequential(
                            nn.Conv2d(prev_ch + ch, out_ch, 1, 1, 0),
                            nn.GroupNorm(min(8, out_ch), out_ch),
                            nn.SiLU(inplace=True)
                        )
                    )
                else:  # 'add'
                    self.fuse_convs.append(nn.Identity())
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表，从低分辨率到高分辨率
        Returns:
            fused_features: 融合后的特征列表
        """
        fused = []
        for i, feat in enumerate(features):
            if i == 0:
                fused.append(feat)
            else:
                # 上采样前一尺度特征
                up = F.interpolate(fused[-1], size=feat.shape[2:], mode='bilinear', align_corners=False)
                if self.fusion_mode == 'concat':
                    # 拼接并降维
                    fused_feat = self.fuse_convs[i](torch.cat([feat, up], dim=1))
                else:
                    # 直接相加
                    fused_feat = feat + up
                fused.append(fused_feat)
        return fused
# Ended by Cursor 10356102 20250303143000000


# Started by Cursor 10356102 20250307143000000
class KBNetFusionBlock(nn.Module):
    """KBNet 风格融合模块，替代 SwinLayers，输入/输出 [B, C, H, W] 32×32、256 通道。"""
    def __init__(self, embed_dim=256, num_blocks=4, ffn_scale=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[KBBlock_s_mobile(embed_dim, FFN_Expand=ffn_scale) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)
# Ended by Cursor 10356102 20250307143000000


@ARCH_REGISTRY.register()
class SwinUnet_psfprediction_cacstage(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=4,
                 use_semantic_loss=False,
                 use_residual=True,
                 # ========== 方案2: Swin Transformer配置升级参数 ==========
                 swin_depth=6,           # 每个RSTB的Transformer深度，默认6，升级为8
                 swin_num_heads=8,       # 注意力头数，默认8，升级为12
                 swin_window_size=8,     # 窗口大小，默认8，升级为16
                 swin_num_blk=4,         # RSTB块数量，默认4，升级为6
                 # ========== 方案4: 多尺度特征融合开关 ==========
                 use_multi_scale_fusion=False,
                 # ========== 方案5: 通道注意力增强开关 ==========
                 use_attention_enhance=False,
                 # ========== 模块级替换: KBNet 融合块开关 ==========
                 # Started by Cursor 10356102 20250307143000000
                 use_kbnet_fusion=False,
                 kbnet_num_blocks=4,
                 kbnet_ffn_scale=2,
                 # Ended by Cursor 10356102 20250307143000000
                 use_nafblock=False,
                 channel_query_dict=None,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]


        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual
        
        # 保存升级配置标志
        self.use_multi_scale_fusion = use_multi_scale_fusion
        self.use_attention_enhance = use_attention_enhance

        if channel_query_dict is not None:
            channel_query_dict = dict(channel_query_dict)
        else:
            channel_query_dict = {
                8: 256,
                16: 256,
                32: 256,
                64: 256,
                128: 128,
                256: 64,
                512: 32,
                1024: 16,  # 支持 gt_resolution=1024
            }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))
        self.psf_encoder = MultiScaleEncoder(
                                67,
                                encode_depth,
                                self.gt_res // self.scale_factor,
                                channel_query_dict,
                                norm_type, act_type, False
                            )
        
        self.psfpredict_encoder = MultiScaleEncoder(
                                in_channel,
                                encode_depth,
                                self.gt_res // self.scale_factor,
                                channel_query_dict,
                                norm_type, act_type, False
                            )

        
        self.psffusion_encoder = MultiScaleEncoder_light(
                                67,
                                encode_depth,
                                self.gt_res // self.scale_factor,
                                channel_query_dict,
                                norm_type, act_type, False
                            )
        #                     )

        # Started by Cursor 10356102 20250311143000000
        self.cac_encoder = MultiScaleEncoder(
                                in_channel,
                                encode_depth,
                                self.gt_res // self.scale_factor,
                                channel_query_dict,
                                norm_type, act_type, False,
                                use_nafblock=use_nafblock
                            )
        # Ended by Cursor 10356102 20250311143000000
        
        # ========== 方案2+5+模块级替换: Swin / 增强Swin / KBNet 融合块 ==========
        # Started by Cursor 10356102 20250307143000000
        if use_kbnet_fusion:
            # 模块级替换: 使用 KBNetFusionBlock 替代 SwinLayers
            self.swin_block = KBNetFusionBlock(
                embed_dim=256,
                num_blocks=kbnet_num_blocks,
                ffn_scale=kbnet_ffn_scale
            )
        elif use_attention_enhance:
            # 方案5: 使用带通道注意力的增强版SwinLayers
            self.swin_block = EnhancedSwinLayers(
                input_resolution=(32, 32),
                embed_dim=256,
                blk_depth=swin_depth,
                num_heads=swin_num_heads,
                window_size=swin_window_size,
                num_blk=swin_num_blk,
                use_attention=True
            )
        else:
            # 方案2: 使用标准SwinLayers，但支持配置参数
            self.swin_block = SwinLayers(
                input_resolution=(32, 32),
                embed_dim=256,
                blk_depth=swin_depth,
                num_heads=swin_num_heads,
                window_size=swin_window_size,
                num_blk=swin_num_blk
            )
        # Ended by Cursor 10356102 20250307143000000


        # build decoder
        # self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            # self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        tt = gt_resolution // 2**self.max_depth
        ch_fuse = channel_query_dict[tt]

        self.fuse_conv = nn.Conv2d(ch_fuse*2, ch_fuse, 1)

        # Started by Cursor 10356102 20250311143000000
        self.cacdecoder = MultiScaleDecoder(
                            in_channel,     
                            self.max_depth,  
                            self.gt_res, 
                            channel_query_dict,
                            norm_type, act_type,
                            use_nafblock=use_nafblock
        )
        # Ended by Cursor 10356102 20250311143000000
        self.psfdecoder = MultiScaleDecoder(
                            in_channel,     
                            self.max_depth,  
                            self.gt_res, 
                            channel_query_dict,
                            norm_type, act_type
        )


        self.psf_out_conv = nn.Conv2d(out_ch, 67, 3, 1, 1)
        

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()
        self.cac_out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        
        # ========== 方案4: 多尺度特征融合模块 ==========
        # 在解码器之前添加多尺度融合，增强不同尺度特征的交互
        # 前 3 个尺度通道为 cac+psf 拼接：256+256, 256+256, 128+128 -> [512, 512, 256]
        # Started by Cursor 10356102 20250303143000000
        if self.use_multi_scale_fusion:
            # 融合3个尺度的特征 (8x8, 16x16, 32x32) -> 对应通道 [512, 512, 256]
            self.multi_scale_fusion = MultiScaleFusion(
                channels_list=[512, 512, 256],
                fusion_mode='concat'
            )
            # 融合后的额外卷积层（最深尺度输出为 256 维，与 swin 输入一致）
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.GroupNorm(8, 256),
                nn.SiLU(inplace=True)
            )
        # Ended by Cursor 10356102 20250303143000000


    def encode_and_decode_train(self, input):
        img = input
        psf_feats = self.psfpredict_encoder(input.detach())
        cac_feats = self.cac_encoder(input.detach())
        
        psf_feats = psf_feats[::-1]
        
        cac_feats = cac_feats[::-1]

        before_quant_feat = psf_feats[0]

        x_psfp = before_quant_feat
        x_psfp = self.psfdecoder(x_psfp)

        out_psf = self.psf_out_conv(x_psfp)
        psffusion_feats = self.psffusion_encoder(out_psf)
        psffusion_feats = psffusion_feats[::-1]

        x = cac_feats[0]
        x_psf = psffusion_feats[0]
        x = self.fuse_conv(torch.cat([x, x_psf], dim=1))
        
        # ========== 方案4: 多尺度特征融合 ==========
        if self.use_multi_scale_fusion:
            # 获取多尺度特征进行融合 (取前3个尺度的特征)
            # cac_feats 和 psffusion_feats 都是从低分辨率到高分辨率排列
            multi_scale_feats = []
            for i in range(min(3, len(cac_feats))):
                feat_cac = cac_feats[i]
                feat_psf = psffusion_feats[i] if i < len(psffusion_feats) else psffusion_feats[-1]
                # 融合当前尺度的特征
                fused_feat = torch.cat([feat_cac, feat_psf], dim=1)
                multi_scale_feats.append(fused_feat)
            
            # 应用多尺度融合
            fused_feats = self.multi_scale_fusion(multi_scale_feats)
            # 使用融合后的最深特征
            x = self.fusion_conv(fused_feats[-1])
        
        x = self.swin_block(x)
        x = self.cacdecoder(x)
        out_img = self.cac_out_conv(x)

        return out_img, out_psf


    def encode_and_decode_test(self, input):
        img = input
        psf_feats = self.psfpredict_encoder(input.detach())
        cac_feats = self.cac_encoder(input.detach())
        
        psf_feats = psf_feats[::-1]
        
        cac_feats = cac_feats[::-1]
        before_quant_feat = psf_feats[0]
        x_psfp = before_quant_feat
        x_psfp = self.psfdecoder(x_psfp)

        out_psf = self.psf_out_conv(x_psfp)
        psffusion_feats = self.psffusion_encoder(out_psf)
        psffusion_feats = psffusion_feats[::-1]

        x = cac_feats[0]
        x_psf = psffusion_feats[0]
        x = self.fuse_conv(torch.cat([x, x_psf], dim=1))
        
        # ========== 方案4: 多尺度特征融合 ==========
        if self.use_multi_scale_fusion:
            # 获取多尺度特征进行融合 (取前3个尺度的特征)
            multi_scale_feats = []
            for i in range(min(3, len(cac_feats))):
                feat_cac = cac_feats[i]
                feat_psf = psffusion_feats[i] if i < len(psffusion_feats) else psffusion_feats[-1]
                # 融合当前尺度的特征
                fused_feat = torch.cat([feat_cac, feat_psf], dim=1)
                multi_scale_feats.append(fused_feat)
            
            # 应用多尺度融合
            fused_feats = self.multi_scale_fusion(multi_scale_feats)
            # 使用融合后的最深特征
            x = self.fusion_conv(fused_feats[-1])
        
        x = self.swin_block(x)
        x = self.cacdecoder(x)
        out_img = self.cac_out_conv(x)

        return out_img, out_psf

    @torch.no_grad()
    def test(self, input, weight_alpha=None):

        _, _, h_old, w_old = input.shape

        # output, _ = self.encode_and_decode_test(input, None, None)
        output, _ = self.encode_and_decode_test(input)
        if output is not None:
            output = output[..., :h_old, :w_old]
        # if output_vq is not None:
        #     output_vq = output_vq[..., :h_old, :w_old]

        return output

    def forward(self, input, weight_alpha=None):

        dec, dec_psf = self.encode_and_decode_train(input)

        return dec, dec_psf