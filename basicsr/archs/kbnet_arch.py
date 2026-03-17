import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from basicsr.utils.registry import ARCH_REGISTRY

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.LayerNormFunction
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

import matplotlib.pyplot as plt
def keshihua(x,name):
        # x = torch.mean(x, dim=1)
        x=x[0,0,:,:]
        x=x.cpu().detach().numpy()
        # x = (x - x.min()) / (x.max() - x.min())
        # plt.imshow(x, cmap='seismic', vmin=-1, vmax=1)
        fig, ax = plt.subplots()
        ax.imshow(x, cmap='seismic', vmin=-1, vmax=1)
        ax.axis('off')  # 关闭坐标轴
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.close()
class SPAB4(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False
                 ):
        super(SPAB4, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = nn.Conv2d(in_channels, mid_channels, 3, 1,1)
        self.c3_r = nn.Conv2d(mid_channels, out_channels, 3,1,1)
        self.act1 = torch.nn.ReLU(inplace=True)
        # self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out3 = self.c3_r(out1_act)

        sim_att = 1-torch.sigmoid(out3)
        keshihua(sim_att,"tmp_show/feature/sim_att.jpg")
        out = (out3 + x) * sim_att
        keshihua(x,"tmp_show/feature/x.jpg")
        keshihua(sim_att,"tmp_show/feature/out3.jpg")

        return out
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class SMFA2(nn.Module):
    def __init__(self, dim=36):
        super(SMFA2, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)
        self.lde2 = DMlp(dim,2)
        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_l=self.lde2(x)
        # x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        # x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        # x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class SMFA3(nn.Module):
    def __init__(self, dim=36):
        super(SMFA3, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)
        self.lde2 = DMlp(dim,2)
        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        x = self.linear_0(f)
        # x_l=self.lde2(x)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        # y_d = self.lde(y)
        return self.linear_2(x_l )
class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x
    
class FMB2(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA2(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x
    
class FMB3(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA3(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y


class LayerNorm2d(nn.Module):

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
        y =self.weight.view(1, self.c, 1, 1) * y + self.bias.view(1, self.c, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


# class LayerNorm2d(nn.Module):

#     def __init__(self, channels, eps=1e-6, requires_grad=True):
#         super(LayerNorm2d, self).__init__()
#         self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
#         self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
#         self.eps = eps

#     def forward(self, x):
#         return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBBlock_s_mobile(nn.Module):
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

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))

        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        # keshihua(x,"tmp_show/feature/x.jpg")
        # keshihua(att,"tmp_show/feature/att.jpg")
        # keshihua(y + x * self.gamma,"tmp_show/feature/y.jpg")
        return y + x * self.gamma

class KBBlock_s_mobile_v4(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, lightweight=False):
        super(KBBlock_s_mobile_v4, self).__init__()
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
        self.sca1 = nn.Sequential(
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

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))

        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        sca1=self.sca1(x)
        x = self.conv4(x)
        x = self.sg(x)*sca1
        x = self.conv5(x)

        x = self.dropout2(x)
        # keshihua(x,"tmp_show/feature/x.jpg")
        # keshihua(att,"tmp_show/feature/att.jpg")
        # keshihua(y + x * self.gamma,"tmp_show/feature/y.jpg")
        return y + x * self.gamma

class KBBlock_s_mobile_v2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, lightweight=False):
        super(KBBlock_s_mobile_v2, self).__init__()
        self.c = c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

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
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
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
        x=self.norm1(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))

        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(y)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

class KBBlock_s_mobile_v3(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, lightweight=False):
        super(KBBlock_s_mobile_v3, self).__init__()
        self.c = c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)
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
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))

        x = self.kba(torch.cat([att, uf], dim=1)) * self.ga1 + uf
        x = x * x1 

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        # FFN
        # x = self.norm2(y)
        return y 
# @ARCH_REGISTRY.register()
class KBNet_s_mobile(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s_mobile', lightweight=False, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)

        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.conv_up1 = nn.Conv2d(width, width, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(width, width, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_last = nn.Conv2d(width, 3, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(width, width, 3, 1, 1)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp_ = self.pixelunshuffle(inp)
        x = self.intro(inp_)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print(x.shape)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        feat = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(feat)))
        inp=F.interpolate(inp, scale_factor=2, mode='nearest')
        # x = self.ending(x)
        # x = self.pixelshuffle(x)

        x = x + inp

        return x

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


# class DWConv(nn.Layer):
#     def __init__(self, dim=768):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x):
#         x = self.dwconv(x)
#         return x
class NAFBlock2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, c,1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(c, c,1)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)
        self.spatial_gating_unit = LKA(c)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.spatial_gating_unit(self.activation(x))
        x = self.conv2(x)
        # x = self.sg(x)
        x = x * self.sca(x)
        x = x + inp
        # x = self.conv3(x)
        # x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma
# @ARCH_REGISTRY.register()
class KBNet_small_mobile_1(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num=8, enc_blk_nums=[1, 2, 2],
                 dec_blk_nums=[2, 2, 1, 1], basicblock='NAFBlock', lightweight=False, ffn_scale=2):
        super().__init__()
        # 1 2 6
        basicblock = eval(basicblock)
        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.intro_0 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.intro = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=5, padding=2, stride=2, groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending_0 = nn.Conv2d(in_channels=16, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                  bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pixelshuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
            )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )

    def forward(self, inp):

        inp_ = self.pixelunshuffle(inp)
        x = self.intro_0(inp_)
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        for decoder, up in zip(self.decoders[:3], self.ups[:3]):
            x = up(x)
            x = decoder(x)
          
        x = self.ups[3](x)
        x = self.ending_0(x)
        x = self.ending(x)
        x = self.pixelshuffle(x)
        
        return x



class KBNet_small_mobile_2(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num=8, enc_blk_nums=[1, 1, 4],
                 dec_blk_nums=[4, 1, 1, 1], basicblock='NAFBlock', lightweight=False, ffn_scale=2):
        super().__init__()
        # 1 2 6
        basicblock = eval(basicblock)
        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.intro_0 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.intro = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=5, padding=2, stride=2, groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending_0 = nn.Conv2d(in_channels=16, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                  bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pixelshuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
            )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )

    def forward(self, inp):

        inp_ = self.pixelunshuffle(inp)
        x = self.intro_0(inp_)
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        for decoder, up in zip(self.decoders[:3], self.ups[:3]):
            x = up(x)
            x = decoder(x)
          
        x = self.ups[3](x)
        x = self.ending_0(x)
        x = self.ending(x)
        x = self.pixelshuffle(x)
        
        return x


@ARCH_REGISTRY.register()
class KBNet(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s_mobile', lightweight=False, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)

        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pixelshuffle = nn.PixelShuffle(2)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** (len(self.encoders)+1)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # return inp
        inp_ = self.pixelunshuffle(inp)
        x = self.intro(inp_)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # print(x.shape)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_
        x = self.pixelshuffle(x)


        return x

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



# @ARCH_REGISTRY.register()
class KBNet_s_student_mobile(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num=8, enc_blk_nums=[1, 2, 6],
                 dec_blk_nums=[6, 2, 1, 1], basicblock='NAFBlock', lightweight=False, ffn_scale=2):
        super().__init__()
        # 1 2 6
        basicblock = eval(basicblock)
        self.pixelunshuffle = nn.PixelUnshuffle(2)
        self.intro_0 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.intro = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=5, padding=2, stride=2, groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending_0 = nn.Conv2d(in_channels=16, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                  bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pixelshuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num)]
            )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** (len(self.decoders)+1)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp_ = self.pixelunshuffle(inp)

        x_0 = self.intro_0(inp_)
        x = self.intro(x_0)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders[:3], self.ups[:3], encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ups[3](x)
        x = self.ending_0(x)
        x = x + x_0

        x = self.ending(x)
        x = self.pixelshuffle(x)

        x = x + inp

        return x[:, :, :H, :W]


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
 

@ARCH_REGISTRY.register()
class KBNet_s_student_mobile_v2(nn.Module):
    def __init__(self, img_channel=12, width=32, middle_blk_num_v2=8, enc_blk_nums_v2=[1, 2, 6],
                 dec_blk_nums_v2=[6, 2, 1, 1], basicblock_v2='NAFBlock', lightweight=False, ffn_scale=2):
        super().__init__()
        # 1 2 6
        basicblock_v2 = eval(basicblock_v2)
        self.v2_pixelunshuffle = nn.PixelUnshuffle(2)
        # 重命名intro_0为v2_intro_0
        self.v2_intro_0 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                    groups=1, bias=True)
        # 重命名intro为v2_intro
        self.v2_intro = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=5, padding=2, stride=2, groups=1,
                                  bias=True)

        self.v2_encoders = nn.ModuleList()
        self.v2_middle_blks = nn.ModuleList()
        self.v2_decoders = nn.ModuleList()

        self.v2_ending_0 = nn.Conv2d(in_channels=16, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                     bias=True)
        self.v2_ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                   groups=1, bias=True)

        self.v2_ups = nn.ModuleList()
        self.v2_downs = nn.ModuleList()
        self.pixelshuffle = nn.PixelShuffle(2)
        # self.act = nn.ReLU()
        # self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        chan = width
        for num in enc_blk_nums_v2:
            self.v2_encoders.append(
                nn.Sequential(
                    *[basicblock_v2(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
            self.v2_downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        self.v2_middle_blks = \
            nn.Sequential(
                *[basicblock_v2(chan, FFN_Expand=ffn_scale) for _ in range(middle_blk_num_v2)]
            )
        for num in dec_blk_nums_v2:
            self.v2_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.v2_decoders.append(
                nn.Sequential(
                    *[basicblock_v2(chan, FFN_Expand=ffn_scale) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** (len(self.v2_decoders)+1)     

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # return inp
        # print(inp.shape)
        inp = self.v2_pixelunshuffle(inp)
        # print(self.padder_size)

        x_0 = self.v2_intro_0(inp)
        x = self.v2_intro(x_0)
        encs = []

        for encoder, down in zip(self.v2_encoders, self.v2_downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.v2_middle_blks(x)
        for decoder, up, enc_skip in zip(self.v2_decoders[:3], self.v2_ups[:3], encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.v2_ups[3](x)
        x = self.v2_ending_0(x)
        sim_att = torch.sigmoid(x) - 0.5
        x = (x_0 + x) * sim_att
        # x = x + x_0

        x = self.v2_ending(x)
        x = x + inp
        x = self.pixelshuffle(x)
        # sim_att = torch.sigmoid(x) - 0.5
        # x = (inp + x) * sim_att
        # x = x + inp

        return x
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# @ARCH_REGISTRY.register()
class NAFNet(nn.Module):
    def __init__(self,
                 img_channel=12,
                 width=32,
                 middle_blk_num=1,
                 enc_blk_nums=[1, 3, 4, 0],
                 dec_blk_nums=[1, 1, 1, 0]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel,
                               out_channels=width,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width,
                                out_channels=img_channel,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
            self.encoders.append(
                nn.Sequential(*[NAFBlock2(chan) for _ in range(num)]))
           

        # self.middle_blks = \
        #     nn.Sequential(
        #         *[Block(chan) for _ in range(middle_blk_num)] 
        #     )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.pixelunshuffle = nn.PixelUnshuffle(2)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_ = self.pixelunshuffle(inp)


        x = self.intro(inp_)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            encs.append(x)
            x = down(x)
            x = encoder(x)

        # x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_
        x = self.pixelshuffle(x)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, [0, mod_pad_w, 0, mod_pad_h])
        return x

if __name__ == '__main__':
    # dummy_inp = torch.randn(1, 3, 1024, 1024)
    import cv2
    import numpy as np
    from basicsr.utils import img2tensor
    convert = True
    convert = False
    path='datasets/udc_tmp/Image00001_kbnet_s_mobile_44.jpg'
    img_lq = cv2.imread(path).astype(np.float32)
    dummy_inp = img_lq / 255.
    dummy_inp = img2tensor(dummy_inp, bgr2rgb=True, float32=True)
    # dummy_inp = torch.randn(1, inp_feat, inp_size, inp_size)
    dummy_inp = dummy_inp.unsqueeze(0).contiguous()
    # dummy_inp = F.pixel_unshuffle(dummy_inp, 2)
    # net = KBNet2()
    # model_path='experiments/KBNet2_1004_s2/models/net_g_13000.pth'
    # state = torch.load(model_path, map_location='cpu')['params']
    # net.load_state_dict(state, strict=True)
    net = KBNet()
    for name, module in net.named_modules():
        if hasattr(module, "convert"):
            module.convert()
    # model_path='experiments/KBNet_s_student_mobile_v2_s1/models/net_g_2000.pth'
    # state1 = torch.load(model_path, map_location='cpu')['params_ema'] #['state_dict']
    # net.load_state_dict(state1, strict=False)
    # net.cpu().eval()
    # with torch.no_grad():
    #     tmp=net(dummy_inp)
    from thop import profile
    if convert:
        net.cpu().eval()
        model_name = "KBNet_s_student_mobile_v2"
        # traced_model = torch.jit.trace(combined_model, dummy_inp)

    # # 保存 TorchScript 模型
        # traced_model.save(f"onnx_models/{model_name}.pt")
        with torch.no_grad():
            torch.onnx.export(
                net,
                dummy_inp,
                f"onnx_models/{model_name}.onnx",
                opset_version=11,
                input_names=['input'],
                output_names=['output_P'])

    if not convert:
        dummy_inp = torch.randn(1, 3, 1024, 1024)
        flops, params = profile(net, inputs=(dummy_inp,))
        print('FLOPs: %.3f GFLOPs' % (flops / 1e9))
        print('Params: %.3f M' % (params / 1e6))

    # macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
    # print(macs)
    # print(params)


    # x = torch.rand(1, 3, 256, 256)
    # out = model(x)
    # print(out.shape)
# KBNet_small_mobile_1
# FLOPs: 13.125 GFLOPs
# Params: 4.719 M

# KBNet_small_mobile_2
# FLOPs: 13.930 GFLOPs
# Params: 5.131 M
    # 276.29GMac
    # 141.97M
