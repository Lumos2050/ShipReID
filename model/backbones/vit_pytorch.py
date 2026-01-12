""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch._six import container_abcs
from collections.abc import Iterable


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

"""
default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}
"""
def orthogonality_loss(qx, qtx):
    """
    qx, qtx: [B, num_heads, N, C]
    约束每个 head 的 qx 和 qtx 正交
    """
    B, H, N, C = qx.shape
    qx = F.normalize(qx, dim=-1)
    qtx = F.normalize(qtx, dim=-1)

    qx_flat = qx.reshape(B * H, N, C)
    qtx_flat = qtx.reshape(B * H, N, C)

    prod = torch.bmm(qx_flat, qtx_flat.transpose(1, 2))  # [B*H, N, N]

    eye = torch.eye(N, device=prod.device).unsqueeze(0).expand(B*H, -1, -1)
    off_diag = prod * (1 - eye)

    loss = torch.norm(off_diag, p='fro')**2 / (B*N*N)
    return loss
def pairwise_feature_distance(xc, yc, metric='l2'):
    """
    计算 batch 内所有 (xc_i, yc_i) 的 pair-wise 距离平均
    :param xc: Tensor (B, N, C) 
    :param yc: Tensor (B, N, C) 
    :param metric: 'l2' (欧氏距离) 或 'cosine' (1 - cos 相似度)
    :return: scalar, 平均距离
    """
    # 归一化最后一维特征
    xc = F.normalize(xc, dim=-1)
    yc = F.normalize(yc, dim=-1)

    if metric == 'l2':
        # L2 距离：对最后一维求范数 (B, N)
        distances = torch.norm(xc - yc, p=2, dim=-1)
    elif metric == 'cosine':
        # 余弦距离：逐token计算 (B, N)
        cos_sim = F.cosine_similarity(xc, yc, dim=-1)
        distances = 1 - cos_sim
    else:
        raise ValueError("Unsupported metric. Use 'l2' or 'cosine'.")

    # 对 batch 和 token 求平均
    return distances.mean()
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Conv_Attention(nn.Module):
    """
    对数据源进行conv+注意力+downsample的一连串操作
    Input: x: (B, H*W, C), H, W
    Output: z: (B, H/2*W/2, 2C)
    """
    def __init__(self, dim, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        out_features = out_features or dim
        hidden_features = hidden_features or dim
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: z: (B, H*W, C), H, W
        """
        B, L, C = x.shape
        #assert L == H * W, "flatten img_tokens has wrong size"
        H = int(L**(0.5))
        W = H
        
        # convolution output
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        conv_x = self.conv1(x)#conv_x(B, C, H, W)
        
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(x).permute(0, 2, 3, 1).contiguous().view(B, L, 1)
        channel_map = self.channel_interaction(x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        conv_x = conv_x + conv_x * torch.sigmoid(spatial_map)
        conv_x = conv_x.transpose(-2,-1).contiguous().view(B, C, H, W)

        conv_x = self.conv2(conv_x)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        conv_x = conv_x + conv_x * torch.sigmoid(channel_map)

        conv_x = self.fc1(conv_x)
        conv_x = self.act(conv_x)
        conv_x = self.drop(conv_x)
        conv_x = self.fc2(conv_x)
        conv_x = self.drop(conv_x)
        return conv_x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(dim, dim)
        self.proj4 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x=None, y=None, mode = 'train'):  # "train", "test-x", "test-y", "test-xy")
        B, N, C = (x.shape if x is not None else y.shape)  

        if mode == "train":
            qkv1 = self.qkv1(x).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv2 = self.qkv2(y).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qx, kx, vx, qtx = qkv1[0], qkv1[1], qkv1[2], qkv1[3]   # make torchscript happy (cannot use tensor as tuple)
            qy, ky, vy, qty = qkv2[0], qkv2[1], qkv2[2], qkv2[3]

            attncy = (qx @ ky.transpose(-2, -1)) * self.scale
            attncx = (qy @ kx.transpose(-2, -1)) * self.scale
            attntx = (qtx @ kx.transpose(-2, -1)) * self.scale
            attnty = (qty @ ky.transpose(-2, -1)) * self.scale
            lorthx = orthogonality_loss(qx, qtx)
            lorthy = orthogonality_loss(qy, qty)
            attncx = attncx.softmax(dim=-1)
            attncy = attncy.softmax(dim=-1)
            attncx = self.attn_drop(attncx)
            attncy = self.attn_drop(attncy)

            xc = (attncx @ vx).transpose(1, 2).reshape(B, N, C)
            yc = (attncy @ vy).transpose(1, 2).reshape(B, N, C)
            xt = (attntx @ vx).transpose(1, 2).reshape(B, N, C)
            yt = (attnty @ vy).transpose(1, 2).reshape(B, N, C)
            xc = self.proj1(xc)
            yc = self.proj2(yc)
            xt = self.proj3(xt)
            yt = self.proj4(yt)
            xc = self.proj_drop(xc)
            yc = self.proj_drop(yc)
            xt = self.proj_drop(xt)
            yt = self.proj_drop(yt)
            lc = pairwise_feature_distance(xc, yc, 'cosine')
            return xc, yc, xt, yt, lorthx, lorthy, lc
        # --------------- Mode: "test-x" ----------------
        elif mode == "test-x":
            qkv1 = self.qkv1(x).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qx, kx, vx, qtx = qkv1[0], qkv1[1], qkv1[2], qkv1[3]

            attntx = (qtx @ kx.transpose(-2, -1)) * self.scale
            lorthx = orthogonality_loss(qx, qtx)
            attntx = attntx.softmax(dim=-1)
            attntx = self.attn_drop(attntx)

            xt = (attntx @ vx).transpose(1, 2).reshape(B, N, C)
            xt = self.proj3(xt)
            xt = self.proj_drop(xt)

            return xt, lorthx
        # --------------- Mode: "test-y" ----------------
        elif mode == "test-y":
            qkv2 = self.qkv2(y).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qy, ky, vy, qty = qkv2[0], qkv2[1], qkv2[2], qkv2[3]

            attnty = (qty @ ky.transpose(-2, -1)) * self.scale
            lorthy = orthogonality_loss(qy, qty)
            attnty = attnty.softmax(dim=-1)
            attnty = self.attn_drop(attnty)

            yt = (attnty @ vy).transpose(1, 2).reshape(B, N, C)
            yt = self.proj4(yt)
            yt = self.proj_drop(yt)

            return yt, lorthy
        # --------------- Mode: "test-xy" ----------------
        elif mode == "test-xy":
            if x is not None and y is None:
                qkv1 = self.qkv1(x).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                qx, kx, vx, qtx = qkv1[0], qkv1[1], qkv1[2], qkv1[3]   # make torchscript happy (cannot use tensor as tuple)

                attncx = (qx @ kx.transpose(-2, -1)) * self.scale
                attncx = attncx.softmax(dim=-1)
                attncx = self.attn_drop(attncx)

                xc = (attncx @ vx).transpose(1, 2).reshape(B, N, C)
                xc = self.proj1(xc)
                xc = self.proj_drop(xc)
                return xc
            elif x is None and y is not None:
                qkv2 = self.qkv2(y).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                qy, ky, vy, qty = qkv2[0], qkv2[1], qkv2[2], qkv2[3]

                attncy = (qy @ ky.transpose(-2, -1)) * self.scale
                attncy = attncy.softmax(dim=-1)
                attncy = self.attn_drop(attncy)

                yc = (attncy @ vy).transpose(1, 2).reshape(B, N, C)
                yc = self.proj2(yc)
                yc = self.proj_drop(yc)
                return yc

class Block_self(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.attn_x = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpx = Conv_Attention(dim=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x=None):
            x = self.attn_x(self.norm1_1(x))
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlpx(self.norm2_1(x)))
            return x


class Block_Trans(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode = 'train'):
        super().__init__()
        self.mode = mode  # "train", "test-x", "test-y", "test-xy"
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        self.norm3_1 = norm_layer(dim)
        self.norm3_2 = norm_layer(dim)       

        self.cross_attn = Cross_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpx = nn.ModuleList([Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)for i in range(4)])
        self.mlpy = nn.ModuleList([Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)for i in range(4)])

    def forward(self, x=None, y=None, mode='train'):###############分类！！！
        if mode == 'train':
            xc, yc, xt, yt, lorthx, lorthy, lc  = self.cross_attn(self.norm1_1(x), self.norm1_2(y))
            xc = xc + self.drop_path(xc)
            yc = yc + self.drop_path(yc)
            xc = xc + self.drop_path(self.mlpx[0](self.norm2_1(xc)))
            yc = yc + self.drop_path(self.mlpy[1](self.norm2_2(yc)))

            xt = xt + self.drop_path(xt)
            yt = yt + self.drop_path(yt)
            xt = xt + self.drop_path(self.mlpx[2](self.norm3_1(xt)))
            yt = yt + self.drop_path(self.mlpy[3](self.norm3_2(yt)))
            return xc, yc, xt, yt, lorthx, lorthy, lc
        # --------------- Mode: "test-x" ----------------
        elif mode == 'test-x':
            xt, lorthx = self.cross_attn(self.norm1_1(x), None, mode)
            xt = xt + self.drop_path(xt)
            xt = xt + self.drop_path(self.mlpx[2](self.norm3_1(xt)))
            return xt, lorthx
        elif mode == 'test-y':
            yt, lorthy = self.cross_attn(None, self.norm1_2(y), mode)
            yt = yt + self.drop_path(yt)
            yt = yt + self.drop_path(self.mlpx[3](self.norm3_2(yt))) 
            return yt, lorthy  
        elif mode == 'test-xy':
            if x is not None and y is None:
                xc  = self.cross_attn(self.norm1_1(x), None, mode)
                xc = xc + self.drop_path(xc)
                xc = xc + self.drop_path(self.mlpx[0](self.norm2_1(xc)))  
                return xc
            elif x is None and y is not None:
                yc = self.cross_attn(None, self.norm1_2(y), mode)
                yc = yc + self.drop_path(yc)
                yc = yc + self.drop_path(self.mlpy[1](self.norm2_2(yc)))         
                return yc
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=256, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim,
                          kernel_size=patch_size,
                          stride=stride_size,
                          bias=False),
                nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1, bias=False),
                nn.ReLU(),
            )
        self.b_norm=nn.BatchNorm2d(embed_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #print("img_size", self.img_size)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.b_norm(self.conv_layers(x))
        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, sie_xishu =1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed1 = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed2 = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed1 = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)
            self.patch_embed2 = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed1.num_patches

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.cam_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #self.view_num = view
        #self.sie_xishu = sie_xishu
        #if view > 1:
            #self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            #trunc_normal_(self.sie_embed, std=.02)
            #print('viewpoint number is : {}'.format(view))
            #print('using SIE_Lambda is : {}'.format(sie_xishu))
        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks_self_x = nn.ModuleList([
            Block_self(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(int(depth/2))])
        self.blocks_self_y = nn.ModuleList([
            Block_self(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(int(depth/2))])
        self.blocks_trans = nn.ModuleList([
            Block_Trans(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+(depth-int(depth/3*2))], norm_layer=norm_layer)
            for i in range(int(depth/3*2))])
        
        self.attn = nn.ModuleList([
            Block_self(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=attn_drop_rate, norm_layer=norm_layer)
            for i in range(4)])

        self.norm0_1 = norm_layer(embed_dim)
        self.norm0_2 = norm_layer(embed_dim)
        self.norm1_1 = norm_layer(embed_dim)
        self.norm1_2 = norm_layer(embed_dim)
        self.norm1_3 = norm_layer(embed_dim)
        self.norm1_4 = norm_layer(embed_dim)
        self.norm1_5 = norm_layer(embed_dim)
        self.norm1_6 = norm_layer(embed_dim)
        self.norm2_1 = norm_layer(embed_dim)
        self.norm2_2 = norm_layer(embed_dim)
        self.norm2_3 = norm_layer(embed_dim)
        self.norm3_1 = norm_layer(embed_dim)
        self.norm3_2 = norm_layer(embed_dim)
        self.norm3_3 = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        #trunc_normal_(self.cam_token, std=.02)
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        self.depth = depth
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, camids, x=None, y=None, mode='train'):
        #都是0可以通用
        
        if mode == 'train':
            #1 1 dim -> B 1 dim
            B = x.shape[0]
            #cls_tokens1 = self.cls_token1.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = self.patch_embed1(x)#BNC
            #x = torch.cat((cls_tokens1, x), dim=1)
            x = x + self.pos_embed1
            x = self.pos_drop(x)

            #B = y.shape[0]
            #cls_tokens2 = self.cls_token2.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            y = self.patch_embed2(y)
            #y = torch.cat((cls_tokens2, y), dim=1)
            y = y + self.pos_embed2
            y = self.pos_drop(y)

            lorthx_list = []
            lorthy_list = []
            lc_list = []
            xt_list = []
            yt_list = []
            weights = torch.linspace(0.5, 1.0, steps=int(self.depth/2))  # 从0.5递增到1.0
            for i, blk in enumerate(self.blocks_self_x):
                x = blk(self.norm0_1(x)) + x
            for i, blk in enumerate(self.blocks_self_y):
                y = blk(self.norm0_2(y)) + y
            for i, blk in enumerate(self.blocks_trans):
                if i == 0:
                    xc, yc, xt, yt, lorthx, lorthy, lc = blk(self.norm1_1(x), self.norm1_2(y), mode)
                else:
                    rxc = xc
                    rxt = xt
                    ryc = yc
                    ryt = yt
                    xc, yc, xt, yt, lorthx, lorthy, lc = blk(self.norm1_3(xc), self.norm1_4(yc), mode)
                    xc = xc + rxc
                    xt = xt + rxt
                    yc = yc + ryc
                    yt = yt + ryt
                
                lorthx_list.append(lorthx)
                lorthy_list.append(lorthy)
                lc_list.append(lc)

                # 记录每次的 xt, yt
                xt_list.append(xt)
                yt_list.append(yt)

            lorthx_sum = sum(w * l for w, l in zip(weights, lorthx_list))
            lorthy_sum = sum(w * l for w, l in zip(weights, lorthy_list))
            lc_sum = sum(w * l for w, l in zip(weights, lc_list))
            # 按原逻辑组合 xt 和 yt


            xt = xt_list[0] + xt_list[1]
            xt = xt + self.attn[0](self.norm2_1(xt))   # 残差
            xt = xt + xt_list[2]
            xt = xt + self.attn[1](self.norm2_2(xt))           # 残差
            xt = self.norm2_3(xt)

            yt = yt_list[0] + yt_list[1]
            yt = yt + self.attn[2](self.norm3_1(yt))  # 残差
            yt = yt + yt_list[2]
            yt = yt + self.attn[3](self.norm3_2(yt))           # 残差
            yt = self.norm3_3(yt)

            xc = self.norm1_5(xc)
            yc = self.norm1_6(yc)
            xyc = (xc + yc)/2

            #return xyc[:, 0], xt[:, 0], yt[:, 0], lorthx_sum, lorthy_sum, lc_sum
            return xyc, xt, yt, lorthx_sum, lorthy_sum, lc_sum
        elif mode == 'test-x':
                B = x.shape[0]
                #cls_tokens1 = self.cls_token1.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = self.patch_embed1(x)
                #x = torch.cat((cls_tokens1, x), dim=1)
                x = x + self.pos_embed1
                x = self.pos_drop(x)
                for i, blk in enumerate(self.blocks_self_x):
                    x = blk(self.norm0_1(x)) + x
                xt, lorthx = self.blocks_trans[0](x=x, y=None, mode=mode)
                xt = xt + self.attn[0](self.norm2_1(xt))
                xt = xt + self.attn[1](self.norm2_2(xt))
                xt = self.norm2_3(xt)
                #return xt[:, 0], lorthx
                return xt, lorthx
        elif mode == 'test-y':
                B = y.shape[0]
                #cls_tokens2 = self.cls_token2.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                y = self.patch_embed2(y)
                #y = torch.cat((cls_tokens2, y), dim=1)
                y = y + self.pos_embed2
                y = self.pos_drop(y)
                for i, blk in enumerate(self.blocks_self_y):
                    y = blk(self.norm0_2(y)) + y
                yt, lorthy = self.blocks_trans[0](x=None, y=y, mode=mode)
                yt = yt + self.attn[2](self.norm3_1(yt))
                yt = yt + self.attn[3](self.norm3_2(yt))
                # 归一化
                yt = self.norm3_3(yt)
                #return yt[:, 0], lorthy
                return yt, lorthy
        elif mode == 'test-xy':
            B = x.shape[0]
            outputs = [None]*B
            unique_camids = camids.unique()
            for cam in unique_camids:
                idxs = (camids == cam).nonzero(as_tuple=True)[0]  # 得到属于该cam的索引
                img = x[idxs]                            # 取出该cam下的图像子集
                if cam == 0:
                    B0 = img.shape[0]
                    #cls_tokens1 = self.cls_token1.expand(B0, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    img = self.patch_embed1(img)
                    #img = torch.cat((cls_tokens1, img), dim=1)
                    img = img + self.pos_embed1
                    img = self.pos_drop(img)

                    for i, blk in enumerate(self.blocks_self_x):
                        img = blk(self.norm0_1(img)) + img

                    for i, blk in enumerate(self.blocks_trans):
                        if i == 0:
                            img = blk(self.norm1_1(img), None, mode)
                        else:
                            img = img + blk(self.norm1_3(img), None, mode)
                    img = self.norm1_5(img)
                    # 放回输出中相应位置
                    for i, idx in enumerate(idxs):
                        outputs[idx] = img[i]
                    
                elif cam == 1:
                    #B0 = img.shape[0]
                    #cls_tokens2 = self.cls_token2.expand(B0, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    img = self.patch_embed2(img)
                    #img = torch.cat((cls_tokens2, img), dim=1)
                    img = img + self.pos_embed2
                    img = self.pos_drop(img)

                    for i, blk in enumerate(self.blocks_self_y):
                        img = blk(self.norm0_2(img)) + img

                    for i, blk in enumerate(self.blocks_trans):
                        if i == 0:
                            img = blk(None, self.norm1_2(img), mode)
                        else:
                            img = img + blk(None, self.norm1_4(img), mode)

                    img = self.norm1_6(img)
                    # 放回输出中相应位置
                    for i, idx in enumerate(idxs):
                        outputs[idx] = img[i]                

            # 堆叠成一个 [B, ...] 的 tensor
            return torch.stack(outputs, dim=0)

    def forward(self, camids, x, y, mode='train'):
        if mode == 'train':
            xyc, xt, yt, lorthx_sum, lorthy_sum, lc_sum = self.forward_features(camids=camids, x=x, y=y, mode= mode)
            return xyc, xt, yt, lorthx_sum, lorthy_sum, lc_sum
        elif mode == 'test-x':
            xt, _ = self.forward_features(camids=camids, x=x, y=None, mode=mode)
            return xt
        elif mode == 'test-y':
            yt, _ = self.forward_features(camids=camids, x=None, y=y, mode=mode)
            return yt
        elif mode == 'test-xy':
            xyc= self.forward_features(camids=camids, x=x, y=None, mode=mode)#都混合在x里面了
            return xyc

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb
        
def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.1, attn_drop_rate=0.1,drop_path_rate=0.1, camera=0, view=0, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
