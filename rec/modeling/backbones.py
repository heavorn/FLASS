from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
import pdb

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def build_backbone(config, model_type):
        
    if model_type == "rec" or model_type == "cls":
        support_dict = [
            'SVTR_G_Net'
        ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class LePEAttention(nn.Layer):
    def __init__(self, dim, resolution, idx, split_size=None, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size[0]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, weight_attr=ParamAttr(initializer=KaimingNormal()))

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose((0, 2, 1)).reshape((B, C, H, W))
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape((-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose((0, 2, 1)).reshape((B, C, H, W))

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.reshape((B, C, H // H_sp, H_sp, W // W_sp, W_sp))
        x = x.transpose((0, 2, 4, 1, 3, 5)).reshape((-1, C, H_sp, W_sp)) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape((-1, self.num_heads, C // self.num_heads, H_sp * W_sp)).transpose((0, 1, 3, 2))

        x = x.reshape((-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp)).transpose((0, 1, 3, 2))
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        # attn = paddle.matmul(q, k.transpose((0, 2, 1)))
        attn = paddle.matmul(q, k, transpose_y=True)  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, axis=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v) + lepe
        x = x.transpose((0, 2, 1, 3)).reshape((-1, self.H_sp* self.W_sp, C)) # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).reshape((B, -1, C))  # B H' W' C

        return x


class CSWinBlock(nn.Layer):

    def __init__(self, dim, reso, num_heads,
                 split_size=None, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim, epsilon=epsilon)
            self.norm2 = norm_layer(dim, epsilon=epsilon)
            
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.LayerList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.LayerList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        # self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape((B, -1, 3, C)).transpose((2, 0, 1, 3))
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            attened_x = paddle.concat([x1,x2], axis=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """

    B, C, H, W = img.shape
    img_reshape = img.reshape((B, C, H // H_sp, H_sp, W // W_sp, W_sp))
    img_perm = img_reshape.transpose((0, 2, 4, 3, 5, 1)).reshape((-1, H_sp* W_sp, C))
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.reshape((B, H // H_sp, W // W_sp, H_sp, W_sp, -1))
    img = img.transpose((0, 1, 3, 2, 4, 5)).reshape((B, H, W, -1))
    return img



class OSRA_Attention(nn.Layer):  ### OSRA
    """
    OSRA Attention Layer.

    Args:
        dim (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention layer. Defaults to 0.
        sr_ratio (int, optional): The ratio of height and width of the feature map to the input resolution. Defaults to 1.
        HW (tuple[int, int] | None, optional): Spatial resolution of the input feature. Defaults to None.

    Inputs:
        x (Tensor): Input feature with shape of (B, L, C)

    Outputs:
        Tensor: Output feature with shape of (B, L, C)

    """

    def __init__(
        self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1, HW=None
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.size = HW
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2D(dim, dim, kernel_size=1, weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.kv = nn.Conv2D(dim, dim * 2, kernel_size=1, weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvBNLayer(
                    dim,
                    dim,
                    kernel_size=sr_ratio + 3,
                    stride=sr_ratio,
                    padding=(sr_ratio + 3) // 2,
                    groups=dim,
                    bias_attr=False,
                    act=nn.GELU,
                ),
                ConvBNLayer(
                    dim, dim, kernel_size=1, groups=dim, bias_attr=False, act=None
                ),
            )
        else:
            self.sr = Identity()
        self.local_conv = nn.Conv2D(dim, dim, kernel_size=3, padding=1, groups=dim, weight_attr=ParamAttr(initializer=KaimingNormal()))

    def forward(self, x, relative_pos_enc=None):
        # B, C, H, W = x.shape
        B, N, C = x.shape
        H, W = self.size
        
        x = x.transpose((0, 2, 1)).reshape((B, -1, H, W))
        q = (
            self.q(x)
            .reshape((B, self.num_heads, C // self.num_heads, -1))
            .transpose((0, 1, 3, 2))
        )
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = paddle.chunk(self.kv(kv), chunks=2, axis=1)
        k = k.reshape((B, self.num_heads, C // self.num_heads, -1))
        v = v.reshape((B, self.num_heads, C // self.num_heads, -1)).transpose((0, 1, 3, 2))
        attn = paddle.matmul(q, k) * self.scale
        # attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = nn.functional.interpolate(
                    relative_pos_enc,
                    size=attn.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
            attn = attn + relative_pos_enc
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v).transpose((0, 1, 3, 2)).reshape((B, C, -1))
        # x = (attn @ v).transpose((0, 1, 3, 2)).reshape((B, C, -1))

        x = x.transpose((0, 2, 1))
        return x  # .reshape(B, C, H, W)


class OSRA_Block(nn.Layer):
    def __init__(
        self,
        dim=64,
        sr_ratio=1,
        num_heads=1,
        mlp_ratio=4,
        norm_cfg=nn.LayerNorm,  # dict(type='GN', num_groups=1),
        act_cfg=nn.GELU,  # dict(type='GELU'),
        drop=0,
        drop_path=0,
        layer_scale_init_value=1e-5,
        HW=None,
        epsilon=1e-6,
    ):
        super().__init__()

        if isinstance(norm_cfg, str):
            self.norm1 = eval(norm_cfg)(dim, epsilon=epsilon)
            self.norm2 = eval(norm_cfg)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_cfg(dim, epsilon=epsilon)
            self.norm2 = norm_cfg(dim, epsilon=epsilon)

        # self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.pos_embed = nn.Conv2D(dim, dim, kernel_size=3, padding=1, groups=dim, weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.token_mixer = OSRA_Attention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio, HW=HW
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_cfg,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        # x = x + self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x), relative_pos_enc))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, relative_pos_enc=None):
        # if self.grad_checkpoint and x.requires_grad:
        #     x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        # else:
        x = self._forward_impl(x, relative_pos_enc)
        return x


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act() if act is not None else Identity()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class ConvMixer(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2D(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()))

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype='float32')
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = paddle.full([H * W, H * W], '-inf', dtype='float32')
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x):
        qkv = self.qkv(x).reshape(
            (0, -1, 3, self.num_heads, self.head_dim)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
        elif mode == 'linear':
            self.proj = nn.Conv2D(
                1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class SubSample(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=[2, 1],
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose((0, 2, 1)))
        else:
            x = self.conv(x)
            out = x.flatten(2).transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTR_G_Net(nn.Layer):
    def __init__(
            self,
            img_size=[32, 100],
            in_channels=3,
            embed_dim=[64, 128, 128, 256],
            depth=[3, 3, 3, 3],
            num_heads=[2, 4, 4, 8],
            mixer=['Local'] * 6 + ['Global'] *6,  # Local atten, Global atten, Conv
            local_mixer=[[7, 11], [7, 11], [7, 11], [7, 11]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=192,
            out_char_num=25,
            block_unit='Block',
            act='nn.GELU',
            last_stage=True,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.sr_ratios=[8, 4, 2, 1]
        self.split_sizes=[[1, 1], [1, 5], [1, 5], [1, 5]]
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        
        self.blocks1 = nn.LayerList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.LayerList([
            CSWinBlock(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                reso=HW,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=self.split_sizes[1],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i]
                ) for i in range(depth[1])
            ])
        self.blocks3 = nn.LayerList([
            OSRA_Block(
                    dim=embed_dim[2],
                    sr_ratio=self.sr_ratios[2],
                    num_heads=num_heads[2] // 2,
                    mlp_ratio=mlp_ratio
                    , drop=drop_rate,
                    drop_path=dpr[depth[0] + depth[1]: depth[0] + depth[1] + depth[2]][i],
                    epsilon=epsilon,
                    HW=HW)
            for i in range(depth[2])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[2],
                embed_dim[3],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks4 = nn.LayerList([
            Block_unit(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                mixer=mixer[depth[0] + depth[1] + depth[2]:][i],
                HW=HW,
                local_mixer=local_mixer[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1] + depth[2]:][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[3])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
            self.last_conv = nn.Conv2D(
                in_channels=embed_dim[3],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[3], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(
                p=last_drop, mode="downscale_in_infer")

        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        for blk in self.blocks3:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks4:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[3], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x