import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.model_zoo as model_zoo
import math


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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class PatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=None):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0]//patch_size[0],img_size[1]//patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0]*patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans,embed_dim,patch_size,patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self,x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1,2)

        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self,input_resolution,dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self,x):
        # x:B,H*W,C
        H,W = self.input_resolution #H,W是patch分辨率，不是像素分辨率
        B,L,C = x.shape
        assert L == H*W , "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B,H,W,C)
        x0 = x[:,0::2,0::2,:]
        x1 = x[:,0::2,1::2,:]
        x2 = x[:,1::2,0::2,:]
        x3 = x[:,1::2,1::2,:]
        x = torch.cat([x0,x1,x2,x3],-1) # B H/2 W/2 4*C
        x = x.view(B,-1,4*C)            # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x) # 本来是C 后来变为4C，通过self.reduction(x)变回2C，最终由 C-->2C, 达到分辨率减半，通道数加倍的结果

        return x

class WindowAttention(nn.Module):
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim =dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )#why?
       
        coords_h = torch.arange(self.window_size[0]) # coords_h = tensor([0,1,2,...,self.window_size[0]-1])  维度=Wh
        coords_w = torch.arange(self.window_size[1]) # coords_w = tensor([0,1,2,...,self.window_size[1]-1])  维度=Ww

        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww


        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim,3*dim,qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        '''
        使用从截断正态分布中提取的值填充输入张量
        self.relative_position_bias_table 是全0张量，通过trunc_normal_ 进行数值填充
        '''
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask=None):
        B_,N,C = x.shape
        '''
        x.shape = (num_windows*B, N, C)
        self.qkv(x).shape = (num_windows*B, N, 3C)
        self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).shape = (num_windows*B, N, 3, num_heads, C//num_heads)
        self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).shape = (3, num_windows*B, num_heads, N, C//num_heads)
        '''
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        '''
        q.shape = k.shape = v.shape = (num_windows*B, num_heads, N, C//num_heads)
        N = M2 代表patches的数量
        C//num_heads代表Q,K,V的维数
        '''
        q,k,v = qkv[0],qkv[1],qkv[2]
        q = q*self.scale
        # attn.shape = (num_windows*B, num_heads, N, N)  N = M2 代表patches的数量
        attn = q@k.transpose(-2,-1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        '''
        attn.shape = (num_windows*B, num_heads, M2, M2)  N = M2 代表patches的数量
        .unsqueeze(0)：扩张维度，在0对应的位置插入维度1
        relative_position_bias.unsqueeze(0).shape = (1, num_heads, M2, M2)
        num_windows*B 通过广播机制传播，relative_position_bias.unsqueeze(0).shape = (1, nH, M2, M2) 的维度1会broadcast到数量num_windows*B
        表示所有batch通用一个索引矩阵和相对位置矩阵
        '''
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            # attn.view(B_ // nW, nW, self.num_heads, N, N).shape = (B, num_windows, num_heads, M2, M2) 第一个M2代表有M2个token，第二个M2代表每个token要计算M2次QKT的值
            # mask.unsqueeze(1).unsqueeze(0).shape =                (1, num_windows, 1,         M2, M2) 第一个M2代表有M2个token，第二个M2代表每个token要计算M2次QKT的值
            # broadcast相加
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # attn.shape = (B, num_windows, num_heads, M2, M2)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTranformerBlock(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,window_size=7,shift_size=0,mlp_ratio=4.,
                qkv_bias=True,qk_scale=None,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,to_2tuple(self.window_size),num_heads,qkv_bias,qk_scale,attn_drop,drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, window_size*window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self,x):
        H,W = self.input_resolution
        B,L,C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B,H,W,C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth 
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTranformerBlock(dim=dim,input_resolution=input_resolution,
                                num_heads=num_heads,window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                                 for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self,x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self,cat,img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.cat = cat
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) 
        self.num_features2 = int(embed_dim * 2 ** (self.num_layers - 1)) 
        self.num_features3 = int(embed_dim * 2 ** (self.num_layers - 2)) 
        self.mlp_ratio = mlp_ratio # 4
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features) # norm_layer = nn.LayerNorm
        self.norm2 = norm_layer(self.num_features2)
        self.norm3 = norm_layer(self.num_features3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)  # 使用self.apply 初始化参数

    def _init_weights(self, m):
        # is_instance 判断对象是否为已知类型
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, cat):
        out = []
        x = self.patch_embed(x)  # x.shape = (H//4, W//4, C)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # self.pos_drop = nn.Dropout(p=drop_rate)

        for layer in self.layers:            
            x = layer(x)
            out.append(x)
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1) # B C
        if cat:
            x2 = self.norm2(out[-2])  # B L C
            x2 = self.avgpool(x2.transpose(1, 2))  # B C 1
            x2 = torch.flatten(x2, 1) # B C

            x3 = self.norm3(out[-3])  # B L C
            x3 = self.avgpool(x3.transpose(1, 2))  # B C 1
            x3 = torch.flatten(x3, 1) # B C
            return x,x2,x3
        else:
            return x

    def forward(self, x):
        if self.cat:
            x,x2,x3 = self.forward_features(x,True)  # x是论文图中Figure 3 a图中最后的输出
        #  self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            x = torch.cat([x,x2,x3],1)
        else:
            x = self.forward_features(x,False)
        x = self.head(x) # x.shape = (B, num_classes)
        return x



