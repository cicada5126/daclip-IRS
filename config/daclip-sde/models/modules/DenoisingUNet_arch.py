import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)

from .attention import SpatialTransformer


class ConditionalUNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4],
                 context_dim=512, use_degra_context=True, use_image_context=False, upscale=1):
        # 调用父类构造函数
        super().__init__()
        # 记录网络深度
        self.depth = len(ch_mult)
        # 上采样因子
        self.upscale = upscale
        # 设置上下文维度，如果未提供则设为-1
        self.context_dim = -1 if context_dim is None else context_dim
        # 是否使用图像上下文
        self.use_image_context = use_image_context
        # 是否使用degra上下文
        self.use_degra_context = use_degra_context

        # 设置每个头部的通道数
        num_head_channels = 32
        # 头部维度
        dim_head = num_head_channels

        # 创建ResBlock的快捷方式，使用默认的卷积和非线性激活函数
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        # 初始化卷积层
        self.init_conv = default_conv(in_nc * 2, nf, 7)

        # 时间嵌入维度
        time_dim = nf * 4

        # 是否使用随机或学习正弦条件
        self.random_or_learned_sinusoidal_cond = False

        # 如果使用随机或学习正弦条件
        if self.random_or_learned_sinusoidal_cond:
            # 学习正弦条件的维度
            learned_sinusoidal_dim = 16
            # 创建正弦位置嵌入
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            # 傅里叶维度
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # 创建正弦位置嵌入
            sinu_pos_emb = SinusoidalPosEmb(nf)
            # 傅里叶维度
            fourier_dim = nf

        # 定义一个名为 time_mlp 的属性，它是一个由多个层组成的顺序模型
        self.time_mlp = nn.Sequential(
            # 第一层是一个位置编码层，可能使用正弦和余弦函数来嵌入位置信息
            sinu_pos_emb,

            # 第二层是一个全连接层，将输入特征从 fourier_dim 维度映射到 time_dim 维度
            nn.Linear(fourier_dim, time_dim),

            # 第三层是 GELU 激活函数，它将非线性引入到模型中，有助于学习复杂的模式
            nn.GELU(),

            # 第四层是另一个全连接层，它将特征再次映射回 time_dim 维度
            # 这通常用于进一步提取特征和增强模型的非线性能力
            nn.Linear(time_dim, time_dim)
        )

        if self.context_dim > 0 and use_degra_context:
            # self.prompt 被定义为一个可学习的参数，使用 nn.Parameter 来创建
            # nn.Parameter 是 PyTorch 中的一个类，它允许将任何对象转换为一个参数，这样它就可以通过神经网络的训练过程进行更新
            self.prompt = nn.Parameter(
                # 调用 torch.rand 来生成一个随机初始化的张量
                # torch.rand(1, time_dim) 生成一个形状为 (1, time_dim) 的张量，其中 time_dim 是之前定义的维度
                # 张量中的每个元素都是从 [0, 1) 区间的均匀分布中随机抽取的
                torch.rand(1, time_dim)
            )

            # 定义一个名为 text_mlp 的属性，它是一个由多个层组成的顺序模型，用于处理文本数据
            self.text_mlp = nn.Sequential(
                # 第一层是一个全连接层，将输入特征从 context_dim 维度映射到 time_dim 维度
                nn.Linear(context_dim, time_dim),

                # 第二层是一个非线性激活函数，这里用 NonLinearity() 表示，它可能是 nn.ReLU、nn.GELU 或其他激活函数
                # 这个非线性激活函数有助于模型捕捉和学习数据中的复杂关系
                NonLinearity(),  # 这里 NonLinearity() 应替换为具体的激活函数，例如 nn.ReLU 或 nn.GELU

                # 第三层是另一个全连接层，它将特征再次映射回 time_dim 维度
                # 这通常用于进一步提取特征和增强模型的非线性能力
                nn.Linear(time_dim, time_dim)
            )

            # 定义一个名为 prompt_mlp 的属性，它是一个单层的顺序模型，可能用于处理提示（prompt）数据
            self.prompt_mlp = nn.Linear(time_dim, time_dim)
            # 这个 MLP 只有一个全连接层，它将时间维度的特征映射回相同的时间维度
            # 这可能用于进一步处理或规范化提示数据的特征表示

        # layers
            # 构建网络层
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            # 通道倍增序列，添加初始值1
            ch_mult = [1] + ch_mult

            # 遍历网络深度
            for i in range(self.depth):
                # 输入和输出维度
                dim_in = nf * ch_mult[i]
                dim_out = nf * ch_mult[i + 1]

                # 输入和输出头部数量
                num_heads_in = dim_in // num_head_channels
                num_heads_out = dim_out // num_head_channels
                # 每个头部的输入维度
                dim_head_in = dim_in // num_heads_in

                # 如果使用图像上下文且上下文维度大于0
                if use_image_context and context_dim > 0:
                    # 这得看 i是否小于3才使用
                    att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head,
                                                                                        depth=1,
                                                                                        context_dim=context_dim)
                    att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head,
                                                                                       depth=1, context_dim=context_dim)
                else:
                    # 使用线性注意力机制
                    att_down = LinearAttention(dim_in)  # if i < 2 else Attention(dim_in)
                    att_up = LinearAttention(dim_out)  # if i < 2 else Attention(dim_out)

                # 下采样模块列表
                # self.downs 是一个用于存储处理块序列的列表，在类的初始化方法中定义
                self.downs.append(
                    nn.ModuleList([
                        # 创建第一个模块，一个自定义的块类（block_class），其输入和输出维度都是 dim_in，时间嵌入维度是 time_dim
                        block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),

                        # 创建第二个模块，与第一个模块相同
                        block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),

                        # 创建第三个模块，一个残差连接（Residual），它包含一个预归一化层（PreNorm）和注意力层（att_down）
                        Residual(PreNorm(dim_in, att_down)),

                        # 根据当前的索引 i 是否等于 (self.depth - 1)，创建不同的模块
                        # 如果 i 不等于 (self.depth - 1)，则创建一个 Downsample 模块，用于在 dim_in 和 dim_out 之间进行下采样
                        # 如果 i 等于 (self.depth - 1)，则使用 default_conv 创建一个默认的二维卷积层，将 dim_in 维度的特征映射到 dim_out 维度
                        Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
                    ])
                )

                # 上采样模块列表
                self.ups.insert(0, nn.ModuleList([
                    block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                    block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, att_up)),
                    Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
                ]))

            # 中间维度
            mid_dim = nf * ch_mult[-1]
            # 中间头部数量
            num_heads_mid = mid_dim // num_head_channels
            # 中间块1
            self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
            # 如果使用图像上下文且上下文维度大于0
            if use_image_context and context_dim > 0:
                # 使用空间变换器
                self.mid_attn = Residual(PreNorm(mid_dim, SpatialTransformer(mid_dim, num_heads_mid, dim_head, depth=1,
                                                                             context_dim=context_dim)))
            else:
                # 使用线性注意力机制
                self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            # 中间块2
            self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

            # 最终残差块
            self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
            # 最终卷积层
            self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time, text_context=None, image_context=None):
        # 检查输入的时间参数是否为整数或浮点数，如果是，则将其转换为一个单元素张量，并移动到xt所在的设备
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        # X=noisy_tensor-LQ_tensor就是文章第一步添加的随机噪声，与LQ_tensor拼接，增加通道维度
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        # 获取输入张量的空间维度H和W
        H, W = x.shape[2:]
        # 检查并调整输入张量x的空间尺寸以匹配原始图像的尺寸
        x = self.check_image_size(x, H, W)

        # 应用初始卷积层
        x = self.init_conv(x)
        # 克隆x，用于后续操作
        x_ = x.clone()

        # 通过时间MLP处理时间参数
        t = self.time_mlp(time)
        # 如果上下文维度大于0，并且使用degra上下文，且文本上下文不为空
        if self.context_dim > 0:
            if self.use_degra_context and text_context is not None:
                # 计算文本上下文的嵌入，将其与提示向量结合，并进行处理
                prompt_embedding = torch.softmax(self.text_mlp(text_context), dim=1) * self.prompt
                prompt_embedding = self.prompt_mlp(prompt_embedding)
                # 将处理后的文本上下文嵌入加到时间参数t上
                t = t + prompt_embedding

            # 如果使用图像上下文，且图像上下文不为空
            if self.use_image_context and image_context is not None:
                # 为图像上下文增加一个通道维度
                image_context = image_context.unsqueeze(1)

        # 存储下采样过程中的特征图
        h = []
        # 遍历下采样模块列表
        for b1, b2, attn, downsample in self.downs:
            # 应用第一个残差块和时间参数t
            x = b1(x, t)
            # 存储特征图
            h.append(x)

            # 应用第二个残差块和时间参数t
            x = b2(x, t)
            # 应用注意力机制，如果提供了图像上下文，则使用它
            x = attn(x, context=image_context)
            # 存储特征图
            h.append(x)

            # 应用下采样操作
            x = downsample(x)

        # 应用中间块1和时间参数t
        x = self.mid_block1(x, t)
        # 如果使用图像上下文，则应用注意力机制
        x = self.mid_attn(x, context=image_context) if self.use_image_context else x
        # 应用中间块2和时间参数t
        x = self.mid_block2(x, t)

        # 遍历上采样模块列表
        for b1, b2, attn, upsample in self.ups:
            # 从历史特征图中弹出并拼接特征，与当前特征图拼接
            x = torch.cat([x, h.pop()], dim=1)
            # 应用第一个残差块和时间参数t
            x = b1(x, t)

            # 再次从历史特征图中弹出并拼接特征，与当前特征图拼接
            x = torch.cat([x, h.pop()], dim=1)
            # 应用第二个残差块和时间参数t
            x = b2(x, t)

            # 应用注意力机制，如果提供了图像上下文，则使用它
            x = attn(x, context=image_context)
            # 应用上采样操作
            x = upsample(x)

        # 将原始输入xt与当前特征图x拼接，增加通道维度
        x = torch.cat([x, x_], dim=1)

        # 应用最终的残差块和时间参数t
        x = self.final_res_block(x, t)
        # 应用最终的卷积层
        x = self.final_conv(x)

        # 裁剪输出张量x，使其空间尺寸与原始输入图像的尺寸相匹配
        x = x[..., :H, :W].contiguous()

        # 返回处理后的输出张量x
        return x


