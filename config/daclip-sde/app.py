import os
import gradio as gr
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import options as option
from models import create_model
import open_clip
import utils as util

parser = argparse.ArgumentParser()
# ArgumentParser对象是argparse模块的核心，它提供了一个接口来添加参数（arguments）和选项（options）到你的程序
parser.add_argument("-opt", type=str, default='options/test.yml', help="Path to options YMAL file.")
# 添加一个命令行选项,这个选项被命名为-opt，它是一个接受字符串类型的参数
opt = option.parse(parser.parse_args().opt, is_train=False)
# 调用options.py中的parse方法,接受两个参数，args.opt是通过 argparse 解析器得到的选项值，采用刚刚定义的默认YMAL地址

opt = option.dict_to_nonedict(opt)
# convert to NoneDict, which return None for missing key.
# load pretrained model by default
model = create_model(opt)
# 在models的init.py中的create_model()
device = model.device
# 根据配置决定CUDA还是CPU

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])

clip_model = clip_model.to(device)

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
# 调用sde_utils.py的IRSDE（）方法


def clip_transform(np_image, resolution=224):
    # 这一行定义了一个名为clip_transform的函数，它接受两个参数：np_image（一个NumPy数组格式的图像）和resolution（一个可选参数，默认值为224，表示图像的目标分辨率）。
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    # 这一行将NumPy数组格式的图像转换为PIL（Python Imaging Library）图像。首先，将NumPy数组中的像素值乘以255，然后转换为无符号的8位整数格式，这是因为图像的像素值通常在0到255的范围内。
    return Compose([
        # 来自torchvision.transforms
        # 这一行开始定义一个转换流程，Compose是来自albumentations库的一个函数，用于组合多个图像转换操作。
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        # 这一行使用Resize操作来调整图像大小到指定的分辨率。interpolation=InterpolationMode.BICUBIC指定了使用双三次插值方法来调整图像大小，这是一种高质量的插值算法。
        CenterCrop(resolution),
        # 这一行应用CenterCrop操作，将调整大小后的图像进行中心裁剪，以确保图像的尺寸严格等于指定的分辨率
        ToTensor(),
        # 这一行使用ToTensor操作将PIL图像转换为PyTorch张量。这是为了使图像能够被深度学习模型处理。
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)
    # 这一行应用Normalize操作，对图像的每个通道进行标准化。它使用两组参数，分别对应图像的均值和标准差。这些参数通常是根据预训练模型的要求来设置的。
    # 然后，将转换流程应用到PIL图像上，并返回处理后的张量。


def restore(image):
    # 定义了一个名为restore的函数，它接受一个参数：image（一个图像张量）。
    image = image / 255.
    # 这一行将输入的图像张量的像素值归一化到0到1的范围内。
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    # 调用刚刚定义的预处理函数clip_transform函数来处理图像，该函数进行图像预处理，将图像转为PIL后进行一系列的转换操作，包括Resize指定分辨率、CenterCrop、ToTensor、Normalize
    # 然后使用unsqueeze(0)在张量前面增加一个维度（通常是为了添加一个批次维度），最后将处理后的张量发送到指定的设备（如GPU）。
    with torch.no_grad(), torch.cuda.amp.autocast():
        # 这一行开始一个上下文管理器，用于关闭梯度计算（torch.no_grad()），这对于推理阶段是必要的，因为我们不需要计算反向传播。
        # torch.cuda.amp.autocast()用于自动将操作转换为半精度浮点数，这可以提高计算速度并减少内存使用
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        # 这一行使用CLIP模型的encode_image方法来编码处理后的图像，生成图像的上下文信息。
        # control=True可能意味着模型在编码时使用了某种控制机制。
        #
        image_context = image_context.float()
        # 这一行将图像上下文张量转换为浮点数类型。
        degra_context = degra_context.float()
        # 同上将degra_context张量转换为浮点数类型

    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # 这一行根据输入的image创建了一个低质量（LQ）图像的张量副本，并调整了维度顺序（从HWC到CHW），这是大多数深度学习模型期望的格式，并在前面增加了一个批次维度。



    noisy_tensor = sde.noise_state(LQ_tensor)
    # 这一行使用一个名为sde的noise_state方法来向 低质量图像张量添加噪声。
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    # 这一行将带有噪声的图像和原始低质量图像以及上下文信息传递给一个模型，这个模型可能是用于图像恢复的深度学习模型。
    model.test(sde)
    # 这一行调用模型的test方法，
    visuals = model.get_current_visuals(need_GT=False)
    # 这一行从模型中获取当前的可视化结果，need_GT=False可能意味着不需要真实的高质量图像作为参考
    output = util.tensor2img(visuals["Output"].squeeze())
    # 这一行将模型输出的恢复图像张量转换为PIL图像格式。visuals["Output"]获取了输出图像的张量，squeeze()方法移除了所有单维度的批次维度。
    return output[:, :, [2, 1, 0]]
    # 这一行将图像的通道顺序从RGB转换为BGR，这是大多数图像处理库和显示设备使用的格式，并返回最终的恢复图像。


examples = [os.path.join(os.path.dirname(__file__), f"images/{i}.jpg") for i in range(1, 11)]
# 这一行创建了一个包含10个示例图像路径的列表。这些图像被用于Gradio界面中的示例展示。
interface = gr.Interface(fn=restore, inputs="image", outputs="image", title="基于DA-CLIP的图像修复", examples=examples)
# 这一行使用Gradio库创建了一个界面对象。fn=restore指定了当用户上传图像时应该调用的函数，inputs="image"和outputs="image"定义了输入和输出类型，title设置了界面的标题，examples提供了界面中的示例图像。
# 启动应用
interface.launch()