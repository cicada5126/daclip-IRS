# 缺少img处理的版本
# 这份代码是在app的基础上做了clip退化类型检测功能，并去掉图像复原处理过程
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


examples = [os.path.join(os.path.dirname(__file__), f"images/{i}.jpg") for i in range(1, 11)]

degradations = [
    "运动模糊", "有雾", "JPEG压缩伪影", "低光照", "噪声", "雨滴", "多雨的", "阴影遮挡的", "多雪的", "遮挡修复"
]
text_tokens = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed',
               'snowy', 'uncompleted']
text = open_clip.tokenize(text_tokens).to(device)


def detect(image):

    image = image / 255.
    # 这一行将输入的图像张量的像素值归一化到0到1的范围内。
    img4clip = clip_transform(image).unsqueeze(0).to(device)

    if image is None:
        pass
    else:
        # 计算daclip识别结果
        with torch.no_grad(), torch.cuda.amp.autocast():
            # image_features, degra_features = clip_model.encode_image(img4clip, control=True)
            degra_features = clip_model.encode_image(img4clip, control=False)
            # , control=True启动图像控制器，不设置只有clip图像编码器
            text_features = clip_model.encode_text(text)

            # normalized features
            degra_features = degra_features / degra_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * degra_features @ text_features.t()

    # ...（省略之前的代码）

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    degradation_probabilities = {degradation: round(prob, 3) for degradation, prob in
                                 zip(degradations, probs[0].flatten())}
    print(degradation_probabilities)
    # 返回恢复后的图像和输入的退化类型。
    return degradation_probabilities


interface = gr.Interface(
    fn=detect,  # 要调用的函数
    inputs=[gr.Image(label="输入图像")]
    ,
    outputs=gr.Label(label="退化类型概率"),
    title="CLIP的图像退化识别" # 界面标题
    , examples=examples
)

interface.launch()

