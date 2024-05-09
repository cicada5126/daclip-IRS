# 在app.py基础上添加功能，功能一退化类型识别结果显示、功能二手动选择退化类型功能
# v2版本 v1版本基础上如果输入的degradation_type等于图像控制器生成的图像嵌入计算概率最大的退化类型，应该使用自动检测的图像编码进行复原以获得最佳复原效果
import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import options as option
from models import create_model
import sys

sys.path.insert(0, "../../")
# 以下文件配置在一级项目目录
import gradio as gr
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
print("load DA-CLIP")

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
print("load IR-SDE")


def find_corresponding_degradation(text_tokens, degradations, max_category):
    # 将 max_category 转换为小写，以便进行不区分大小写的比较
    max_category_lower = max_category.lower()

    # 在 text_tokens 列表中查找 max_category
    for index, token in enumerate(text_tokens):
        if token.lower() == max_category_lower:
            # 如果找到匹配的项，返回对应的退化描述
            return degradations[index]

    # 如果没有找到匹配的项，返回一个消息说明
    return f"未找到与 '{max_category}' 对应的退化类型。"


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


examples = [os.path.join(os.path.dirname(__file__), f"images/{i}.jpg") for i in range(1, 11)]
degradations = [
    "运动模糊", "有雾", "JPEG压缩伪影", "低光照", "噪声", "雨滴", "多雨的", "阴影遮挡的", "多雪的", "遮挡修复"
]
degradations3 = [
    "遮挡修复", "有雾", "多雨的", "低光照", "噪声", "雨滴", "运动模糊", "阴影遮挡的", "多雪的", "JPEG压缩伪影"
]
# degradations3主要是对应原版准备的image
# 初始化一个新的二维列表
paired_examples = []

# 遍历两个列表并将它们两两配对
for example, degradation in zip(examples, degradations3):
    # 创建一个新的子列表，包含图片地址和退化类型
    paired_example = [example, degradation]
    # 将这个子列表添加到二维列表中
    paired_examples.append(paired_example)

degradations2 = [
    "自动选择", "运动模糊", "有雾", "JPEG压缩伪影", "低光照", "噪声", "雨滴", "多雨的", "阴影遮挡的", "多雪的", "遮挡修复"
]
text_tokens = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed',
               'snowy', 'uncompleted']
text = open_clip.tokenize(text_tokens).to(device)

degradation_to_token = {degradation: token for degradation, token in zip(degradations, text_tokens)}


# 函数：根据退化类型查找对应的文本token
def find_text_token(degradation):
    return degradation_to_token.get(degradation)


def restore(image, degradation_type=None):
    # 定义了一个名为restore的函数，它接受两个参数：image（一个图像张量）和degradation_type（退化类型）。
    image = image / 255.
    # 这一行将输入的图像张量的像素值归一化到0到1的范围内。
    img4clip = clip_transform(image).unsqueeze(0).to(device)

    if degradation_type is not None and degradation_type != "自动选择":

        with torch.no_grad(), torch.cuda.amp.autocast():
            # 指定退化类型
            image_context, degra_context = clip_model.encode_image(img4clip, control=True)
            image_context = image_context.float()

            degradation = find_text_token(degradation_type)
            # 找到对应英文
            text = open_clip.tokenize(degradation).to(device)
            # tokenize
            degradation_features = clip_model.encode_text(text)
            degradation_features = degradation_features.float()

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features, degra_features = clip_model.encode_image(img4clip, control=True)
        #  control=True启动图像控制器
        image_context = image_features.float()
        degra_context = degra_features.float()

        # 计算probs
        text_tokens = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy',
                       'shadowed',
                       'snowy', 'uncompleted']
        text = open_clip.tokenize(text_tokens).to(device)
        text_features = clip_model.encode_text(text)

        # normalized features
        degra_features1 = degra_features / degra_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * degra_features1 @ text_features.t()

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)
    max_prob = probs[0].max()
    max_index = probs[0].argmax()
    max_category = text_tokens[max_index]
    # 调用函数并打印结果
    da_degradation_type = find_corresponding_degradation(text_tokens, degradations, max_category)
    print(f"Most likely degradation type: {da_degradation_type} with a probability of: {max_prob:.4f}")

    # 以上根据是否指定退化类型，生成了文本编码degradation_features和图像编码degra_context
    # 由于文本编码效果较差，如果输入的degradation_type等于图像控制器生成的图像嵌入计算概率最大的退化类型，
    # 应该使用自动检测的图像编码进行复原以获得最佳复原效果
    if degradation_type is not None and degradation_type != "自动选择":
        degra_context = degra_context if degradation_type==da_degradation_type else degradation_features
    # 复原阶段
    # 添加随机噪声
    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    #  permute改变次序
    noisy_tensor = sde.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    restore_image = util.tensor2img(visuals["Output"].squeeze())
    image = restore_image[:, :, [2, 1, 0]]
    # 返回恢复后的图像和输入的退化类型。

    return image, da_degradation_type


interface = gr.Interface(
    fn=restore,  # 要调用的函数
    inputs=[gr.Image(label="输入图像")],  # 第一个输入，图像类型
    additional_inputs=[
        # 下拉菜单输入
        gr.Radio(choices=degradations2,
                 label="对应退化类型")
    ],

    outputs=[gr.Image(label="修复后的图像"),  # 第一个输出，图像类型
             gr.Textbox(label="daclip自动检测识别结果为：")  # 第二个输出，文本类型
             ],
    title="基于DA-CLIP的图像修复",  # 界面标题
    description="上传图片后，选择某种退化类型或者自动检测退化类型。",  # 界面描述
    examples=paired_examples
)

# interface.launch(share=True)
# 启动public URL
interface.launch()
