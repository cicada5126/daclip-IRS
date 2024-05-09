import argparse
import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype
from .coca_model import CoCa
from .daclip_model import DaCLIP
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, DaClipLoss
from .openai import load_openai_model
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform, AugmentationCfg
from .tokenizer import HFTokenizer, tokenize


HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name):
    if model_name.startswith(HF_HUB_PREFIX):
        tokenizer = HFTokenizer(model_name[len(HF_HUB_PREFIX):])
    else:
        config = get_model_config(model_name)
        tokenizer = HFTokenizer(
            config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # if state_dict.items():
        # 'RecursiveScriptModule' object has no attribute 'items'
        # pass
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 这段代码的作用是检查一个 PyTorch 模型的状态字典（state_dict）中的键是否以 'module.' 开头，
    # 并且如果是的话，将这些键中的 'module.' 前缀去掉，从而得到一个新的状态字典。
    # 在 PyTorch 中，当你使用 nn.DataParallel 或者其他并行处理模块时，
    # 模型的状态字典中的键会被添加一个 'module.' 前缀。
    # 这是因为 nn.DataParallel 会将模型封装在一个额外的模块中，以便在多个 GPU 上并行运行。
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # 检测旧格式并使之与新格式兼容
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    #     检测旧格式:
    # 这一行检查状态字典中是否存在 'positional_embedding' 键，
    # 同时检查模型是否有一个名为 positional_embedding 的属性。
    # 如果状态字典中有这个键，但模型中没有相应的属性，这可能意味着状态字典来自一个旧版本的模型。

    resize_pos_embed(state_dict, model)

    # resize_pos_embed(state_dict, model) 这一行调用了一个名为 resize_pos_embed 的函数，它根据模型的需要调整位置嵌入（positional embedding）的大小。
    # 这通常是因为模型可能期望位置嵌入的大小与状态字典中保存的大小不同。

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    # incompatible_keys = model.load_state_dict(state_dict, strict=strict) 这一行使用模型的 load_state_dict 方法将状态字典加载到模型中。strict 参数指定是否严格要求状态字典中的所有键都与模型中的参数完全匹配。
    # 如果设置为 True，则只有完全匹配的键会被加载；如果存在不匹配的键，将会抛出一个错误。如果设置为 False，则不匹配的键会被忽略
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
):
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        pretrained_cfg = config['preprocess_cfg']
        model_cfg = config['model_cfg']
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        pretrained_cfg = {}
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)
    # 如果 pretrained 变量的值在转换为小写后是 'openai'
    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )

    else:
        model_cfg = model_cfg or get_model_config(model_name)
        # 通过model_name获取model_cfg或者上面的hf\或openai
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        if custom_text:
            if is_hf_model:
                model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
            if "coca" in model_name:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            elif "daclip" in model_name:
                clip_model = CLIP(**model_cfg, cast_dtype=cast_dtype)
                model = DaCLIP(clip_model)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from .transformer import LayerNormFp32
                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''

            if "daclip" in model_name:
                pretrained_cfg = get_pretrained_cfg(model_name[7:], pretrained)
                # pretrained_cfg为空
            else:
                pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)

            elif os.path.exists(pretrained):
                checkpoint_path = pretrained
            # print("checkpoint_path",checkpoint_path)
            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                if pretrained_cfg and "daclip" in model_name:
                    load_checkpoint(model.clip, checkpoint_path)
                    model.initial_controller()
                    model.lock_clip()
                else:
                    load_checkpoint(model, checkpoint_path)
            #         加载权重字典
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model


# opt = {'name': 'universal-ir', 'suffix': None, 'model': 'denoising', 'distortion': ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted'], 'gpu_ids': [0], 'sde': {'max_sigma': 50, 'T': 100, 'schedule': 'cosine', 'eps': 0.005}, 'degradation': {'sigma': 25, 'noise_type': 'G', 'scale': 4}, 'datasets': {'test1': {'name': 'Test', 'mode': 'LQGT', 'dataroot_GT': 'C:\\Users\\86136\\Desktop\\LQ_test\\shadow\\GT', 'dataroot_LQ': 'C:\\Users\\86136\\Desktop\\LQ_test\\shadow\\LQ', 'phase': 'test1', 'scale': 1, 'distortion': ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted'], 'data_type': 'img'}}, 'network_G': {'which_model_G': 'ConditionalUNet', 'setting': {'in_nc': 3, 'out_nc': 3, 'nf': 64, 'ch_mult': [1, 2, 4, 8], 'context_dim': 512, 'use_degra_context': True, 'use_image_context': True}}, 'path': {'pretrain_model_G': 'E:\\daclip\\pretrained\\universal-ir.pth', 'daclip': 'E:\\daclip\\pretrained\\daclip_ViT-B-32.pt', 'results_root': 'C:\\Users\\86136\\Desktop\\daclip-uir-main\\results\\daclip-sde\\universal-ir', 'log': 'C:\\Users\\86136\\Desktop\\daclip-uir-main\\results\\daclip-sde\\universal-ir', 'root': 'C:\\Users\\86136\\Desktop\\daclip-uir-main'}, 'is_train': False}
#
# model = create_model(opt)

def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "da" in args.model.lower():
        return DaClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        require_pretrained=True,
    )

    if not return_transform:
        return model

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess
