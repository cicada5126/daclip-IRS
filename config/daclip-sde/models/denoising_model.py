import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")

# 继承自BaseModel
class DenoisingModel(BaseModel):
    def __init__(self, opt):

        super(DenoisingModel, self).__init__(opt)
        # dist参数用来指示是否启用分布式训练的标志。分布式训练是一种并行化技术，
        # 它允许你将训练过程分散到多个计算设备上，从而加速训练过程，特别是在处理大型数据集和复杂模型时。
        # yml中未配置，采用nonedict从而跳过相关参数缺失
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        # 调用了一个名为 define_G 的函数
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )

        else:
            self.model = DataParallel(self.model)
            # 这行代码使用 PyTorch 的 DataParallel 模块将模型包装起来。
            # DataParallel 是一个用于在多个 GPU 上并行化模型的类。如果你有多个 GPU，使用 DataParallel 可以显著加快模型的训练速度。
            # 包装后的模型（DataParallel(self.model)）被赋值回 self.model，这样后续的调用将作用于 DataParallel 包装的模型。
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()
    # 准备模型输入数据
    def feed_data(self, state, LQ, GT=None, text_context=None, image_context=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        self.text_context = text_context
        self.image_context = image_context
    # 执行模型的参数优化。
    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze(), text_context=self.text_context, image_context=self.image_context)
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()
    # 测试模型。
    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)
        self.model.eval()
        # 将模型设置为评估模式。
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states, text_context=self.text_context, image_context=self.image_context)

        self.model.train()
    #     将模型设置为训练模式。
    # 测试模型。
    def get_current_log(self):
        return self.log_dict
    # 返回当前的日志信息。
    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict
    # 获取当前的可视化信息。
    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)
    # 打印网络结构信息。
    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        # 加载yml路径下的模型预训练权重
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])
    # 加载预训练模型。
    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
    # 保存当前模型和 EMA 模型。