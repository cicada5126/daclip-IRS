import logging

logger = logging.getLogger("base")



def create_model(opt):
    model = opt["model"]
    # YAML中默认model参数为denoising

    if model == "denoising":
        from .denoising_model import DenoisingModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format( model))
    m = M(opt)
    # 这个类提供了一个模型完整的训练和测试流程，包括数据准备、模型优化、评估、日志记录和模型保存。
    # 它使用 PyTorch 框架，并且考虑了分布式训练的情况。通过 opt 配置字典，用户可以灵活地配置模型的各种参数。
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
