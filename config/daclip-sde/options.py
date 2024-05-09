import logging
import os
import os.path as osp
import sys
import yaml

# sys优先寻找../../路径下的文件，方便import utils
sys.path.insert(0, "../../")
from utils.file_utils import OrderedYaml
Loader, Dumper = OrderedYaml()
# 使用自定义的OrderedYaml()返回的 Loader 来加载一个YAML文件。
# 由于 Loader 被设计为支持有·序字典，因此加载后的数据中的字典将保持其键的原始顺序。

def parse(opt_path, is_train=True):
    # 读取YAML文件
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES 输出可用的GPU list,默认只有gpu 0
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
   # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = is_train
    # 修改 YAML 文件中的 is_train 键的值为 is_train 参数的值。
    scale = 1
    if opt['distortion'] == 'sr':
        scale = opt['degradation']['scale']

        ##### sr network ####
        opt["network_G"]["setting"]["upscale"] = scale
        # opt["network_G"]["setting"]["in_nc"] *= scale**2
    # 根据 YAML 文件中的 distortion 和 degradation 键来设置缩放比例 scale
    # datasets
    # 遍历 YAML 文件中的 datasets 键，处理每个数据集的相关配置，如数据集的phase?、缩放比例和失真类型，并根据数据根目录判断是否使用 LMDB 格式。
    for phase, dataset in opt["datasets"].items():
        phase = phase.split("_")[0]
        # print(dataset)
        dataset["phase"] = phase
        dataset["scale"] = scale
        dataset["distortion"] = opt["distortion"]
        
        is_lmdb = False
        # lmdb是什么
        if dataset.get("dataroot_GT", None) is not None:
            dataset["dataroot_GT"] = osp.expanduser(dataset["dataroot_GT"])
            if dataset["dataroot_GT"].endswith("lmdb"):
                is_lmdb = True
        # if dataset.get('dataroot_GT_bg', None) is not None:
        #     dataset['dataroot_GT_bg'] = osp.expanduser(dataset['dataroot_GT_bg'])
        if dataset.get("dataroot_LQ", None) is not None:
            dataset["dataroot_LQ"] = osp.expanduser(dataset["dataroot_LQ"])
            if dataset["dataroot_LQ"].endswith("lmdb"):
                is_lmdb = True
        dataset["data_type"] = "lmdb" if is_lmdb else "img"
        if dataset["mode"].endswith("mc"):  # for memcached
            dataset["data_type"] = "mc"
            dataset["mode"] = dataset["mode"].replace("_mc", "")

    # path 处理，将路径字符串扩展为绝对路径，并根据当前文件的位置设置一些路径选项。
    # for key, path in opt["path"].items():
    #
    #     if path and key in opt["path"] and key != "strict_load":
    #         opt["path"][key] = osp.expanduser(path)
            # 这行代码将 path 变量的值添加到 opt 字典的 "path" 键下。
            # osp.expanduser 函数来自 pathlib 模块，它的作用是将路径中的 ~（波浪号）扩展为当前用户的主目录路径。
            # 如果 path 中包含 ~，这个函数会将其替换为实际的路径。如果 path 已经是绝对路径或不包含 ~，则不会进行替换。

    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir)
    )
    # osp 是 os.path 的常用别名，在 Python 中用于处理操作系统路径相关的操作。
    # 这行代码设置 opt 字典下的 "path" 键的 "root" 子键的值为当前文件（__file__）所在目录的上上上上级（即daclip-uir-main的）绝对路径。
    # osp.join 函数用于连接多个路径部分，osp.pardir 是 pathlib 模块中的一个常量，表示当前目录的上一级目录。
    # 连续使用四次 osp.pardir 实际上是将路径向上移动了四级，然后 osp.abspath 函数将这个相对路径转换为绝对路径。

    path = osp.abspath(__file__)
    # 这行代码获取当前文件的绝对路径并赋值给变量 path。__file__ 是一个特殊变量，它包含了当前脚本的路径。osp.abspath 函数将这个路径转换为绝对路径。

    config_dir = path.split("\\")[-2]
    # split 方法将 path 字符串按 \ 分割成一个列表，然后通过索引 [-2] 获取这个列表的倒数第二个元素，即配置目录的名称


    # 根据 is_train 参数的值，进一步设置实验和结果的根目录路径，以及调试模式下的某些选项
    if is_train:
        experiments_root = osp.join(
            opt["path"]["root"], "experiments", config_dir, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")

        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(opt["path"]["root"], "results", config_dir)
        opt["path"]["results_root"] = osp.join(results_root, opt["name"])
        opt["path"]["log"] = osp.join(results_root, opt["name"])
        # 更新了opt字典中的"path"键，为其添加了两个新的子键："results_root"和"log"。
        # "results_root"子键的值是results_root路径与一个名为opt["name"]的字符串连接后的结果。
        # "log"子键的值是results_root路径与opt["name"]连接后的结果，这通常用于存放日志文件。opt["name"]=universal-ir
    return opt

# test
# opt = parse('options/test.yml', is_train=False)
# print(opt["path"].get("results_root", None))
# print(opt["path"].get("log", None))
def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None
# NoneDict 类定义，它能够接受一个字典或关键字参数来初始化，它继承自 dict 类，
# convert to NoneDict, which return None for missing key.
# 旨在将一个普通的 Python 字典或列表转换为一个 NoneDict 对象，后者是一个特殊类型的字典，
# 它返回 None 而不是抛出 KeyError 当尝试访问不存在的键 它允许程序在遇到缺失的键时不会崩溃，而是可以安全地继续执行。
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        # 这一行检查 opt 是否是一个字典类型（dict）。如果是，它将进入 if 语句块。
        new_opt = dict()
        # 创建了一个新的空字典 new_opt，这是为了存储转换后的键值对。
        for key, sub_opt in opt.items():
            # 遍历 opt 字典中的所有键值对。key 是字典的键，sub_opt 是与 key 相关联的值。
            new_opt[key] = dict_to_nonedict(sub_opt)
            # 调用 dict_to_nonedict 函数递归地处理 sub_opt。这意味着如果 sub_opt 是一个字典或列表，
            # 它也会被转换成 NoneDict 或递归地转换成包含 NoneDict 的列表。转换后的值被存储在 new_opt 字典中，与原始的 key 关联
        return NoneDict(**new_opt)
    # 在处理完所有的键值对后，new_opt 字典被解包（** 操作符）并作为参数传递给 NoneDict 类的构造函数，创建一个新的 NoneDict 实例。
    # 这个新的 NoneDict 实例是原始字典的深度拷贝，并且所有的字典和列表都被转换成了 NoneDict 类型。最后，这个新的 NoneDict 实例被返回。
    elif isinstance(opt, list):
        # 如果 opt 是一个列表，那么进入这个 elif 语句块
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    #  这个列表推导式遍历 opt 列表中的每个元素 sub_opt，并对每个元素递归地调用 dict_to_nonedict 函数。
    #  这确保了列表中的每个元素（无论是字典还是列表）都被转换成 NoneDict 或递归地包含 NoneDict 的列表。然后返回这个新列表。
    else:
        return opt
    # return opt 返回原始的 opt 值，如果它不是字典或列表。


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths"""
    logger = logging.getLogger("base")
    if opt["path"]["resume_state"]:
        if (
            opt["path"].get("pretrain_model_G", None) is not None
            or opt["path"].get("pretrain_model_D", None) is not None
        ):
            logger.warning(
                "pretrain_model path will be ignored when resuming training."
            )

        opt["path"]["pretrain_model_G"] = osp.join(
            opt["path"]["models"], "{}_G.pth".format(resume_iter)
        )
        logger.info("Set [pretrain_model_G] to " + opt["path"]["pretrain_model_G"])
        if "gan" in opt["model"]:
            opt["path"]["pretrain_model_D"] = osp.join(
                opt["path"]["models"], "{}_D.pth".format(resume_iter)
            )
            logger.info("Set [pretrain_model_D] to " + opt["path"]["pretrain_model_D"])

