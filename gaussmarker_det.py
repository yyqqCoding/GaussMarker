"""
GaussMarker: 双域水印检测系统 - 水印检测与验证模块

本模块实现了GaussMarker双域水印系统的核心检测功能，能够从可能被攻击或操作的图像中
检测和验证水印的存在。该系统结合了空间域(高斯阴影)和频域(Tree-Ring)两种水印技术，
并集成了高斯噪声恢复器(GNR)来增强检测的鲁棒性。

检测系统架构:
1. **图像预处理**: 加载待检测图像并应用可选的图像变换
2. **潜在空间反演**: 通过DDIM反演将图像转换回潜在噪声
3. **双域水印检测**:
   - 空间域: 高斯阴影水印检测 (pred_gs)
   - 频域: Tree-Ring模式检测 (pred_tr)
   - GNR增强: 噪声恢复后的检测 (pred_restore)
4. **集成评估**: 结合多种检测结果的最终判决

技术特点:
- **多重检测**: 三种独立的检测方法提供冗余验证
- **鲁棒性增强**: GNR显著提升对攻击的抵抗能力
- **统计评估**: 基于ROC曲线和AUC的性能评估
- **实时检测**: 支持批量图像的高效检测

检测流程:
输入图像 → 图像变换 → 潜在空间反演 → 双域检测 → GNR增强 → 集成判决 → 检测结果

作者: GaussMarker研究团队
论文: "GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models"
"""

import os
import argparse
from tqdm import tqdm  # 批处理进度条

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image  # 图像处理
from sklearn import metrics  # 性能评估指标
import numpy as np
import logging
from scipy.special import betainc  # Beta不完全函数，用于阈值计算
import joblib  # 机器学习模型加载

# CLIP模型相关 (可选)
# from transformers import CLIPModel, CLIPTokenizer
import open_clip  # OpenCLIP实现

# GaussMarker核心组件
from inverse_stable_diffusion import InversableStableDiffusionPipeline  # 可逆扩散管道
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler  # 扩散调度器
from watermark import Gaussian_Shading_chacha, Gaussian_Shading  # 空间域水印
from utils import Affine, measure_similarity, image_distortion, transform_img, set_random_seed  # 工具函数

# Tree-Ring频域水印和GNR网络
from tr_utils import get_watermarking_mask, eval_watermark  # 频域水印工具
from unet.unet_model import UNet  # GNR网络架构

def set_logger(workdir, args):
    """
    配置水印检测过程的日志记录系统。

    为检测过程设置统一的日志格式，记录检测结果、性能指标和错误信息，
    便于后续分析和调试检测性能。

    Args:
        workdir (str): 检测结果输出目录
        args (argparse.Namespace): 检测配置参数

    Returns:
        None

    Side Effects:
        - 创建输出目录
        - 配置全局日志系统
        - 记录检测配置参数
    """
    # 创建检测结果输出目录
    os.makedirs(workdir, exist_ok=True)

    # 配置文件日志记录
    gfile_stream = open(os.path.join(workdir, 'log.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # 记录检测配置参数
    logging.info(f"水印检测配置: {args}")



class Evaluator(object):
    """
    GaussMarker水印检测性能评估器。

    该类实现了水印检测系统的统计评估功能，包括检测阈值计算、
    性能指标评估和ROC分析。基于Beta分布的统计理论，为不同的
    假阳性率要求计算相应的检测阈值。

    评估指标:
    - AUC (Area Under Curve): ROC曲线下面积
    - TPR@FPR: 指定假阳性率下的真阳性率
    - 检测准确率: 正确分类的比例
    - 集成评估: 多种检测方法的融合结果

    统计原理:
    基于二项分布的假设，使用Beta不完全函数计算检测阈值，
    确保在指定的假阳性率约束下获得最优的检测性能。
    """

    def __init__(self, ch_factor, w_factor, h_factor, fpr, user_number):
        """
        初始化评估器并计算检测阈值。

        Args:
            ch_factor (int): 通道复制因子
            w_factor (int): 宽度复制因子
            h_factor (int): 高度复制因子
            fpr (float): 期望的假阳性率
            user_number (int): 潜在用户数量
        """
        # 保存水印配置参数
        self.ch = ch_factor
        self.w = w_factor
        self.h = h_factor

        # 计算潜在空间和水印的长度
        self.latentlength = 4 * 64 * 64  # 潜在空间总长度 (4通道 × 64×64)
        self.marklength = self.latentlength // (self.ch * self.w * self.h)  # 有效水印长度

        # 初始化检测阈值
        self.tau_onebit = None   # 单比特检测阈值
        self.tau_bits = None     # 多比特检测阈值
        self.fpr = fpr          # 目标假阳性率

        # ==================== 基于Beta分布计算检测阈值 ====================
        # 使用Beta不完全函数计算在指定假阳性率下的最优阈值
        # 理论基础: 在零假设下，匹配比特数服从二项分布B(n, 0.5)
        print(f"计算检测阈值 (水印长度: {self.marklength}, 目标FPR: {fpr})")

        for i in range(self.marklength):
            # 计算单比特检测的假阳性率
            # betainc(a, b, x) 计算正则化不完全Beta函数 I_x(a,b)
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)

            # 计算多比特检测的假阳性率 (考虑用户数量)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number

            # 找到满足假阳性率要求的最小阈值
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
                print(f"  单比特检测阈值: {self.tau_onebit:.4f} (匹配{i}/{self.marklength}比特)")

            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
                print(f"  多比特检测阈值: {self.tau_bits:.4f} (匹配{i}/{self.marklength}比特)")

        # 确保阈值计算成功
        if self.tau_onebit is None:
            self.tau_onebit = 0.9  # 默认高阈值
            print(f"  警告: 无法满足单比特FPR要求，使用默认阈值: {self.tau_onebit}")
        if self.tau_bits is None:
            self.tau_bits = 0.9    # 默认高阈值
            print(f"  警告: 无法满足多比特FPR要求，使用默认阈值: {self.tau_bits}")
    
    def eval(self, ws, preds):
        """
        基础水印检测评估方法。

        计算预测结果与真实水印的匹配度，并根据预设阈值统计
        检测成功的样本数量。这是最基本的检测评估方法。

        Args:
            ws (list): 真实水印列表，每个元素为二进制水印
            preds (list): 预测水印列表，每个元素为检测到的水印

        Returns:
            tuple: (准确率列表, 单比特检测成功数, 多比特检测成功数)
                - accs: 每个样本的比特匹配准确率
                - tp_onebit_count: 超过单比特阈值的样本数
                - tp_bits_count: 超过多比特阈值的样本数
        """
        tp_onebit_count = 0   # 单比特检测真阳性计数
        tp_bits_count = 0     # 多比特检测真阳性计数
        accs = []             # 比特准确率列表

        # 逐样本评估检测性能
        for i in range(len(ws)):
            w = ws[i]           # 真实水印
            pred = preds[i]     # 预测水印

            # 计算比特级匹配准确率
            correct = (pred == w).float().mean().item()

            # 根据阈值判断检测是否成功
            if correct >= self.tau_onebit:
                tp_onebit_count += 1
            if correct >= self.tau_bits:
                tp_bits_count += 1

            accs.append(correct)

        return accs, tp_onebit_count, tp_bits_count

    def eval2(self, ws, w_preds, no_ws, no_w_preds):
        """
        二分类ROC评估方法。

        将水印检测建模为二分类问题，计算ROC曲线和相关性能指标。
        该方法同时考虑带水印样本和无水印样本，提供更全面的性能评估。

        Args:
            ws (list): 带水印样本的真实水印
            w_preds (list): 带水印样本的预测结果
            no_ws (list): 无水印样本的参考水印
            no_w_preds (list): 无水印样本的预测结果

        Returns:
            tuple: (AUC, 准确率, TPR@FPR, 阈值, 比特准确率)
                - auc: ROC曲线下面积
                - acc: 最大分类准确率
                - low: 指定FPR下的TPR
                - thre: 对应的检测阈值
                - bit_acc: 带水印样本的平均比特准确率
        """
        t_labels = []  # 真实标签 (1: 带水印, 0: 无水印)
        preds = []     # 预测分数 (比特匹配率)

        # ==================== 处理带水印样本 (正样本) ====================
        for i in range(len(ws)):
            w = ws[i]           # 真实水印
            pred = w_preds[i]   # 预测水印

            # 计算比特匹配率作为检测分数
            correct = (pred == w).float().mean().item()
            t_labels.append(1)  # 正样本标签
            preds.append(correct)

        # 计算带水印样本的平均比特准确率
        bit_acc = np.mean(preds)

        # ==================== 处理无水印样本 (负样本) ====================
        for i in range(len(no_ws)):
            w = no_ws[i]           # 参考水印 (随机或固定)
            pred = no_w_preds[i]   # 从无水印图像中的"检测"结果

            # 计算"匹配率" (应该很低，因为没有真实水印)
            correct = (pred == w).float().mean().item()
            t_labels.append(0)  # 负样本标签
            preds.append(correct)

        # 输出检测分数分布，用于调试
        print(f"带水印样本检测分数: {preds[:len(ws)]}")
        print(f"无水印样本检测分数: {preds[-len(no_ws):]}")

        # ==================== ROC分析和性能指标计算 ====================
        # 计算ROC曲线
        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)

        # 计算AUC (Area Under Curve)
        auc = metrics.auc(fpr, tpr)

        # 计算最大分类准确率 (最小化分类错误率)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # 计算指定假阳性率下的真阳性率
        valid_fpr_indices = np.where(fpr < self.fpr)[0]
        if len(valid_fpr_indices) > 0:
            low = tpr[valid_fpr_indices[-1]]
            thre = thresholds[valid_fpr_indices[-1]]
        else:
            # 如果无法满足FPR要求，使用最严格的阈值
            low = tpr[-1]
            thre = thresholds[-1]

        return auc, acc, low, thre, bit_acc

    def eval_ring(self, no_w_metrics_affine, w_metrics_affine):
        """
        Tree-Ring频域水印的专用评估方法。

        专门评估频域Tree-Ring水印的检测性能。Tree-Ring检测基于
        FFT域中的圆环模式匹配，返回相似度度量值。

        Args:
            no_w_metrics_affine (list): 无水印样本的Tree-Ring度量值
            w_metrics_affine (list): 带水印样本的Tree-Ring度量值

        Returns:
            tuple: (AUC, 准确率, TPR@FPR, 阈值)
                - auc: ROC曲线下面积
                - acc: 最大分类准确率
                - low: 指定FPR下的TPR
                - thre: 对应的检测阈值

        Note:
            Tree-Ring度量值越高表示越可能包含水印。
            该方法将度量值作为检测分数进行二分类评估。
        """
        # 合并所有样本的Tree-Ring度量值
        preds_affine = no_w_metrics_affine + w_metrics_affine

        # 创建对应的标签 (0: 无水印, 1: 带水印)
        t_labels_affine = [0] * len(no_w_metrics_affine) + [1] * len(w_metrics_affine)

        # 计算ROC曲线和性能指标
        fpr, tpr, thresholds = metrics.roc_curve(t_labels_affine, preds_affine, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # 计算指定FPR下的TPR和阈值
        valid_indices = np.where(fpr < self.fpr)[0]
        if len(valid_indices) > 0:
            low = tpr[valid_indices[-1]]
            thre = thresholds[valid_indices[-1]]
        else:
            low = tpr[-1]
            thre = thresholds[-1]

        return auc, acc, low, thre
    
    def eval_ensemble(self, ws, w_preds, no_ws, no_w_preds, no_w_metrics_affine, w_metrics_affine, clf=None):
        """
        集成检测评估方法 - GaussMarker的核心评估函数。

        该方法结合空间域(高斯阴影)和频域(Tree-Ring)两种检测结果，
        使用预训练的机器学习分类器进行最终的水印判决。这是GaussMarker
        双域水印系统的核心评估方法。

        特征组合:
        1. 空间域特征: 高斯阴影水印的比特匹配率
        2. 频域特征: Tree-Ring模式的相似度度量

        Args:
            ws (list): 带水印样本的真实水印
            w_preds (list): 带水印样本的空间域检测结果
            no_ws (list): 无水印样本的参考水印
            no_w_preds (list): 无水印样本的空间域检测结果
            no_w_metrics_affine (list): 无水印样本的频域度量
            w_metrics_affine (list): 带水印样本的频域度量
            clf: 预训练的集成分类器 (如随机森林、SVM等)

        Returns:
            tuple: (AUC, 准确率, TPR@FPR, 阈值, 分类器)
                - auc: 集成检测的ROC曲线下面积
                - acc: 最大分类准确率
                - low: 指定FPR下的TPR
                - thre: 对应的检测阈值
                - clf: 使用的分类器对象
        """
        x = []  # 特征矩阵 [空间域特征, 频域特征]
        y = []  # 标签向量 (1: 带水印, 0: 无水印)

        # ==================== 构建带水印样本的特征 ====================
        for i in range(len(ws)):
            w = ws[i]           # 真实水印
            pred = w_preds[i]   # 空间域检测结果

            # 计算空间域特征: 比特匹配率
            correct = (pred == w).float().mean().item()

            # 组合双域特征: [空间域匹配率, 频域度量值]
            x.append([correct, w_metrics_affine[i]])
            y.append(1)  # 正样本标签

        # ==================== 构建无水印样本的特征 ====================
        for i in range(len(no_ws)):
            w = no_ws[i]           # 参考水印
            pred = no_w_preds[i]   # 空间域"检测"结果

            # 计算空间域特征 (应该很低)
            correct = (pred == w).float().mean().item()

            # 组合双域特征
            x.append([correct, no_w_metrics_affine[i]])
            y.append(0)  # 负样本标签

        # 转换为numpy数组
        x = np.array(x)
        y = np.array(y)

        # ==================== 集成分类器预测 ====================
        # 使用预训练的分类器进行最终判决
        # predict_proba返回 [P(class=0), P(class=1)]，取P(class=1)作为检测分数
        preds = clf.predict_proba(x)[:, 1].tolist()

        # ==================== 性能评估 ====================
        # 基于集成分类器的输出计算ROC曲线
        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # 计算指定FPR下的性能
        valid_indices = np.where(fpr < self.fpr)[0]
        if len(valid_indices) > 0:
            low = tpr[valid_indices[-1]]
            thre = thresholds[valid_indices[-1]]
        else:
            low = tpr[-1]
            thre = thresholds[-1]

        return auc, acc, low, thre, clf


def main(args):
    """
    GaussMarker双域水印检测的主函数。

    该函数实现了完整的水印检测流程，包括图像加载、潜在空间反演、
    双域水印检测、GNR增强和集成评估。支持多种攻击场景下的鲁棒性测试。

    检测流程:
    1. 环境初始化: 加载扩散模型和相关组件
    2. 水印配置: 加载与生成阶段一致的水印参数
    3. 模型加载: 加载GNR模型和集成分类器
    4. 批量检测: 对每张图像执行完整的检测流程
    5. 性能评估: 计算各种检测指标和统计结果

    Args:
        args: 检测配置参数，包含模型路径、检测参数等

    Returns:
        None (结果保存到文件和日志)
    """
    # ==================== 计算设备和扩散模型初始化 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用计算设备: {device}")

    # 初始化扩散模型调度器
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')

    # 加载可逆扩散管道 - 支持DDIM反演的关键组件
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,  # 使用半精度节省内存
            revision='fp16',
    )
    pipe.set_progress_bar_config(disable=True)  # 禁用内置进度条
    pipe.safety_checker = None                  # 禁用安全检查
    pipe = pipe.to(device)

    print(f"扩散模型加载完成: {args.model_path}")

    # ==================== CLIP模型初始化 (可选) ====================
    # 用于计算图像-文本相似度，评估图像质量
    if args.reference_model is not None:
        print(f"加载CLIP参考模型: {args.reference_model}")
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
    else:
        print("未使用CLIP参考模型")

    # ==================== 输出目录准备 ====================
    os.makedirs(args.output_path, exist_ok=True)
    print(f"检测结果将保存到: {args.output_path}")

    # ==================== 水印配置加载 ====================
    # 加载与生成阶段完全一致的水印配置，确保检测的准确性
    w1_path = args.w1_path  # 空间域水印配置文件
    w2_path = args.w2_path  # 频域水印配置文件

    print(f"加载水印配置:")
    print(f"  空间域水印: {w1_path}")
    print(f"  频域水印: {w2_path}")

    if args.chacha:
        # 使用ChaCha20加密的高斯阴影水印
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading_chacha(
                args.channel_copy, args.w_copy, args.h_copy,
                args.fpr, args.user_number,
                watermark=w_info["w"],    # 预生成的水印模式
                m=w_info["m"],            # ChaCha20加密的消息
                key=w_info["key"],        # 加密密钥
                nonce=w_info["nonce"]     # 加密随机数
            )
            print(f"  加密方式: ChaCha20")
        else:
            raise FileNotFoundError(f"水印配置文件不存在: {w1_path}")

        # 重塑加密消息为潜在空间格式
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)

    else:
        # 使用简单XOR加密的高斯阴影水印
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading(
                args.channel_copy, args.w_copy, args.h_copy,
                args.fpr, args.user_number,
                watermark=w_info["w"],    # 预生成的水印模式
                m=w_info["m"],            # XOR加密的消息
                key=w_info["key"]         # XOR密钥
            )
            print(f"  加密方式: 简单XOR")
        else:
            raise FileNotFoundError(f"水印配置文件不存在: {w1_path}")

        # 重塑消息为潜在空间格式
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)

    print(f"  水印消息形状: {m.shape}")
    print(f"  水印长度: {watermark.marklength} 比特")

    # ==================== 性能评估器初始化 ====================
    # 创建评估器，计算检测阈值和性能指标
    evaluator = Evaluator(
        args.channel_copy, args.w_copy, args.h_copy,
        args.fpr, args.user_number
    )
    print(f"评估器初始化完成，检测阈值已计算")

    # ==================== 检测环境准备 ====================
    # 在检测时假设原始提示词未知，使用空提示词进行DDIM反演
    # 这是实际应用场景的真实假设
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    print(f"使用空提示词进行检测 (模拟真实场景)")

    # ==================== 检测结果存储初始化 ====================
    # 准备存储各种检测方法的结果
    acc = []           # 准确率列表
    clip_scores = []   # CLIP相似度分数

    # 三种检测方法的结果存储
    pred_gs = []       # 高斯阴影检测结果 (带水印样本)
    pred_tr = []       # Tree-Ring检测结果 (带水印样本)
    pred_restore = []  # GNR增强检测结果 (带水印样本)

    pred_gs_neg = []       # 高斯阴影检测结果 (无水印样本)
    pred_tr_neg = []       # Tree-Ring检测结果 (无水印样本)
    pred_restore_neg = []  # GNR增强检测结果 (无水印样本)

    w_list = []        # 真实水印列表

    # ==================== 频域水印配置加载 ====================
    # 加载Tree-Ring频域水印模式和掩码
    if os.path.exists(w2_path):
        gt_patch = torch.load(w2_path).to(device)
        print(f"频域水印模式加载成功: {gt_patch.shape}")
    else:
        raise FileNotFoundError(f"频域水印文件不存在: {w2_path}")

    # 生成频域水印检测掩码
    watermarking_mask = get_watermarking_mask(gt_patch.real, args, device)
    print(f"频域检测掩码生成完成: {watermarking_mask.shape}")

    # ==================== GNR模型加载 ====================
    # 加载训练好的高斯噪声恢复器，用于增强检测鲁棒性
    if args.GNR_path is not None:
        print(f"加载GNR模型: {args.GNR_path}")

        # 根据分类器类型确定输入通道数
        # classifier_type=1: 8通道输入 (原始消息+检测消息)
        # classifier_type=0: 4通道输入 (仅检测消息)
        input_channels = 8 if args.classifier_type == 1 else 4

        model = UNet(input_channels, 4, nf=args.model_nf).cuda()
        model.eval()  # 设置为评估模式
        model.load_state_dict(torch.load(args.GNR_path))

        print(f"  GNR输入通道: {input_channels}")
        print(f"  GNR特征数: {args.model_nf}")
        print(f"  分类器类型: {args.classifier_type}")
    else:
        model = None
        print("未使用GNR模型")

    # ==================== 集成分类器加载 ====================
    # 加载预训练的集成分类器，用于最终的水印判决
    try:
        clf_2 = joblib.load('sd21_cls2.pkl')
        print("集成分类器加载成功: sd21_cls2.pkl")
    except FileNotFoundError:
        print("警告: 集成分类器文件不存在，将使用默认评估方法")
        clf_2 = None

    # ==================== 批量水印检测循环 ====================
    print(f"\n开始批量检测，共 {args.num} 张图像")
    print("=" * 60)

    for i in tqdm(range(args.num), desc="检测水印"):
        # 检查是否已有检测结果，避免重复计算
        if os.path.exists(os.path.join(args.output_path, 'pred_res.pth')):
            print("发现已有检测结果，跳过重复计算")
            break

        # 设置当前样本的随机种子
        seed = i + args.gen_seed

        # ==================== 图像加载阶段 ====================
        # 生成参考的无水印潜在变量 (用于对比)
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()

        # 加载待检测的带水印图像
        if args.advanced_attack is None:
            # 标准检测：使用原始生成的带水印图像
            image_w_path = os.path.join(args.input_path, "w_img", f"{i}.png")
            if os.path.exists(image_w_path):
                image_w = Image.open(image_w_path)
            else:
                print(f"警告: 带水印图像不存在: {image_w_path}")
                break
        else:
            # 高级攻击检测：使用经过特定攻击的图像
            attack_image_path = os.path.join(args.input_path, args.advanced_attack, f"{i}.png")
            if os.path.exists(attack_image_path):
                image_w = Image.open(attack_image_path)
                print(f"加载攻击图像: {args.advanced_attack}/{i}.png")
            else:
                print(f"攻击图像不存在: {attack_image_path}")
                break

        # 加载对应的无水印参考图像
        if args.no_w_path is None:
            # 使用标准的无水印图像路径
            image_no_w_path = os.path.join(args.input_path, "no_w_img", f"{i}.png")
        else:
            # 使用自定义的无水印图像路径
            image_no_w_path = os.path.join(args.no_w_path, f"{i}.png")

        if os.path.exists(image_no_w_path):
            image_no_w = Image.open(image_no_w_path)
        else:
            print(f"警告: 无水印图像不存在: {image_no_w_path}")
            break

        # 加载对应的文本提示词
        prompt_path = os.path.join(args.input_path, "prompt", f"{i}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                current_prompt = f.read().strip()
        else:
            current_prompt = ""
            print(f"警告: 提示词文件不存在: {prompt_path}")

        # 保存真实水印用于后续评估
        w_list.append(watermark.watermark)

        if i % 100 == 0:
            print(f"\n处理第 {i+1}/{args.num} 张图像")
            print(f"  提示词: {current_prompt[:50]}..." if len(current_prompt) > 50 else f"  提示词: {current_prompt}")
            print(f"  种子: {seed}")

        # ==================== 图像预处理和变换 ====================
        # 应用各种图像变换来模拟真实世界的攻击场景
        while True:
            try:
                # 对带水印图像应用变换 (JPEG压缩、噪声、模糊等)
                image_w_distortion = image_distortion(image_w, seed, args)
                image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)

                # 对无水印图像应用相同的变换，确保公平对比
                image_no_w_distortion = image_distortion(image_no_w, seed, args)
                image_no_w_distortion = transform_img(image_no_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)

                break  # 变换成功，退出重试循环
            except Exception as e:
                # 如果变换失败，重试 (可能由于随机参数导致的异常)
                print(f"图像变换失败，重试中... 错误: {e}")
                pass

        # ==================== 水印检测主流程 ====================
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算

            # ==================== 几何变换处理 (可选) ====================
            if args.affine is not None:
                # 应用仿射变换攻击并获取变换后的潜在表示
                print(f"    应用仿射变换攻击: {args.affine_params}")

                # 对带水印图像应用仿射变换
                affine_image_w_distortion, affine_image_latents_w, ori_image_latents_no_w, params, trans_m = Affine(
                    image_w_distortion,      # 输入图像
                    pipe.get_image_latents,  # 潜在编码函数
                    m.float(),              # 原始水印消息 (用于变换)
                    args.affine_params      # 仿射变换参数
                )

                # 对无水印图像应用相同的仿射变换
                affine_image_no_w_distortion, affine_image_latents_no_w, ori_image_latents_no_w, _, _ = Affine(
                    image_no_w_distortion,   # 输入图像
                    pipe.get_image_latents,  # 潜在编码函数
                    None,                   # 无水印消息
                    args.affine_params      # 相同的变换参数
                )

            else:
                # 不应用仿射变换，直接获取潜在表示
                affine_image_latents_w = pipe.get_image_latents(image_w_distortion)
                affine_image_latents_no_w = pipe.get_image_latents(image_no_w_distortion)

            # ==================== DDIM反演过程 ====================
            # 通过前向扩散过程将图像反演回初始噪声，提取潜在的水印信号
            print(f"    执行DDIM反演 (步数: {args.num_inversion_steps})")

            # 对带水印图像执行DDIM反演
            reversed_latents_w_affine = pipe.forward_diffusion(
                latents=affine_image_latents_w,           # 带水印图像的潜在表示
                text_embeddings=text_embeddings,          # 文本嵌入 (空提示词)
                guidance_scale=args.guidance_scale,       # 引导强度
                num_inference_steps=args.num_inversion_steps,  # 反演步数
            )

            # 对无水印图像执行相同的DDIM反演
            reversed_latents_no_w_affine = pipe.forward_diffusion(
                latents=affine_image_latents_no_w,        # 无水印图像的潜在表示
                text_embeddings=text_embeddings,          # 相同的文本嵌入
                guidance_scale=args.guidance_scale,       # 相同的引导强度
                num_inference_steps=args.num_inversion_steps,  # 相同的反演步数
            )

            # ==================== 二值化处理 ====================
            # 将连续的潜在变量转换为二进制形式，便于水印检测
            # 使用0作为阈值：正值→1，负值→0
            reversed_m_affine = (reversed_latents_w_affine > 0).float().cpu()
            reversed_no_m_affine = (reversed_latents_no_w_affine > 0).float().cpu()

            # ==================== 仿射变换验证 (可选) ====================
            if args.affine is not None:
                # 验证仿射变换的准确性
                trans_m = (trans_m > 0.5).int()
                synthe_acc = (trans_m == reversed_m_affine.int()).float().mean()
                print(f"    仿射变换合成准确率: {synthe_acc:.4f}")

                if synthe_acc < 0.7:
                    print(f"    警告: 仿射变换准确率较低，可能影响检测性能")
            
            # ==================== 方法1: 高斯阴影空间域检测 ====================
            # 直接从反演的潜在变量中检测高斯阴影水印
            print(f"    执行高斯阴影检测")

            # 检测带水印样本
            w = watermark.pred_w_from_latent(reversed_latents_w_affine)
            pred_gs.append(w)

            # 检测无水印样本 (作为负样本对照)
            w = watermark.pred_w_from_latent(reversed_latents_no_w_affine)
            pred_gs_neg.append(w)

            # ==================== 方法2: GNR增强检测 ====================
            # 使用训练好的GNR模型恢复被攻击的水印信号，然后进行检测
            if model is not None:
                print(f"    执行GNR增强检测")

                # 对无水印样本进行GNR恢复和检测
                if args.classifier_type == 1:
                    # 类型1: 使用原始消息和检测消息的拼接作为输入
                    gnr_input_neg = torch.cat([m.float(), reversed_no_m_affine], dim=1).cuda()
                    restored_reversed_m = (F.sigmoid(model(gnr_input_neg)) > 0.5).int()
                else:
                    # 类型0: 仅使用检测消息作为输入
                    restored_reversed_m = (F.sigmoid(model(reversed_no_m_affine.cuda())) > 0.5).int()

                w = watermark.pred_w_from_m(restored_reversed_m)
                pred_restore_neg.append(w)

                # 对带水印样本进行GNR恢复和检测
                if args.classifier_type == 1:
                    # 类型1: 使用原始消息和检测消息的拼接
                    gnr_input = torch.cat([m.float(), reversed_m_affine], dim=1).cuda()
                    restored_reversed_m = (F.sigmoid(model(gnr_input)) > 0.5).int().cpu()
                else:
                    # 类型0: 仅使用检测消息
                    restored_reversed_m = (F.sigmoid(model(reversed_m_affine.cuda())) > 0.5).int()

                w = watermark.pred_w_from_m(restored_reversed_m)
                pred_restore.append(w)
            else:
                # 如果没有GNR模型，使用原始检测结果
                pred_restore_neg.append(pred_gs_neg[-1])
                pred_restore.append(pred_gs[-1])

            # ==================== 方法3: Tree-Ring频域检测 ====================
            # 在FFT域中检测Tree-Ring圆环模式
            print(f"    执行Tree-Ring频域检测")

            # 评估频域水印的相似度度量
            no_w_metric_affine, w_metric_affine = eval_watermark(
                reversed_latents_no_w_affine,  # 无水印样本的潜在变量
                reversed_latents_w_affine,     # 带水印样本的潜在变量
                watermarking_mask,             # 频域检测掩码
                gt_patch,                      # Tree-Ring模式
                args                           # 检测参数
            )

            # 存储Tree-Ring检测度量 (取负值，因为相似度越高表示越可能有水印)
            pred_tr_neg.append(-no_w_metric_affine)
            pred_tr.append(-w_metric_affine)

            # ==================== CLIP相似度评估 (可选) ====================
            # 计算生成图像与文本提示词的相似度，评估图像质量
            if args.reference_model is not None:
                print(f"    计算CLIP相似度")
                socre = measure_similarity(
                    [image_w],              # 待评估图像
                    current_prompt,         # 对应的文本提示词
                    ref_model,             # CLIP模型
                    ref_clip_preprocess,   # 图像预处理
                    ref_tokenizer,         # 文本分词器
                    device                 # 计算设备
                )
                clip_socre = socre[0].item()
                print(f"    CLIP相似度: {clip_socre:.4f}")
            else:
                clip_socre = 0
            clip_scores.append(clip_socre)

            # 输出当前样本的检测结果摘要
            if i % 100 == 0 or i < 10:
                print(f"  检测结果摘要:")
                print(f"    高斯阴影检测: {len(pred_gs)} 个结果")
                print(f"    GNR增强检测: {len(pred_restore)} 个结果")
                print(f"    Tree-Ring检测: {len(pred_tr)} 个结果")
                print(f"    CLIP相似度: {clip_socre:.4f}")
                print(f"  ----------------------------------------")



    # ==================== 检测结果保存和加载 ====================
    print(f"\n检测完成，开始性能评估")
    print("=" * 60)

    # 检查是否已有保存的检测结果
    pred_res_path = os.path.join(args.output_path, 'pred_res.pth')
    if os.path.exists(pred_res_path):
        # 加载已有的检测结果
        print(f"加载已保存的检测结果: {pred_res_path}")
        pred_res = torch.load(pred_res_path)
        w_list = pred_res['w_list']
        pred_gs = pred_res['pred_gs_w']
        pred_gs_neg = pred_res['pred_gs_no_w']
        pred_restore = pred_res['pred_r_w']
        pred_restore_neg = pred_res['pred_r_no_w']
        pred_tr = pred_res['pred_tr_w']
        pred_tr_neg = pred_res['pred_tr_no_w']

        print(f"检测结果统计:")
        print(f"  样本数量: {len(w_list)}")
        print(f"  高斯阴影检测: {len(pred_gs)} 带水印 + {len(pred_gs_neg)} 无水印")
        print(f"  GNR增强检测: {len(pred_restore)} 带水印 + {len(pred_restore_neg)} 无水印")
        print(f"  Tree-Ring检测: {len(pred_tr)} 带水印 + {len(pred_tr_neg)} 无水印")
    else:
        # 保存当前检测结果
        print(f"保存检测结果到: {pred_res_path}")
        pred_res = {
            "w_list": w_list,                    # 真实水印列表
            "pred_gs_w": pred_gs,               # 高斯阴影检测结果 (带水印)
            "pred_gs_no_w": pred_gs_neg,        # 高斯阴影检测结果 (无水印)
            "pred_r_w": pred_restore,           # GNR增强检测结果 (带水印)
            "pred_r_no_w": pred_restore_neg,    # GNR增强检测结果 (无水印)
            "pred_tr_w": pred_tr,               # Tree-Ring检测结果 (带水印)
            "pred_tr_no_w": pred_tr_neg,        # Tree-Ring检测结果 (无水印)
        }
        torch.save(pred_res, pred_res_path)

    # ==================== 性能评估和结果输出 ====================
    print(f"\n开始性能评估...")

    # 主要评估: GNR增强 + Tree-Ring集成检测
    if clf_2 is not None:
        auc, acc, low, thre, _ = evaluator.eval_ensemble(
            w_list,              # 真实水印 (带水印样本)
            pred_restore,        # GNR增强检测结果 (带水印样本)
            w_list,              # 参考水印 (无水印样本)
            pred_restore_neg,    # GNR增强检测结果 (无水印样本)
            pred_tr_neg,         # Tree-Ring度量 (无水印样本)
            pred_tr,             # Tree-Ring度量 (带水印样本)
            clf_2                # 集成分类器
        )

        print(f"\n=== GNR增强 + Tree-Ring集成检测结果 ===")
        print(f"AUC (曲线下面积): {auc:.4f}")
        print(f"最大准确率: {acc:.4f}")
        print(f"TPR@{args.fpr*100:.1f}%FPR: {low:.4f}")
        print(f"检测阈值: {thre:.4f}")

        # 记录到日志
        logging.info(f'GNR+TR集成检测 - AUC: {auc:.4f}, ACC: {acc:.4f}, TPR@{args.fpr*100:.1f}%FPR: {low:.4f}, Threshold: {thre:.4f}')
    else:
        print("警告: 集成分类器不可用，跳过集成评估")

    # 可选评估: 其他检测方法的性能
    # 这些评估被注释掉，可根据需要启用

    # 高斯阴影 + Tree-Ring检测 (不使用GNR)
    # auc, acc, low, thre = evaluator.eval_ensemble(w_list, pred_gs, w_list, pred_gs_neg, pred_tr_neg, pred_tr)
    # logging.info('GS+TR: '+f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}, Threshold: {thre}')

    # CLIP相似度统计
    if clip_scores and any(score > 0 for score in clip_scores):
        clip_mean = np.mean(clip_scores)
        clip_std = np.std(clip_scores)
        print(f"\nCLIP相似度统计:")
        print(f"  平均值: {clip_mean:.4f}")
        print(f"  标准差: {clip_std:.4f}")
        logging.info(f'CLIP相似度 - 平均: {clip_mean:.4f}, 标准差: {clip_std:.4f}')

    print(f"\n检测和评估完成！")
    print(f"详细结果已保存到: {args.output_path}")
    print("=" * 60)


if __name__ == '__main__':
    def parse_floats(string):
        """
        解析浮点数字符串的辅助函数。

        用于解析仿射变换参数等复杂的浮点数配置。

        Args:
            string (str): 浮点数字符串，格式如 "1.0,2.0-3.0,4.0"

        Returns:
            list: 解析后的浮点数列表

        Raises:
            argparse.ArgumentTypeError: 当字符串格式无效时
        """
        try:
            return [[float(x) for x in string_i.split(',')] for string_i in string.split('-')]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float value: {string}")

    # ==================== 命令行参数配置 ====================
    parser = argparse.ArgumentParser(
        description='GaussMarker双域水印检测系统',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== 基础检测参数 ====================
    parser.add_argument('--num', default=1000, type=int,
                       help='要检测的图像数量')
    parser.add_argument('--image_length', default=512, type=int,
                       help='图像尺寸 (正方形)')
    parser.add_argument('--guidance_scale', default=1., type=float,
                       help='DDIM反演的引导强度，通常设为1.0')
    parser.add_argument('--num_inference_steps', default=50, type=int,
                       help='扩散模型推理步数')
    parser.add_argument('--num_inversion_steps', default=None, type=int,
                       help='DDIM反演步数，默认与推理步数相同')
    parser.add_argument('--gen_seed', default=0, type=int,
                       help='随机种子起始值，确保可重现性')

    # ==================== 水印配置参数 ====================
    # 这些参数必须与生成阶段完全一致
    parser.add_argument('--channel_copy', default=1, type=int,
                       help='通道复制因子，必须与生成时一致')
    parser.add_argument('--w_copy', default=8, type=int,
                       help='宽度复制因子，必须与生成时一致')
    parser.add_argument('--h_copy', default=8, type=int,
                       help='高度复制因子，必须与生成时一致')
    parser.add_argument('--user_number', default=1000000, type=int,
                       help='潜在用户数量，用于计算检测阈值')
    parser.add_argument('--fpr', default=0.01, type=float,
                       help='期望假阳性率，控制检测严格程度')

    # ==================== 路径配置 ====================
    parser.add_argument('--input_path', default='./gr_gen_1000_sd21',
                       help='输入图像目录，包含w_img/, no_w_img/, prompt/子目录')
    parser.add_argument('--output_path', default='./output_EUattack_num10/',
                       help='检测结果输出目录')
    parser.add_argument('--chacha', action='store_true',
                       help='使用ChaCha20加密，必须与生成时一致')
    parser.add_argument('--reference_model', default=None,
                       help='CLIP参考模型名称，用于图像质量评估')
    parser.add_argument('--reference_model_pretrain', default=None,
                       help='CLIP模型预训练权重')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts',
                       help='数据集路径 (兼容参数)')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base',
                       help='Stable Diffusion模型路径')
    parser.add_argument('--no_w_path', default=None,
                       help='自定义无水印图像路径，默认使用input_path/no_w_img/')
    parser.add_argument('--GNR_path', default=None,
                       help='GNR模型路径，用于增强检测鲁棒性')
    parser.add_argument('--classifier_type', default=0, type=int,
                       help='GNR分类器类型: 0=4通道输入, 1=8通道输入')
    parser.add_argument('--w1_path', default='w1.pth',
                       help='空间域水印配置文件路径')
    parser.add_argument('--w2_path', default='w2.pth',
                       help='频域水印配置文件路径')

    # ==================== 图像变换和攻击参数 ====================
    # 用于模拟各种真实世界的图像操作和攻击
    parser.add_argument('--advanced_attack', default=None, type=str,
                       help='高级攻击类型，指定攻击图像的子目录名')
    parser.add_argument('--affine', default=None, type=float,
                       help='仿射变换强度，None表示不使用仿射变换')
    parser.add_argument('--affine_params', default=(8, 0, 1., 0), type=parse_floats,
                       help='仿射变换参数: (旋转角度, 平移, 缩放, 剪切)')
    parser.add_argument('--jpeg_ratio', default=None, type=int,
                       help='JPEG压缩质量 (1-100)，None表示不压缩')
    parser.add_argument('--random_crop_ratio', default=None, type=float,
                       help='随机裁剪比例 (0-1)，None表示不裁剪')
    parser.add_argument('--random_drop_ratio', default=None, type=float,
                       help='随机像素丢弃比例，模拟传输错误')
    parser.add_argument('--gaussian_blur_r', default=None, type=int,
                       help='高斯模糊半径，None表示不模糊')
    parser.add_argument('--median_blur_k', default=None, type=int,
                       help='中值滤波核大小，None表示不滤波')
    parser.add_argument('--resize_ratio', default=None, type=float,
                       help='缩放比例，None表示不缩放')
    parser.add_argument('--gaussian_std', default=None, type=float,
                       help='高斯噪声标准差，None表示不加噪声')
    parser.add_argument('--sp_prob', default=None, type=float,
                       help='椒盐噪声概率，None表示不加噪声')
    parser.add_argument('--brightness_factor', default=None, type=float,
                       help='亮度调节因子，None表示不调节')

    # ==================== GNR模型参数 ====================
    parser.add_argument('--model_nf', type=int, default=128,
                       help='GNR模型的基础特征通道数')

    # ==================== Tree-Ring频域水印参数 ====================
    # 这些参数必须与生成阶段完全一致
    parser.add_argument('--w_seed', default=999999, type=int,
                       help='频域水印随机种子，必须与生成时一致')
    parser.add_argument('--w_channel', default=3, type=int,
                       help='频域水印目标通道，-1表示所有通道')
    parser.add_argument('--w_pattern', default='ring',
                       help='水印模式类型: ring, seed_ring')
    parser.add_argument('--w_mask_shape', default='circle',
                       help='水印掩码形状: circle, signal_circle')
    parser.add_argument('--w_radius', default=10, type=int,
                       help='频域水印圆环半径，影响检测精度')
    parser.add_argument('--w_measurement', default='l1_complex',
                       help='频域相似度度量方法: l1_complex')
    parser.add_argument('--w_injection', default='complex',
                       help='频域注入方式: complex, seed, signal')
    parser.add_argument('--w_pattern_const', default=0, type=float,
                       help='频域水印模式常数')

    # ==================== 参数解析和程序执行 ====================
    args = parser.parse_args()

    # 设置默认的反演步数
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    # 初始化日志系统
    set_logger(args.output_path, args)

    print("=" * 80)
    print("GaussMarker 双域水印检测系统")
    print("=" * 80)
    print(f"输入路径: {args.input_path}")
    print(f"输出路径: {args.output_path}")
    print(f"检测图像数量: {args.num}")
    print(f"扩散模型: {args.model_path}")
    print(f"GNR模型: {args.GNR_path if args.GNR_path else '未使用'}")
    print(f"加密方式: {'ChaCha20' if args.chacha else '简单XOR'}")
    print(f"假阳性率: {args.fpr}")

    # 处理仿射变换参数
    all_affine_params = args.affine_params
    work_dir = args.output_path

    if args.affine is not None:
        # 对每组仿射变换参数执行检测
        print(f"\n将对 {len(all_affine_params)} 组仿射变换参数进行检测:")
        for i, affine_params in enumerate(all_affine_params):
            print(f"  参数组 {i+1}: {affine_params}")
            logging.info(f"开始仿射变换检测 - 参数: {affine_params}")

            # 设置当前的仿射变换参数
            args.affine_params = affine_params

            # 可选: 为每组参数创建独立的输出目录
            # args.output_path = os.path.join(work_dir, str(affine_params))

            # 执行检测
            main(args)

            print(f"  参数组 {i+1} 检测完成")
    else:
        # 不使用仿射变换，直接执行检测
        print(f"\n开始标准检测 (无仿射变换)")
        main(args)

    print(f"\n所有检测任务完成！")
    print("=" * 80)