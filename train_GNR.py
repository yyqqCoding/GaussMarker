"""
GaussMarker: 高斯噪声恢复器 (Gaussian Noise Restorer, GNR) 训练模块

本模块实现了GaussMarker双域水印系统中的关键组件——高斯噪声恢复器(GNR)的训练过程。
GNR是一个模型无关的可学习网络，专门用于从被攻击或操作后的图像中恢复原始的水印噪声信号，
从而显著提升水印检测的鲁棒性。

GNR在GaussMarker系统中的作用:
1. **噪声恢复**: 从被攻击的图像中恢复原始的高斯阴影水印信号
2. **鲁棒性增强**: 提升对几何变换、压缩、噪声等攻击的抵抗能力
3. **模型无关性**: 不依赖特定的扩散模型，具有良好的泛化能力
4. **检测增强**: 与原始检测方法结合，提供更可靠的水印验证

技术创新点:
- **U-Net架构**: 采用编码器-解码器结构，有效处理空间信息
- **数据增强策略**: 模拟真实攻击场景的训练数据生成
- **对抗训练**: 使用正负样本平衡训练，提升判别能力
- **几何变换模拟**: 通过仿射变换模拟各种几何攻击

作者: GaussMarker研究团队
论文: "GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models"
"""

import os
import argparse
import logging
from tqdm import tqdm  # 训练进度条
import datetime
import numpy as np  # 数值计算

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from torchvision import transforms  # 图像变换和数据增强
from torchvision.utils import save_image  # 训练过程可视化

# GaussMarker核心组件
from watermark import *  # 高斯阴影水印实现
from unet.unet_model import UNet  # U-Net网络架构

def flip_tensor(tensor, flip_prob):
    """
    随机翻转张量中的二进制值，模拟噪声攻击。

    该函数模拟图像在传输或处理过程中可能遇到的位翻转噪声攻击，
    这是训练GNR鲁棒性的重要数据增强策略。

    Args:
        tensor (torch.Tensor): 输入的二进制张量 (值为0或1)
        flip_prob (float): 每个位被翻转的概率 (0.0-1.0)

    Returns:
        torch.Tensor: 翻转后的张量，与输入形状相同

    数学原理:
        对于每个位置 (i,j)，以概率 flip_prob 执行：
        output[i,j] = 1 - input[i,j]  (0变1，1变0)
    """
    # 生成与输入张量相同形状的随机数张量
    random_tensor = torch.rand(tensor.size())

    # 克隆输入张量以避免原地修改
    flipped_tensor = tensor.clone()

    # 在随机位置执行位翻转操作 (0→1, 1→0)
    flipped_tensor[random_tensor < flip_prob] = 1 - flipped_tensor[random_tensor < flip_prob]

    return flipped_tensor


def Affine_random(latent, r, t, s_min, s_max, sh):
    """
    对潜在变量应用随机仿射变换，模拟几何攻击。

    该函数是GNR训练中的核心数据增强方法，通过模拟真实世界中可能遇到的
    各种几何变换攻击来训练网络的鲁棒性。这些变换包括旋转、平移、缩放和剪切。

    Args:
        latent (torch.Tensor): 输入潜在变量，形状为 (batch, channels, height, width)
        r (float): 旋转角度范围 (度)，实际旋转角度在 [-r, r] 之间
        t (float): 平移范围，相对于图像尺寸的比例
        s_min (float): 最小缩放因子
        s_max (float): 最大缩放因子
        sh (float): 剪切角度范围 (度)

    Returns:
        tuple: (变换后的潜在变量, 变换参数)
            - new_latent: 应用仿射变换后的张量
            - params: 变换参数元组 (旋转角度, x平移, y平移, 缩放因子)

    技术细节:
        1. 使用PyTorch的RandomAffine生成随机变换参数
        2. 对每个通道独立应用相同的几何变换
        3. 使用特殊填充值(999999)标记变换后的无效区域
        4. 用随机二进制值填充无效区域，保持数据分布
    """
    # 配置随机仿射变换的参数范围
    config = dict(
        degrees=(-r, r),                    # 旋转角度范围
        translate=(t, t),                   # 平移范围 (x, y方向)
        scale_ranges=(s_min, s_max),        # 缩放因子范围
        shears=(-sh, sh),                   # 剪切角度范围
        img_size=latent.shape[-2:]          # 图像尺寸 (height, width)
    )

    # 随机生成具体的变换参数
    r, (tx, ty), s, (shx, shy) = transforms.RandomAffine.get_params(**config)

    # 获取张量维度信息
    b, c, w, h = latent.shape

    # 将多通道张量重塑为单通道格式以应用仿射变换
    # PyTorch的仿射变换函数要求输入为 (batch*channels, 1, height, width)
    new_latent = transforms.functional.affine(
        latent.view(b*c, 1, w, h),          # 重塑为单通道格式
        angle=r,                            # 旋转角度
        translate=(tx, ty),                 # 平移量
        scale=s,                            # 缩放因子
        shear=(shx, shy),                   # 剪切参数
        fill=999999                         # 填充值，用于标记无效区域
    )

    # 恢复原始的多通道格式
    new_latent = new_latent.view(b, c, w, h)

    # 创建有效区域掩码：值小于999998的区域为有效区域
    mask = (new_latent[:, :1, ...] < 999998).float()

    # 对无效区域填充随机二进制值，保持水印数据的统计特性
    # 有效区域保持变换后的值，无效区域用随机0/1填充
    new_latent = new_latent * mask + torch.randint_like(new_latent, low=0, high=2) * (1-mask)

    return new_latent, (r, tx, ty, s)

class LatentDataset_m(IterableDataset):
    """
    GNR训练专用的潜在空间数据集类。

    该数据集类是GNR训练的核心组件，负责生成用于训练高斯噪声恢复器的样本对。
    它通过模拟各种攻击场景来生成训练数据，使GNR能够学会从被攻击的水印信号中
    恢复原始的水印信息。

    数据集设计原理:
    1. **正样本生成**: 对真实水印信号应用攻击，训练网络恢复原始信号
    2. **负样本生成**: 对随机噪声应用攻击，训练网络识别非水印信号
    3. **平衡采样**: 通过neg_p参数控制正负样本比例
    4. **多水印支持**: 支持多个不同的水印模式以提升泛化能力

    训练策略:
    - 输入: 被攻击后的水印信号 (几何变换 + 位翻转噪声)
    - 目标: 原始的干净水印信号
    - 损失: BCE损失，训练网络输出0-1之间的概率值
    """

    def __init__(self, watermark, args):
        """
        初始化潜在空间数据集。

        Args:
            watermark: 高斯阴影水印对象，包含加密的水印信息
            args: 训练配置参数，包含数据增强和采样策略设置
        """
        super(LatentDataset_m, self).__init__()
        self.watermark = watermark
        self.args = args

        # ==================== 多水印模式配置 ====================
        # 支持多个水印模式训练，提升GNR的泛化能力
        if self.args.num_watermarks > 1:
            # 真实水印信号：从ChaCha20加密的消息中获取
            t_m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)

            # 生成额外的随机水印模式作为数据增强
            # 这些随机模式帮助网络学习更通用的噪声恢复能力
            o_m = torch.randint(low=0, high=2, size=(self.args.num_watermarks-1, 4, 64, 64))

            # 合并真实水印和随机水印
            self.m = torch.cat([t_m, o_m])
        else:
            # 单一水印模式：仅使用真实的加密水印信号
            self.m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)

        # ==================== 负样本概率配置 ====================
        # 计算负样本采样概率，确保正负样本的平衡
        # neg_p = 1/(1+N)，其中N是水印数量
        # 这确保了正样本和负样本的期望比例为 N:1
        self.args.neg_p = 1 / (1 + self.args.num_watermarks)

        print(f"数据集初始化完成:")
        print(f"  水印数量: {self.args.num_watermarks}")
        print(f"  负样本概率: {self.args.neg_p:.3f}")
        print(f"  水印信号形状: {self.m.shape}")

    def __iter__(self):
        """
        无限迭代器，持续生成训练样本对。

        每次迭代生成一个 (输入, 目标) 样本对：
        - 正样本: (攻击后的水印, 原始水印) - 训练恢复能力
        - 负样本: (攻击后的随机噪声, 攻击后的随机噪声) - 训练判别能力

        Yields:
            tuple: (输入张量, 目标张量)
                - 输入: 被攻击的信号 (4, 64, 64)
                - 目标: 期望恢复的信号 (4, 64, 64)
        """
        while True:
            # ==================== 随机选择水印模式 ====================
            # 从可用的水印模式中随机选择一个
            random_index = torch.randint(0, self.args.num_watermarks, (1,)).item()
            latents_m = self.m[random_index:random_index+1]  # 选中的水印信号

            # 生成随机的假水印信号作为负样本基础
            false_latents_m = torch.randint_like(latents_m, low=0, high=2)

            # ==================== 正负样本采样决策 ====================
            if np.random.rand() > self.args.neg_p:
                # ========== 正样本生成 ==========
                # 目标：训练网络从攻击后的水印中恢复原始水印

                # 对真实水印应用几何攻击
                aug_latents_m, params = Affine_random(
                    latents_m.float(),
                    self.args.r,        # 旋转范围
                    self.args.t,        # 平移范围
                    self.args.s_min,    # 最小缩放
                    self.args.s_max,    # 最大缩放
                    self.args.sh        # 剪切范围
                )

                # 应用位翻转噪声攻击
                aug_latents_m = flip_tensor(aug_latents_m, self.args.fp)

                # 返回 (攻击后的水印, 原始水印)
                yield aug_latents_m.squeeze(0).float(), latents_m.squeeze(0).float()

            else:
                # ========== 负样本生成 ==========
                # 目标：训练网络识别并保持非水印信号不变

                # 对随机噪声应用相同的攻击
                aug_false_latents_m, params = Affine_random(
                    false_latents_m.float(),
                    self.args.r, self.args.t,
                    self.args.s_min, self.args.s_max,
                    self.args.sh
                )

                # 应用位翻转噪声攻击
                aug_false_latents_m = flip_tensor(aug_false_latents_m, self.args.fp)

                # 返回 (攻击后的随机噪声, 攻击后的随机噪声)
                # 目标是网络学会对非水印信号保持不变
                yield aug_false_latents_m.squeeze(0).float(), aug_false_latents_m.squeeze(0).float()


def set_logger(gfile_stream):
    """
    配置训练过程的日志记录系统。

    设置统一的日志格式，将训练过程中的重要信息记录到文件中，
    便于后续分析和调试。

    Args:
        gfile_stream: 文件流对象，用于写入日志信息
    """
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def main(args):
    """
    GNR训练的主函数，实现完整的训练流程。

    该函数协调整个GNR训练过程，包括模型初始化、数据加载、
    训练循环、模型保存等关键步骤。GNR的训练目标是学会从
    被攻击的水印信号中恢复原始的水印信息。

    训练流程:
    1. 环境初始化：创建输出目录、配置日志系统
    2. 模型构建：初始化U-Net网络架构
    3. 数据准备：加载水印信息、创建训练数据集
    4. 训练循环：迭代优化网络参数
    5. 模型保存：定期保存检查点和最终模型

    Args:
        args: 训练配置参数，包含所有必要的超参数和路径设置
    """
    # ==================== 环境初始化 ====================
    # 创建输出目录用于保存模型、日志和可视化结果
    os.makedirs(args.output_path, exist_ok=True)

    # 配置日志系统，记录训练过程
    gfile_stream = open(os.path.join(args.output_path, 'log.txt'), 'a')
    set_logger(gfile_stream)
    logging.info(f"开始GNR训练，配置参数: {args}")

    # 提取关键训练参数
    num_steps = args.train_steps  # 总训练步数
    bs = args.batch_size         # 批次大小

    # ==================== U-Net模型初始化 ====================
    # 构建GNR的核心网络：U-Net架构
    # 输入: 4通道 (被攻击的水印信号)
    # 输出: 4通道 (恢复的水印信号)
    # nf: 网络的基础特征通道数，控制模型容量
    model = UNet(
        n_channels=4,           # 输入通道数：对应潜在空间的4个通道
        n_classes=4,            # 输出通道数：恢复的4通道水印信号
        nf=args.model_nf        # 基础特征数：控制网络容量和表达能力
    )

    # ==================== 多GPU配置 ====================
    # 检测可用的GPU数量并配置多GPU训练
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"检测到 {torch.cuda.device_count()} 个GPU，启用多GPU训练")
        model = torch.nn.DataParallel(model)
        effective_batch_size = args.batch_size * torch.cuda.device_count()
        print(f"  多GPU模式: DataParallel")
        print(f"  有效批次大小: {effective_batch_size} (单GPU: {args.batch_size})")
    else:
        print(f"使用单GPU训练")
        effective_batch_size = args.batch_size

    model = model.cuda()

    # 统计模型参数数量，评估模型复杂度
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'GNR模型参数数量: {n_params:,}')
    logging.info(f'GNR模型参数数量: {n_params:,}')

    # ==================== 损失函数和优化器配置 ====================
    # BCE损失：适用于二进制分类任务，输出0-1概率值
    # 这里将水印恢复任务建模为逐像素的二进制分类问题
    criterion = torch.nn.BCEWithLogitsLoss()

    # Adam优化器：自适应学习率，适合深度学习训练
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"模型配置完成:")
    print(f"  网络架构: U-Net")
    print(f"  输入/输出通道: 4/4")
    print(f"  基础特征数: {args.model_nf}")
    print(f"  损失函数: BCEWithLogitsLoss")
    print(f"  优化器: Adam (lr={args.lr})")

    # ==================== 水印信息加载与配置 ====================
    # GNR训练需要使用与生成阶段相同的水印配置，确保训练数据的一致性
    if os.path.exists(args.w_info_path):
        # 加载已存在的水印配置文件
        print(f"加载水印配置文件: {args.w_info_path}")
        w_info = torch.load(args.w_info_path)

        # 使用ChaCha20加密的高斯阴影水印
        # 这确保了GNR训练使用的水印信号与实际部署时完全一致
        watermark = Gaussian_Shading_chacha(
            args.channel_copy,      # 通道复制因子
            args.w_copy,           # 宽度复制因子
            args.h_copy,           # 高度复制因子
            args.fpr,              # 假阳性率
            args.user_number,      # 用户数量
            watermark=w_info["w"], # 预生成的水印模式
            m=w_info["m"],         # ChaCha20加密的消息
            key=w_info["key"],     # 加密密钥
            nonce=w_info["nonce"]  # 加密随机数
        )
        print(f"  水印类型: ChaCha20加密高斯阴影")
        print(f"  消息形状: {w_info['m'].shape}")

    else:
        # 创建新的水印配置（通常不在训练阶段执行）
        print(f"创建新的水印配置: {args.w_info_path}")
        watermark = Gaussian_Shading_chacha(
            args.channel_copy, args.w_copy, args.h_copy,
            args.fpr, args.user_number
        )

        # 生成水印和加密消息
        _ = watermark.create_watermark_and_return_w_m()

        # 保存水印配置供后续使用
        torch.save({
            "w": watermark.watermark,
            "m": watermark.m,
            "key": watermark.key,
            "nonce": watermark.nonce
        }, args.w_info_path)

    # ==================== 训练数据集创建 ====================
    # 根据采样类型创建相应的数据集
    if args.sample_type == "m":
        # 使用消息级别的数据集（标准配置）
        # 直接在潜在空间的消息表示上进行训练
        dataset = LatentDataset_m(watermark, args)
        print(f"创建消息级训练数据集")
        print(f"  数据增强配置:")
        print(f"    旋转范围: ±{args.r}°")
        print(f"    平移范围: ±{args.t}")
        print(f"    缩放范围: [{args.s_min}, {args.s_max}]")
        print(f"    剪切范围: ±{args.sh}°")
        print(f"    位翻转概率: {args.fp}")
        print(f"    负样本概率: {args.neg_p}")
    else:
        # 其他采样类型暂未实现
        raise NotImplementedError(f"采样类型 '{args.sample_type}' 暂未实现")

    # ==================== 数据加载器配置 ====================
    # 创建多进程数据加载器，提升训练效率
    data_loader = DataLoader(
        dataset,
        batch_size=bs,                    # 批次大小
        num_workers=args.num_workers      # 并行工作进程数
    )

    print(f"数据加载器配置:")
    print(f"  批次大小: {bs}")
    print(f"  工作进程数: {args.num_workers}")
    print(f"  数据集类型: 无限迭代器")

    # ==================== 主训练循环 ====================
    print(f"\n开始GNR训练，总步数: {num_steps}")
    print("=" * 60)

    for i, batch in tqdm(enumerate(data_loader), desc="训练GNR", total=num_steps):
        # ==================== 数据准备 ====================
        # 获取一个批次的训练数据
        x, y = batch  # x: 被攻击的水印信号, y: 目标恢复信号

        # 将数据移动到GPU并确保数据类型正确
        x = x.cuda()              # 输入：被攻击的水印信号
        y = y.cuda().float()      # 目标：期望恢复的信号

        # ==================== 前向传播 ====================
        # 通过U-Net网络预测恢复的水印信号
        # 网络输出logits，需要通过sigmoid转换为概率值
        pred = model(x)

        # ==================== 损失计算 ====================
        # 使用BCE损失计算预测值与目标值之间的差异
        # BCEWithLogitsLoss内部包含sigmoid激活，更数值稳定
        loss = criterion(pred, y)

        # ==================== 反向传播和参数更新 ====================
        # 清零梯度，避免梯度累积
        optimizer.zero_grad()

        # 计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # ==================== 定期可视化和保存 ====================
        if i % 2000 == 0:
            # 每2000步进行一次训练过程可视化
            # 将logits转换为概率值用于可视化
            pred_vis = F.sigmoid(pred)

            # 创建可视化图像：[输入, 预测, 目标] 的对比
            # 只可视化第一个通道，便于观察
            save_imgs = torch.cat([
                x[:, :1, ...].unsqueeze(0),        # 输入的被攻击信号
                pred_vis[:, :1, ...].unsqueeze(0), # 网络预测的恢复信号
                y[:, :1, ...].unsqueeze(0)         # 目标的原始信号
            ]).permute(1, 0, 2, 3, 4).contiguous()

            # 重塑为可保存的格式，限制显示数量为64个样本
            save_imgs = save_imgs.view(-1, save_imgs.shape[2], save_imgs.shape[3], save_imgs.shape[4])[:64]

            # 保存可视化结果，每行6个图像
            save_image(save_imgs, os.path.join(args.output_path, f"sample_{i}.png"), nrow=6)

            print(f"\n步骤 {i}: 已保存可视化结果")
            print(f"  当前损失: {loss.item():.6f}")
            print(f"  输入范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
            print(f"  预测范围: [{pred_vis.min().item():.3f}, {pred_vis.max().item():.3f}]")
            print(f"  目标范围: [{y.min().item():.3f}, {y.max().item():.3f}]")

        # ==================== 定期检查点保存 ====================
        if i % 200 == 0:
            # 每200步保存一次检查点和日志
            current_loss = loss.item()
            print(f"步骤 {i}: 损失 = {current_loss:.6f}")

            # 保存当前模型状态
            checkpoint_path = os.path.join(args.output_path, "checkpoint.pth")
            torch.save(model.state_dict(), checkpoint_path)

            # 记录训练日志
            logging.info(f"训练步骤 {i}, 损失: {current_loss:.6f}")

        # ==================== 训练完成检查 ====================
        if i > num_steps:
            print(f"\n训练完成！总步数: {i}")
            break

    # ==================== 保存最终模型 ====================
    final_model_path = os.path.join(args.output_path, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)

    print(f"\n训练完成！")
    print(f"最终模型已保存至: {final_model_path}")
    print(f"训练日志已保存至: {os.path.join(args.output_path, 'log.txt')}")
    print("=" * 60)
    

if __name__ == "__main__":
    # ==================== 命令行参数配置 ====================
    parser = argparse.ArgumentParser(
        description='GaussMarker高斯噪声恢复器(GNR)训练程序',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== 基础配置参数 ====================
    # 这些参数主要用于兼容性，在GNR训练中不直接使用
    parser.add_argument('--num', default=1000, type=int,
                       help='图像生成数量（兼容参数）')
    parser.add_argument('--image_length', default=512, type=int,
                       help='图像尺寸（兼容参数）')
    parser.add_argument('--guidance_scale', default=7.5, type=float,
                       help='引导强度（兼容参数）')
    parser.add_argument('--num_inference_steps', default=50, type=int,
                       help='推理步数（兼容参数）')
    parser.add_argument('--num_inversion_steps', default=None, type=int,
                       help='反演步数（兼容参数）')
    parser.add_argument('--gen_seed', default=0, type=int,
                       help='生成种子（兼容参数）')

    # ==================== 水印配置参数 ====================
    # 这些参数必须与水印生成阶段保持一致
    parser.add_argument('--channel_copy', default=1, type=int,
                       help='通道复制因子，控制水印在通道维度的重复')
    parser.add_argument('--w_copy', default=8, type=int,
                       help='宽度复制因子，控制水印在宽度维度的重复')
    parser.add_argument('--h_copy', default=8, type=int,
                       help='高度复制因子，控制水印在高度维度的重复')
    parser.add_argument('--user_number', default=1000000, type=int,
                       help='潜在用户数量，用于计算检测阈值')
    parser.add_argument('--fpr', default=0.000001, type=float,
                       help='期望假阳性率，控制检测严格程度')

    # ==================== 路径配置 ====================
    parser.add_argument('--output_path', default='./GNR',
                       help='GNR模型和训练结果的输出目录')
    parser.add_argument('--chacha', action='store_true',
                       help='使用ChaCha20加密（推荐启用）')
    parser.add_argument('--reference_model', default=None,
                       help='参考模型名称（兼容参数）')
    parser.add_argument('--reference_model_pretrain', default=None,
                       help='参考模型预训练权重（兼容参数）')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts',
                       help='数据集路径（兼容参数）')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base',
                       help='扩散模型路径（兼容参数）')
    parser.add_argument('--w_info_path', default='./w1.pth',
                       help='水印信息文件路径，必须与生成阶段一致')

    # ==================== 训练超参数 ====================
    parser.add_argument('--train_steps', type=int, default=10000,
                       help='总训练步数，建议50000+以获得最佳性能')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小，根据GPU内存调整')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率，Adam优化器的初始学习率')
    parser.add_argument('--sample_type', default="m",
                       help='采样类型：m=消息级采样（推荐）')

    # ==================== 数据增强参数 ====================
    # 这些参数控制训练时的攻击模拟强度，直接影响GNR的鲁棒性
    parser.add_argument('--r', type=float, default=8,
                       help='旋转攻击范围（度），±r度内随机旋转')
    parser.add_argument('--t', type=float, default=0,
                       help='平移攻击范围，相对图像尺寸的比例')
    parser.add_argument('--s_min', type=float, default=0.5,
                       help='最小缩放因子，模拟缩小攻击')
    parser.add_argument('--s_max', type=float, default=2.0,
                       help='最大缩放因子，模拟放大攻击')
    parser.add_argument('--sh', type=float, default=0,
                       help='剪切攻击范围（度），±sh度内随机剪切')
    parser.add_argument('--fp', type=float, default=0.00,
                       help='位翻转概率，模拟噪声攻击（0.0-1.0）')
    parser.add_argument('--neg_p', type=float, default=0.5,
                       help='负样本概率，控制正负样本平衡')

    # ==================== 系统配置参数 ====================
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载器的并行工作进程数')
    parser.add_argument('--num_watermarks', type=int, default=1,
                       help='训练使用的水印模式数量，增加可提升泛化能力')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='启用多GPU训练 (DataParallel)')

    # ==================== 模型架构参数 ====================
    parser.add_argument('--model_nf', type=int, default=64,
                       help='U-Net的基础特征通道数，控制模型容量')
    parser.add_argument('--exp_description', '-ed', default="",
                       help='实验描述，用于区分不同的训练配置')

    # ==================== 参数解析和路径配置 ====================
    args = parser.parse_args()

    # 生成带时间戳的输出目录（可选）
    # nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # args.output_path = args.output_path + f'_r{args.r}_t{args.t}_s_{args.s_min}_{args.s_max}_sh{args.sh}_fp{args.fp}_np{args.neg_p}_{args.exp_description}_{nowTime}'

    # 使用实验描述作为输出目录后缀
    if args.exp_description:
        args.output_path = args.output_path + '_' + args.exp_description

    print("=" * 80)
    print("GaussMarker 高斯噪声恢复器 (GNR) 训练程序")
    print("=" * 80)
    print(f"输出目录: {args.output_path}")
    print(f"水印配置: {args.w_info_path}")
    print(f"训练步数: {args.train_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"数据增强配置:")
    print(f"  旋转范围: ±{args.r}°")
    print(f"  缩放范围: [{args.s_min}, {args.s_max}]")
    print(f"  位翻转概率: {args.fp}")
    print("=" * 80)

    # 启动训练
    main(args)