"""
GaussMarker: 扩散模型的鲁棒双域水印 - 图像生成模块

本模块实现了GaussMarker的水印图像生成组件，这是一个针对扩散模型的最先进双域水印系统。
GaussMarker在空间域和频域中同时嵌入水印，以实现对各种图像操作和攻击的卓越鲁棒性。

主要特性:
- 双域水印: 结合空间域(高斯阴影)和频域(Tree-Ring)水印
- ChaCha20加密: 用于安全的多比特水印生成
- 兼容性: 支持Stable Diffusion模型 (v1.5, v2.1, SDXL)
- 对比生成: 同时生成带水印和无水印图像用于对比

架构概述:
1. 空间域 (W1): 使用带ChaCha20加密的高斯阴影技术生成多比特水印
2. 频域 (W2): 在FFT域中使用Tree-Ring模式生成零比特水印
3. 流水线注入: 将两种水印结合到初始潜在噪声中

作者: GaussMarker研究团队
论文: "GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models"
"""

import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 可选：使用HuggingFace镜像加速下载
import argparse
from tqdm import tqdm  # 批处理进度条
# from huggingface_hub import login  # 可选：HuggingFace身份验证

# login()  # 如果使用私有HuggingFace模型请取消注释

import torch
import logging

# 具有水印注入能力的核心扩散管道
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

# 用于可重现性和数据集处理的工具函数
from utils import set_random_seed, get_dataset

# 空间域水印生成类(高斯阴影)
from watermark import Gaussian_Shading_chacha, Gaussian_Shading

# 频域Tree-Ring水印工具
from tr_utils import get_watermarking_pattern, get_watermarking_mask, inject_watermark, eval_watermark

def set_logger(workdir, args):
    """
    初始化日志系统并创建输出目录结构。

    该函数为水印生成过程设置日志基础设施，并创建必要的目录结构来组织输出文件。

    创建的目录结构:
    - workdir/w_img/: 带水印图像
    - workdir/no_w_img/: 无水印参考图像
    - workdir/prompt/: 生成时使用的文本提示词
    - workdir/log.txt: 执行日志文件

    参数:
        workdir (str): 基础输出目录路径
        args (argparse.Namespace): 要记录的命令行参数

    返回:
        None

    副作用:
        - 创建目录结构
        - 配置全局文件日志
        - 记录提供的参数
    """
    # 创建主输出目录和子目录
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "w_img"), exist_ok=True)      # 带水印图像
    os.makedirs(os.path.join(workdir, "no_w_img"), exist_ok=True)   # 干净参考图像
    os.makedirs(os.path.join(workdir, "prompt"), exist_ok=True)     # 文本提示词

    # 设置文件日志
    gfile_stream = open(os.path.join(workdir, 'log.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # 记录配置参数
    logging.info(args)


def main(args):
    """
    使用GaussMarker双域方法生成带水印图像的主函数。

    该函数实现了完整的GaussMarker水印流水线:
    1. 初始化Stable Diffusion管道
    2. 设置双域水印(空间域 + 频域)
    3. 生成带水印和参考图像
    4. 以有组织的目录结构保存输出

    双域方法结合了:
    - 空间域: 带ChaCha20加密的高斯阴影水印
    - 频域: FFT空间中的Tree-Ring模式

    参数:
        args (argparse.Namespace): 配置参数，包括:
            - model_path: Stable Diffusion模型路径
            - output_path: 结果保存目录
            - num: 要生成的图像数量
            - chacha: 是否使用ChaCha20加密
            - w1_path, w2_path: 水印存储路径

    返回:
        None

    副作用:
        - 生成并保存带水印/参考图像
        - 创建水印文件 (w1.pth, w2.pth)
        - 记录生成进度
    """
    # 确定计算设备(优先使用CUDA以加快生成速度)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化DPM-Solver++调度器以实现高效采样
    # DPM-Solver++在保持质量的同时提供比DDIM更快的收敛速度
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')

    # 加载具有水印注入能力的修改版Stable Diffusion管道
    # InversableStableDiffusionPipeline扩展了标准管道，具有:
    # - 前向/后向扩散过程
    # - 用于水印注入的潜在空间操作
    # - 自定义噪声初始化
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,  # 使用半精度以提高内存效率
            revision='fp16',            # 如果可用则加载fp16权重
    )

    # 禁用进度条以获得更清洁的输出(我们使用tqdm代替)
    pipe.set_progress_bar_config(disable=True)

    # 禁用安全检查器以允许所有内容生成
    # 注意: 在生产环境中，考虑保持安全检查启用
    pipe.safety_checker = None

    # 将管道移动到GPU以加快推理速度
    pipe = pipe.to(device)

    # ==================== 数据集准备阶段 ====================
    # 确保输出目录存在，用于保存生成的图像和相关文件
    os.makedirs(args.output_path, exist_ok=True)

    # 加载文本提示词数据集，支持HuggingFace数据集或本地数据集
    # dataset: 包含文本提示词的数据集对象
    # prompt_key: 数据集中文本提示词字段的键名
    dataset, prompt_key = get_dataset(args)

    # ==================== 双域水印系统初始化 ====================
    # GaussMarker使用两种互补的水印技术：
    # W1 (空间域): 高斯阴影技术，嵌入多比特用户信息
    # W2 (频域): Tree-Ring技术，提供零比特鲁棒性增强

    w1_path = args.w1_path  # 空间域水印文件路径 (多比特水印)
    w2_path = args.w2_path  # 频域水印文件路径 (零比特水印)

    # ==================== 空间域水印 (W1) 配置 ====================
    # 空间域水印使用高斯阴影技术，通过截断正态分布在噪声中编码信息
    if args.chacha:
        # 使用ChaCha20流密码的安全版本
        # ChaCha20是一种现代、快速且密码学安全的流密码算法
        if os.path.exists(w1_path):
            # 加载已存在的水印配置，确保一致性
            print(f"加载现有的ChaCha20水印配置: {w1_path}")
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading_chacha(
                args.channel_copy,      # 通道复制因子：控制水印在通道维度的重复
                args.w_copy,           # 宽度复制因子：控制水印在宽度维度的重复
                args.h_copy,           # 高度复制因子：控制水印在高度维度的重复
                args.fpr,              # 假阳性率：控制检测的严格程度 (默认1e-6)
                args.user_number,      # 潜在用户数量：用于计算检测阈值
                watermark=w_info["w"], # 预生成的二进制水印模式
                m=w_info["m"],         # ChaCha20加密后的消息
                key=w_info["key"],     # ChaCha20加密密钥 (256位)
                nonce=w_info["nonce"]  # ChaCha20随机数 (96位)，确保加密安全性
            )
        else:
            # 创建新的ChaCha20水印配置
            print(f"创建新的ChaCha20水印配置: {w1_path}")
            watermark = Gaussian_Shading_chacha(
                args.channel_copy, args.w_copy, args.h_copy,
                args.fpr, args.user_number
            )
            # 生成水印和加密消息
            # create_watermark_and_return_w_m() 执行以下步骤：
            # 1. 生成随机二进制水印模式
            # 2. 使用ChaCha20加密水印信息
            # 3. 通过截断正态分布将加密消息转换为高斯噪声
            _ = watermark.create_watermark_and_return_w_m()

            # 保存水印配置以供后续使用和检测
            torch.save({
                "w": watermark.watermark,  # 原始二进制水印模式
                "m": watermark.m,          # ChaCha20加密后的消息
                "key": watermark.key,      # 加密密钥
                "nonce": watermark.nonce   # 加密随机数
            }, w1_path)

        # 将加密消息重塑为Stable Diffusion潜在空间的标准维度
        # (1, 4, 64, 64) 对应 (batch_size, channels, height, width)
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)

    else:
        # 使用简单XOR加密的基础版本 (不推荐用于生产环境)
        # 这种方法安全性较低，但计算开销更小
        if os.path.exists(w1_path):
            # 加载现有的简单水印配置
            print(f"加载现有的简单水印配置: {w1_path}")
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading(
                args.channel_copy,      # 通道复制因子
                args.hw_copy,          # 高宽复制因子 (简化版使用相同的高宽因子)
                args.fpr,              # 假阳性率
                args.user_number,      # 用户数量
                watermark=w_info["w"], # 预生成的水印
                m=w_info["m"],         # XOR加密的消息
                key=w_info["key"]      # XOR密钥
            )
        else:
            # 创建新的简单水印配置
            print(f"创建新的简单水印配置: {w1_path}")
            watermark = Gaussian_Shading(
                args.channel_copy, args.hw_copy, args.fpr, args.user_number
            )
            # 生成简单加密的水印
            _ = watermark.create_watermark_and_return_w_m()

            # 保存简单水印配置
            torch.save({
                "w": watermark.watermark,  # 二进制水印模式
                "m": watermark.m,          # XOR加密的消息
                "key": watermark.key       # XOR密钥
            }, w1_path)

        # 重塑消息维度以匹配潜在空间
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)


    # ==================== 频域水印 (W2) 配置 ====================
    # 频域水印使用Tree-Ring技术，在傅里叶变换域中嵌入圆环模式
    # 这种方法对旋转、缩放等几何变换具有天然的鲁棒性

    if os.path.exists(w2_path):
        # 加载已存在的频域水印模式
        print(f"加载现有的Tree-Ring频域水印: {w2_path}")
        gt_patch = torch.load(w2_path).to(device)
    else:
        # 生成新的Tree-Ring水印模式
        print(f"创建新的Tree-Ring频域水印: {w2_path}")
        # get_watermarking_pattern() 在FFT域中生成圆环结构：
        # 1. 对随机噪声进行2D傅里叶变换
        # 2. 在频谱中创建同心圆环模式
        # 3. 圆环半径由 args.w_radius 控制
        # 4. 模式类型由 args.w_pattern 决定 ('ring', 'seed_ring')
        gt_patch = get_watermarking_pattern(
            pipe, args, device,
            shape=(1, 4, 64, 64)  # 与潜在空间维度匹配
        )
        # 保存频域水印模式以确保一致性
        torch.save(gt_patch, w2_path)

    # 创建频域水印的注入掩码
    # get_watermarking_mask() 生成圆形掩码，定义哪些频率分量将被修改：
    # 1. 根据 args.w_mask_shape 创建掩码形状 ('circle', 'signal_circle')
    # 2. 掩码半径由 args.w_radius 控制
    # 3. 目标通道由 args.w_channel 指定 (-1表示所有通道)
    # 4. 掩码为布尔张量，True表示该位置将被水印替换
    watermarking_mask = get_watermarking_mask(gt_patch.real, args, device)

    # ==================== 批量图像生成循环 ====================
    # 对每个文本提示词生成一对图像：带水印图像和无水印参考图像
    print(f"开始生成 {args.num} 对图像 (带水印 + 无水印参考)")

    for i in tqdm(range(args.num), desc="生成双域水印图像"):
        # ==================== 种子设置和提示词准备 ====================
        # 为每张图像创建确定性种子，确保可重现的生成结果
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        print(f"处理第 {i+1}/{args.num} 张图像")
        print(f"  种子: {seed}")
        print(f"  提示词: {current_prompt[:50]}..." if len(current_prompt) > 50 else f"  提示词: {current_prompt}")

        # ==================== 双域水印注入流程 ====================
        # 设置随机种子确保噪声生成的一致性
        set_random_seed(seed)

        # 步骤1: 生成空间域水印噪声 (高斯阴影)
        # create_watermark_and_return_w_m() 返回：
        # - init_latents_w_gs: 包含空间域水印的初始潜在变量
        # - _: 加密后的消息矩阵 (此处不使用)
        print(f"    步骤1: 生成空间域水印噪声 (高斯阴影)")
        init_latents_w_gs, _ = watermark.create_watermark_and_return_w_m()

        # 生成无水印的参考噪声，用于对比实验
        # 这确保了我们可以评估水印对图像质量的影响
        print(f"    步骤2: 生成无水印参考噪声")
        init_latents_no_w = pipe.get_random_latents()

        # 步骤2: 注入频域水印 (Tree-Ring)
        # inject_watermark() 执行以下操作：
        # 1. 对空间域水印噪声进行2D FFT变换
        # 2. 在指定的频率位置 (由watermarking_mask定义) 注入Tree-Ring模式
        # 3. 执行逆FFT变换回到空间域
        # 4. 返回同时包含空间域和频域水印的最终噪声
        print(f"    步骤3: 注入频域水印 (Tree-Ring模式)")
        print(f"      - 频域掩码形状: {watermarking_mask.shape}")
        print(f"      - Tree-Ring模式形状: {gt_patch.shape}")
        print(f"      - 注入方式: {args.w_injection}")
        print(f"      - 目标通道: {args.w_channel}")
        print(f"      - 圆环半径: {args.w_radius}")

        init_latents_w = inject_watermark(
            init_latents_w_gs.float().cuda(),  # 空间域水印噪声 (转换为float32和CUDA)
            watermarking_mask,                 # 频域注入掩码 (定义修改位置)
            gt_patch,                         # Tree-Ring水印模式 (复数域)
            args                              # 注入参数配置
        ).half()  # 转换回half精度以节省显存

        print(f"    双域水印注入完成，最终噪声形状: {init_latents_w.shape}")

        # ==================== 扩散模型图像生成阶段 ====================
        print(f"    步骤4: 开始扩散模型图像生成")

        # 4.1 生成带双域水印的图像
        # 使用包含空间域和频域水印的初始噪声进行扩散生成
        print(f"      4.1 生成带水印图像...")
        print(f"        - 引导强度: {args.guidance_scale}")
        print(f"        - 推理步数: {args.num_inference_steps}")
        print(f"        - 图像尺寸: {args.image_length}x{args.image_length}")

        image_w = pipe(
            current_prompt,                    # 文本提示词：指导图像内容生成
            num_images_per_prompt=1,          # 每个提示词生成一张图像
            guidance_scale=args.guidance_scale, # 分类器自由引导强度：控制文本遵循度
            num_inference_steps=args.num_inference_steps, # 去噪步数：影响图像质量和生成时间
            height=args.image_length,         # 输出图像高度 (像素)
            width=args.image_length,          # 输出图像宽度 (像素)
            latents=init_latents_w,           # 双域水印初始噪声：包含空间域+频域水印
        ).images[0]

        # 4.2 生成无水印的参考图像用于质量对比和评估
        print(f"      4.2 生成参考图像 (无水印)...")
        image_no_w = pipe(
            current_prompt,                    # 使用相同提示词确保公平对比
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,        # 干净的初始噪声：不包含任何水印
        ).images[0]

        # ==================== 结果保存和组织 ====================
        print(f"    步骤5: 保存生成结果")

        # 5.1 保存带水印图像到指定目录
        w_img_path = os.path.join(args.output_path, "w_img", "{}.png".format(i))
        image_w.save(w_img_path)
        print(f"      带水印图像已保存: {w_img_path}")

        # 5.2 保存无水印参考图像到指定目录
        no_w_img_path = os.path.join(args.output_path, "no_w_img", "{}.png".format(i))
        image_no_w.save(no_w_img_path)
        print(f"      参考图像已保存: {no_w_img_path}")

        # 5.3 保存文本提示词以便后续复现和评估
        prompt_path = os.path.join(args.output_path, "prompt", "{}.txt".format(i))
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(current_prompt)
        print(f"      提示词已保存: {prompt_path}")

        print(f"  第 {i+1} 张图像处理完成\n")




if __name__ == '__main__':
    def parse_floats(string):
        """
        解析浮点数字符串的辅助函数

        Args:
            string (str): 包含浮点数的字符串，格式如 "1.0,2.0-3.0,4.0"

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
    parser = argparse.ArgumentParser(description='GaussMarker: 扩散模型的鲁棒双域水印生成器')

    # 基础生成参数
    parser.add_argument('--num', default=1000, type=int,
                       help='要生成的图像数量')
    parser.add_argument('--image_length', default=512, type=int,
                       help='生成图像的尺寸 (正方形)')
    parser.add_argument('--guidance_scale', default=7.5, type=float,
                       help='分类器自由引导强度，控制文本提示词的遵循程度')
    parser.add_argument('--num_inference_steps', default=50, type=int,
                       help='扩散模型推理步数，更多步数通常产生更高质量图像')
    parser.add_argument('--num_inversion_steps', default=None, type=int,
                       help='DDIM反演步数，默认与推理步数相同')
    parser.add_argument('--gen_seed', default=0, type=int,
                       help='生成随机种子的起始值，确保可重现性')

    # 空间域水印参数 (高斯阴影)
    parser.add_argument('--channel_copy', default=1, type=int,
                       help='通道复制因子，控制水印在通道维度的重复')
    parser.add_argument('--w_copy', default=8, type=int,
                       help='宽度复制因子，控制水印在宽度维度的重复')
    parser.add_argument('--h_copy', default=8, type=int,
                       help='高度复制因子，控制水印在高度维度的重复')
    parser.add_argument('--user_number', default=1000000, type=int,
                       help='潜在用户数量，用于计算检测阈值')
    parser.add_argument('--fpr', default=0.000001, type=float,
                       help='期望的假阳性率，控制检测的严格程度')

    # 输入输出路径
    parser.add_argument('--output_path', default='./',
                       help='输出目录路径')
    parser.add_argument('--chacha', action='store_true',
                       help='使用ChaCha20加密算法增强水印安全性')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts',
                       help='HuggingFace数据集路径或本地数据集路径')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base',
                       help='Stable Diffusion模型路径')
    parser.add_argument('--w1_path', default='w1.pth',
                       help='空间域水印保存路径')
    parser.add_argument('--w2_path', default='w2.pth',
                       help='频域水印保存路径')

    # ==================== Tree-Ring频域水印参数 ====================
    # Tree-Ring是一种在频域中嵌入圆环模式的零比特水印技术
    parser.add_argument('--w_seed', default=999999, type=int,
                       help='频域水印生成的随机种子，确保水印模式的一致性')
    parser.add_argument('--w_channel', default=3, type=int,
                       help='频域水印嵌入的通道索引 (-1表示所有通道)')
    parser.add_argument('--w_pattern', default='ring',
                       help='水印模式类型：ring(圆环), seed_ring(种子圆环)')
    parser.add_argument('--w_mask_shape', default='circle',
                       help='水印掩码形状：circle(圆形), signal_circle(信号圆形)')
    parser.add_argument('--w_radius', default=4, type=int,
                       help='频域水印的圆环半径，控制水印在频谱中的分布范围')
    parser.add_argument('--w_measurement', default='l1_complex',
                       help='水印检测的度量方法：l1_complex(L1复数距离)')
    parser.add_argument('--w_injection', default='complex',
                       help='水印注入方式：complex(复数域), seed(种子域), signal(信号域)')
    parser.add_argument('--w_pattern_const', default=0, type=float,
                       help='水印模式常数，用于调节水印强度')

    # ==================== 参数解析和程序执行 ====================
    args = parser.parse_args()

    # 如果未指定反演步数，则使用与推理步数相同的值
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    # 初始化日志系统和输出目录
    set_logger(args.output_path, args)

    # 执行主要的水印图像生成流程
    main(args)