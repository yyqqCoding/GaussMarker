"""
GaussMarker: 高斯噪声恢复器 (GNR) 分布式训练模块

本模块实现了基于DistributedDataParallel (DDP) 的多GPU训练，
相比DataParallel具有更好的性能和扩展性。

使用方法:
# 双GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train_GNR_ddp.py [args]

# 或使用torchrun (推荐)
torchrun --nproc_per_node=2 train_GNR_ddp.py [args]
"""

import os
import argparse
import logging
from tqdm import tqdm
import datetime
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.utils import save_image

# GaussMarker核心组件
from watermark import *
from unet.unet_model import UNet

def setup_ddp(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def flip_tensor(tensor, flip_prob):
    """随机翻转张量中的二进制值，模拟噪声攻击"""
    random_tensor = torch.rand(tensor.size())
    flipped_tensor = tensor.clone()
    flipped_tensor[random_tensor < flip_prob] = 1 - flipped_tensor[random_tensor < flip_prob]
    return flipped_tensor

def Affine_random(latent, r, t, s_min, s_max, sh):
    """对潜在变量应用随机仿射变换，模拟几何攻击"""
    config = dict(degrees=(-r, r), translate=(t, t), scale_ranges=(s_min, s_max), 
                  shears=(-sh, sh), img_size=latent.shape[-2:])
    r, (tx, ty), s, (shx, shy) = transforms.RandomAffine.get_params(**config)
    
    b, c, w, h = latent.shape
    new_latent = transforms.functional.affine(
        latent.view(b*c, 1, w, h), angle=r, translate=(tx, ty), 
        scale=s, shear=(shx, shy), fill=999999
    )
    new_latent = new_latent.view(b, c, w, h)
    
    mask = (new_latent[:, :1, ...] < 999998).float()
    new_latent = new_latent * mask + torch.randint_like(new_latent, low=0, high=2) * (1-mask)
    
    return new_latent, (r, tx, ty, s)

class LatentDataset_m_DDP(IterableDataset):
    """分布式训练专用的潜在空间数据集类"""
    
    def __init__(self, watermark, args, rank, world_size):
        super(LatentDataset_m_DDP, self).__init__()
        self.watermark = watermark
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # 设置每个进程的随机种子
        torch.manual_seed(args.gen_seed + rank)
        np.random.seed(args.gen_seed + rank)
        
        if self.args.num_watermarks > 1:
            t_m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
            o_m = torch.randint(low=0, high=2, size=(self.args.num_watermarks-1, 4, 64, 64))
            self.m = torch.cat([t_m, o_m])
        else:
            self.m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
        
        self.args.neg_p = 1 / (1 + self.args.num_watermarks)
    
    def __iter__(self):
        while True:
            random_index = torch.randint(0, self.args.num_watermarks, (1,)).item()
            latents_m = self.m[random_index:random_index+1]
            false_latents_m = torch.randint_like(latents_m, low=0, high=2)
            
            if np.random.rand() > self.args.neg_p:
                aug_latents_m, params = Affine_random(
                    latents_m.float(), self.args.r, self.args.t, 
                    self.args.s_min, self.args.s_max, self.args.sh
                )
                aug_latents_m = flip_tensor(aug_latents_m, self.args.fp)
                yield aug_latents_m.squeeze(0).float(), latents_m.squeeze(0).float()
            else:
                aug_false_latents_m, params = Affine_random(
                    false_latents_m.float(), self.args.r, self.args.t,
                    self.args.s_min, self.args.s_max, self.args.sh
                )
                aug_false_latents_m = flip_tensor(aug_false_latents_m, self.args.fp)
                yield aug_false_latents_m.squeeze(0).float(), aug_false_latents_m.squeeze(0).float()

def set_logger(workdir, args, rank):
    """配置分布式训练的日志系统"""
    if rank == 0:  # 只在主进程中设置日志
        os.makedirs(workdir, exist_ok=True)
        gfile_stream = open(os.path.join(workdir, 'log.txt'), 'a')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        logging.info(f"分布式训练配置: {args}")

def main_ddp(rank, world_size, args):
    """分布式训练主函数"""
    # 初始化分布式环境
    setup_ddp(rank, world_size)
    
    # 设置日志
    set_logger(args.output_path, args, rank)
    
    if rank == 0:
        print(f"开始分布式GNR训练")
        print(f"  世界大小: {world_size}")
        print(f"  当前进程: {rank}")
        print(f"  总训练步数: {args.train_steps}")
        print(f"  每GPU批次大小: {args.batch_size}")
        print(f"  有效批次大小: {args.batch_size * world_size}")
    
    # 模型初始化
    model = UNet(4, 4, nf=args.model_nf).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # 水印配置加载
    if os.path.exists(args.w_info_path):
        w_info = torch.load(args.w_info_path)
        watermark = Gaussian_Shading_chacha(
            args.channel_copy, args.w_copy, args.h_copy, 
            args.fpr, args.user_number,
            watermark=w_info["w"], m=w_info["m"], 
            key=w_info["key"], nonce=w_info["nonce"]
        )
    else:
        raise FileNotFoundError(f"水印配置文件不存在: {args.w_info_path}")
    
    # 数据集和数据加载器
    dataset = LatentDataset_m_DDP(watermark, args, rank, world_size)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers // world_size
    )
    
    # 训练循环
    if rank == 0:
        print(f"开始训练...")
    
    for i, batch in enumerate(data_loader):
        if i > args.train_steps:
            break
            
        x, y = batch
        x = x.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True).float()
        
        # 混合精度前向传播
        with autocast():
            pred = model(x)
            loss = criterion(pred, y)
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 日志记录 (仅主进程)
        if rank == 0 and i % 200 == 0:
            print(f"步骤 {i}: 损失 = {loss.item():.6f}")
            logging.info(f"训练步骤 {i}, 损失: {loss.item():.6f}")
            
            # 保存检查点
            if i % 2000 == 0:
                torch.save(model.module.state_dict(), 
                          os.path.join(args.output_path, f"checkpoint_{i}.pth"))
    
    # 保存最终模型 (仅主进程)
    if rank == 0:
        torch.save(model.module.state_dict(), 
                  os.path.join(args.output_path, "model_final.pth"))
        print("分布式训练完成！")
    
    cleanup_ddp()

if __name__ == "__main__":
    # 解析命令行参数 (复用原有参数)
    parser = argparse.ArgumentParser(description='GaussMarker分布式GNR训练')
    
    # 基础参数
    parser.add_argument('--train_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_nf', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gen_seed', default=0, type=int)
    
    # 水印参数
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--w_copy', default=8, type=int)
    parser.add_argument('--h_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--num_watermarks', type=int, default=1)
    
    # 数据增强参数
    parser.add_argument('--r', type=float, default=180)
    parser.add_argument('--t', type=float, default=0)
    parser.add_argument('--s_min', type=float, default=1.0)
    parser.add_argument('--s_max', type=float, default=1.2)
    parser.add_argument('--sh', type=float, default=0)
    parser.add_argument('--fp', type=float, default=0.35)
    parser.add_argument('--neg_p', type=float, default=0.5)
    
    # 路径参数
    parser.add_argument('--output_path', default='./GNR_DDP')
    parser.add_argument('--w_info_path', default='w1_256.pth')
    parser.add_argument('--exp_description', '-ed', default="")
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.exp_description:
        args.output_path = args.output_path + '_' + args.exp_description
    
    # 获取GPU数量
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("警告: 检测到少于2个GPU，建议使用单GPU训练脚本")
    
    # 启动分布式训练
    torch.multiprocessing.spawn(
        main_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
