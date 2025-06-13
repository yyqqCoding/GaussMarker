import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy


def set_complex_sign(original_tensor, sign_tensor):
    # 0: real+ imag+
    # 1: real+ imag-
    # 2: real- imag+
    # 3: real- imag-
    real = original_tensor.real.abs()
    imag = original_tensor.imag.abs()
    
    sign_map_real = 1 - 2 * (sign_tensor >= 2).float()
    sign_map_imag = 1 - 2 * ((sign_tensor % 2) == 1).float()
    
    signed_real = real * sign_map_real
    signed_imag = imag * sign_map_imag
    
    new_tensor = torch.complex(signed_real, signed_imag).to(original_tensor.dtype)
    return new_tensor

def extract_complex_signal(complex_tensor):
    # 0: real+ imag+
    # 1: real+ imag-
    # 2: real- imag+
    # 3: real- imag-
    real = complex_tensor.real
    imag = complex_tensor.imag

    sign_map_real = (real <= 0).long()
    sign_map_imag = (imag <= 0).long()
    sign_tensor = 2 * sign_map_real + sign_map_imag
    return sign_tensor

# def set_complex_sign(original_tensor, sign_tensor):
#     # 0: real+
#     # 1: real-
#     real = original_tensor.real.abs()
#     imag = original_tensor.imag
    
#     sign_map_real = 1 - 2 * (sign_tensor >= 0.5).half()
    
#     signed_real = real * sign_map_real
    
#     new_tensor = torch.complex(signed_real, imag).to(original_tensor.dtype)
#     return new_tensor

# def extract_complex_signal(complex_tensor):
#     # 0: real+
#     # 1: real-
#     real = complex_tensor.real

#     sign_map_real = (real <= 0).long()
#     return sign_map_real

# def set_complex_sign(original_tensor, sign_tensor):
#     # 0: imag+
#     # 1: imag-
#     real = original_tensor.real
#     imag = original_tensor.imag.abs()
    
#     sign_map_imag = 1 - 2 * (sign_tensor >= 0.5).half()
    
#     signed_imag = imag * sign_map_imag
    
#     new_tensor = torch.complex(real, signed_imag).to(original_tensor.dtype)
#     return new_tensor

# def extract_complex_signal(complex_tensor):
#     # 0: imag+
#     # 1: imag-
#     imag = complex_tensor.imag

#     sign_map_imag = (imag <= 0).long()
#     return sign_map_imag


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, args, device, radius_list=None):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'signal_circle':
        watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.long).to(device)
        w_i = 1
        for r_i in radius_list:
            tmp_mask = circle_mask(watermarking_mask.shape[-1], r=r_i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(watermarking_mask.shape[1]):
                watermarking_mask[:, j, tmp_mask] = w_i
                w_i += 1
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None, w=None, radius_list=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'signal_ring' in args.w_pattern:
        gt_patch = torch.randint_like(gt_init, low=0, high=1)
        if w is None:
            w = torch.randint(low=0, high=2, size=(args.w_length,))
        w_i = 0
        for r_i in radius_list:
            tmp_mask = circle_mask(gt_init.shape[-1], r=r_i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = w[w_i-1].item()
                w_i += 1
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    elif args.w_injection == 'signal':
        init_latents_w_fft_ = set_complex_sign(init_latents_w_fft, gt_patch)
        init_latents_w_fft[watermarking_mask!=0] = init_latents_w_fft_[watermarking_mask!=0]
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args, w=None):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item() * 0.01
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item() * 0.01
    elif 'signal' in args.w_measurement:
        no_w_real, no_w_imag = reversed_latents_no_w_fft[watermarking_mask].real, reversed_latents_no_w_fft[watermarking_mask].imag
        w_real, w_imag = reversed_latents_w_fft[watermarking_mask].real, reversed_latents_w_fft[watermarking_mask].imag
        target_real, target_imag = target_patch[watermarking_mask].real, target_patch[watermarking_mask].imag

        no_w_real_signal, no_w_imag_signal = (no_w_real>0.).int(), (no_w_imag>0.).int()
        w_real_signal, w_imag_signal = (w_real>0.).int(), (w_imag>0.).int()
        target_real_signal, target_imag_signal = (target_real>0.).int(), (target_imag>0.).int()

        # np_w_real_acc = (no_w_real_signal == target_real_signal).float().mean()
        # no_w_imag_acc = (no_w_imag_signal == target_imag_signal).float().mean()
        # w_real_acc = (w_real_signal == target_real_signal).float().mean()
        # w_imag_acc = (w_imag_signal == target_imag_signal).float().mean()
        # print(np_w_real_acc, no_w_imag_acc)
        # print(w_real_acc, w_imag_acc)

        # no_w_metric = -(np_w_real_acc+no_w_imag_acc).item() / 2
        # w_metric = -(w_real_acc+w_imag_acc).item() / 2

        no_w_metric = -((no_w_real_signal == target_real_signal) * (no_w_imag_signal == target_imag_signal)).float().mean().item()
        w_metric = -((w_real_signal == target_real_signal) * (w_imag_signal == target_imag_signal)).float().mean().item()

        # no_w_metric = -no_w_imag_acc.item()
        # w_metric = -w_imag_acc.item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')
    print(no_w_metric, w_metric)
    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


if __name__ == "__main__":
    gt_init = torch.randint(low=0, high=2, size=(1, 4, 64, 64)).float()
    # gt_init = torch.randn(size=(1, 4, 64, 64)).float()
    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

    watermarking_mask = torch.zeros(gt_patch.shape, dtype=torch.bool)
    np_mask = circle_mask(gt_patch.shape[-1], r=10)
    torch_mask = torch.tensor(np_mask)
    watermarking_mask[:, 0] = torch_mask

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(gt_patch, dim=(-1, -2))).real

    print(gt_init[0, 0])
    print(gt_patch[0, 0])
    print(init_latents_w[0, 0])
