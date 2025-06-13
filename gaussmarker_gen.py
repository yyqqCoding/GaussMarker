import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
from tqdm import tqdm
# from huggingface_hub import login
 
# login()

import torch
import logging

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from utils import set_random_seed, get_dataset
from watermark import Gaussian_Shading_chacha, Gaussian_Shading

from tr_utils import get_watermarking_pattern, get_watermarking_mask, inject_watermark, eval_watermark

def set_logger(workdir, args):
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "w_img"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "no_w_img"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "prompt"), exist_ok=True)
    gfile_stream = open(os.path.join(workdir, 'log.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info(args)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # dataset
    os.makedirs(args.output_path, exist_ok=True)
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    w1_path = args.w1_path
    w2_path = args.w2_path
    if args.chacha:
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"], nonce=w_info["nonce"])
        else:
            watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number)
            _ = watermark.create_watermark_and_return_w_m()
            torch.save({"w": watermark.watermark, "m": watermark.m, "key": watermark.key, "nonce": watermark.nonce}, w1_path)
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)
    else:
        #a simple implement,
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"])
        else:
            watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
            _ = watermark.create_watermark_and_return_w_m()
            torch.save({"w": watermark.watermark, "m": watermark.m, "key": watermark.key}, w1_path)
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)


    if os.path.exists(w2_path):
        gt_patch = torch.load(w2_path).to(device)
    else:
        gt_patch = get_watermarking_pattern(pipe, args, device, shape=(1, 4, 64, 64))
        torch.save(gt_patch, w2_path)
    watermarking_mask = get_watermarking_mask(gt_patch.real, args, device)

    #test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        #generate with watermark
        set_random_seed(seed)
        init_latents_w_gs, _ = watermark.create_watermark_and_return_w_m()
        # init_latents_no_w = init_latents_no_w.to(device).half()
        init_latents_no_w = pipe.get_random_latents()

        init_latents_w = inject_watermark(init_latents_w_gs.float().cuda(), watermarking_mask, gt_patch, args).half()

        image_w = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        ).images[0]
        image_no_w = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
        ).images[0]

        image_w.save(os.path.join(args.output_path, "w_img", "{}.png".format(i)))
        image_no_w.save(os.path.join(args.output_path, "no_w_img", "{}.png".format(i)))
        with open(os.path.join(args.output_path, "prompt", "{}.txt".format(i)), 'w') as f:
            f.write(current_prompt)




if __name__ == '__main__':
    def parse_floats(string):
        try:
            return [[float(x) for x in string_i.split(',')] for string_i in string.split('-')]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float value: {string}")
    parser = argparse.ArgumentParser(description='GaussMarker')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--w_copy', default=8, type=int)
    parser.add_argument('--h_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--w1_path', default='w1.pth')
    parser.add_argument('--w2_path', default='w2.pth')

    # treering
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=3, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=4, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    set_logger(args.output_path, args)
    main(args)