<div align=center>
  
# GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models
</div>

## 1. Contents
- GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
  - [3. Get Start](#3-get-start)

## 2. Introduction

This is the official implementation of paper ***GaussMarker: Robust Dual-Domain Watermarks for Diffusion Models***.

As Diffusion Models (DM) generate increasingly realistic images, related issues such as copyright and misuse have become a growing concern. Watermarking is one of the promising solutions. Existing methods inject the watermark into the single-domain of initial Gaussian noise for generation, which suffers from unsatisfactory robustness. 
This paper presents the first dual-domain DM watermarking approach using a pipelined injector to consistently embed watermarks in both the spatial and frequency domains. To further boost robustness against certain image manipulations and advanced attacks, we introduce a model-independent learnable Gaussian Noise Restorer (GNR) to refine Gaussian noise extracted from manipulated images and enhance detection robustness by integrating the detection scores of both watermarks.
GaussMarker efficiently achieves state-of-the-art performance under eight image distortions and four advanced attacks across three versions of Stable Diffusion with better recall and lower false positive rates, as preferred in real applications.

## 3. Get Start

### 3.1 Installation

To setup the environment of GaussMarker, we use `conda` to manage our dependencies. 

Run the following commands to install GaussMarker:
 ```
conda create -n gaussmarker python=3.9 -y && conda activate gaussmarker
pip install -r requirements.txt
 ```

### 3.2 Generating Watermarked Images
Run:
```
python gaussmarker_gen.py --chacha --num 10 --output_path './gen_10'  --w1_path w1_256.pth
```

The multi-bit and zero-bit watermarks will be saved into `./w1_256.pth` and `./w2.pth`, respectively. The watermarked and un-watermarked images will be saved into `./gen_10/w_img/` and `./gen_10/no_w_img/`, respectively.

### 3.3 Training GNR

Run:
```
python train_GNR.py --train_steps 50000 --r 180 --s_min 1.0 --s_max 1.2 --fp 0.35 --neg_p 0.5 --model_nf 128 --batch_size 32 --num_workers 16 -ed 256bits --w_info_path w1_256.pth
```
After training, the model weight will be saved as `./GNR_256bits/model_final.pth`.

### 3.3 Evaluation

Extract the watermark without any distortion:
```
python gaussmarker_det.py --chacha --model_nf 128 --GNR_path './GNR_bits256/model_final.pth' --input_path './gen_10'  --fpr 0.01 --num 10  --output_path './results/Clean' --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --w1_path './w1_256.pth'
```

Extract the watermark with Rotation 75:
```
python gaussmarker_det.py --chacha --model_nf 128 --GNR_path './GNR_bits256/model_final.pth' --input_path './gen_10'  --fpr 0.01 --num 10  --output_path './results/Rotate75' --affine 1 --affine_params 75,0,1,0 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --w1_path './w1_256.pth'
```