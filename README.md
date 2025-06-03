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

To appear.
