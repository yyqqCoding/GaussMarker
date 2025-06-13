import torch
import torch.nn.functional as F
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes


class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, w_factor, h_factor, fpr, user_number, watermark=None, key=None, nonce=None, m=None):
        self.ch = ch_factor
        self.w = w_factor
        self.h = h_factor
        self.nonce = nonce
        self.key = key
        self.watermark = watermark
        self.m = m
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.w * self.h)

        self.threshold = 1 if self.h == 1 and self.w == 1 and self.ch == 1 else self.ch * self.w * self.h // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z

    def create_watermark_and_return_w(self):
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.w, 64 // self.h])
            sd = self.watermark.repeat(1,self.ch,self.w,self.h)
            m = self.stream_key_encrypt(sd.flatten().numpy())
            self.m = torch.from_numpy(m).reshape(1, 4, 64, 64)
        w = self.truncSampling(self.m)
        return w
    
    # def create_watermark_and_return_w_sd(self):
    #     self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw])
    #     sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
    #     m = self.stream_key_encrypt(sd.flatten().numpy())
    #     w = self.truncSampling(m)
    #     return w, sd

    def create_watermark_and_return_w_m(self):
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.w, 64 // self.h])
            sd = self.watermark.repeat(1, self.ch, self.w, self.h)
            self.m = self.stream_key_encrypt(sd.flatten().numpy())
        w = self.truncSampling(self.m)
        return w, torch.from_numpy(self.m).reshape(1, 4, 64, 64)
    
    def stream_key_encrypt(self, sd):
        if self.key is None or self.nonce is None:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor
    
    # def stream_key_encrypt(self, sd):
    #     return sd

    # def stream_key_decrypt(self, reversed_m):
    #     return torch.from_numpy(reversed_m).reshape(1, 4, 64, 64).to(torch.uint8)

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        w_stride = 64 // self.w
        h_stride = 64 // self.h
        ch_list = [ch_stride] * self.ch
        w_list = [w_stride] * self.w
        h_list = [h_stride] * self.h
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(w_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote
    
    def pred_m_from_latent(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        return reversed_m
    
    def pred_w_from_latent(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        return reversed_watermark
    
    def pred_w_from_m(self, reversed_m):
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        return reversed_watermark



class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote


if __name__ == "__main__":
    # wm = Gaussian_Shading_chacha(1, 8, 0.000001, 100000)
    # _ = wm.create_watermark_and_return_w_m()
    # print(wm.watermark)
    # print(wm.key, type(wm.key))
    # print(wm.nonce, type(wm.nonce))
    # torch.save({"w": wm.watermark, "m": wm.m, "key": wm.key, "nonce": wm.nonce}, 'info.pth')
    info = torch.load('info.pth')
    wm = Gaussian_Shading_chacha(1, 8, 0.000001, 100000, watermark=info["w"], m=info["m"], key=info["key"], nonce=info["nonce"])
    _, m = wm.create_watermark_and_return_w_m()
    # print(m)


