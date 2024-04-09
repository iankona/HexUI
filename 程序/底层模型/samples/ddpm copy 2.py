"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np


import tqdm





def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

# a = torch.nn.Module # SyntaxError('invalid syntax', ('<string>', 1, 1, "<class 'torch.nn.modules.module.Module'>", 1, 2))
class DDPMSampler:
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

        # self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.parameterization = "eps"
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
    
        self.v_posterior = 0.0

        linear_start = 0.00085
        linear_end = 0.0120
        ddpm_num_timestep = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, ddpm_num_timestep, dtype=torch.float64) ** 2 # beta_schedule="linear",
        betas = betas.numpy() 

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])


        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer('recip_sqrt_alphas', torch.tensor(1.0/np.sqrt(alphas), dtype=torch.float32))

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('sqrt_betas', torch.tensor(np.sqrt(betas), dtype=torch.float32))

        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32))

        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        # # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / ( 1.0 - alphas_cumprod) + self.v_posterior * betas
        # self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))
        # # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # self.register_buffer('posterior_log_variance_clipped', torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32))
        # self.register_buffer('posterior_mean_coef1', torch.tensor( betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=torch.float32))
        # self.register_buffer('posterior_mean_coef2', torch.tensor( (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=torch.float32))





    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        # print(name, attr)
        setattr(self, name, attr)




    @torch.no_grad()
    def p_sample(self, X_t, cond, ts, index, unconditional_guidance_scale, unconditional_conditioning, repeat_noise=False, temperature=1.0, noise_dropout=0.0, ):
        # model_out = self.model.apply_model(X_t, ts, cond) # 不继承torch.nn.Module，会出现not callback
        # x_in = torch.cat([X_t] * 2)
        # t_in = torch.cat([ts] * 2)
        # c_in = torch.cat([unconditional_conditioning, cond])
        e_t_uncond = self.model.apply_model(X_t, ts, unconditional_conditioning)
        e_t = self.model.apply_model(X_t, ts, cond)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        model_out = e_t

        noise = torch.randn(X_t.shape, device=X_t.device)
        if index == 0:
            X_t_1 = self.recip_sqrt_alphas[index]*(X_t - (self.betas[index])/(self.sqrt_one_minus_alphas_cumprod[index])*model_out) + self.sqrt_betas[index]*noise
        else:
            X_t_1 = self.recip_sqrt_alphas[index]*(X_t - (self.betas[index])/(self.sqrt_one_minus_alphas_cumprod[index])*model_out)

        return X_t_1




    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        timesteps = 1000
        iterator_list = [i for i in range(0, timesteps)][::-1] # for i in iterator_index: print(i) # 999 -> 0
        iterator_tqdm = tqdm.tqdm(iterator_list, desc='Sampling t ddpm') 
        for i in iterator_tqdm: 
            ts = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts, i, unconditional_guidance_scale, unconditional_conditioning)

        return img

    @torch.no_grad()
    def sample(self, cond=None, shape=None, **kwargs):
        batch_size = 1
        shape = [1, 4, 64, 64] # 1 是 batch_size
        prompt = "two cute cat"
        prompts = [prompt]
        cond = self.model.get_learned_conditioning(prompts)
        unconditional_guidance_scale = 7.5
        unconditional_conditioning = self.model.get_learned_conditioning(batch_size * [""])

        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img


