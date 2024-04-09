"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm



class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

        self.parameterization = "eps"
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
    
        linear_start = 0.00085
        linear_end = 0.0120
        ddpm_num_timestep = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, ddpm_num_timestep, dtype=torch.float64) ** 2 # beta_schedule="linear",
        betas = betas.numpy() 

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))        
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        eta = 0.0 # [0, 1] 之间
        sigmas = eta * torch.sqrt( (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas )
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))



    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    @torch.no_grad()
    def sample(self):
        batch_size = 1
        shape = (1, 4, 64, 64)
        print(f'Data shape for DDIM sampling is {shape}')
        prompt = "two cute cat"
        prompts = [prompt]
        cond = self.model.get_learned_conditioning(prompts)
        unconditional_guidance_scale = 7.5
        unconditional_conditioning = self.model.get_learned_conditioning(batch_size * [""])

        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device) # if x_T is None:

        ddpm_num_timesteps = 1000
        num_timesteps = 50
        timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        timesteps = timesteps + 1 
        timesteps = timesteps[::-1]
        # num_timestep = 676
        # timesteps = [i for i in range(num_timestep)][::-1]

        iterator = tqdm(timesteps, desc='DDIM Sampler')

        for i, timestep in enumerate(iterator):
            ts = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            try:
                tf = timesteps[i+1]
            except:
                tf = 0
            img = self.p_sample(tf,
                                img, cond, ts, 
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample(self, tf, X_t, cond, ts, unconditional_guidance_scale, unconditional_conditioning):

        e_t_uncond = self.model.apply_model(X_t, ts, unconditional_conditioning)
        e_t = self.model.apply_model(X_t, ts, cond)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        model_out = e_t 

        noise = torch.randn(X_t.shape, device=X_t.device)
        if ts[0] == 1: return X_t
        X_0 = (X_t - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
        u_t_f = torch.sqrt(self.alphas_cumprod[tf])*X_0 + torch.sqrt(1.0-self.alphas_cumprod[tf]-self.sigmas[ts]**2) * noise

        X_t_f = u_t_f + self.sigmas[ts]*noise
        return X_t_f


class DDIMSampler_noise(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

        self.parameterization = "eps"
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
    
        linear_start = 0.00085
        linear_end = 0.0120
        ddpm_num_timestep = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, ddpm_num_timestep, dtype=torch.float64) ** 2 # beta_schedule="linear",
        betas = betas.numpy() 

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))        
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        eta = 1.0 # [0, 1] 之间
        sigmas = eta * torch.sqrt( (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas )
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))



    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    @torch.no_grad()
    def sample(self):
        batch_size = 1
        shape = (1, 4, 64, 64)
        print(f'Data shape for DDIM sampling is {shape}')
        prompt = "two cute cat"
        prompts = [prompt]
        cond = self.model.get_learned_conditioning(prompts)
        unconditional_guidance_scale = 7.5
        unconditional_conditioning = self.model.get_learned_conditioning(batch_size * [""])

        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device) # if x_T is None:

        # ddpm_num_timesteps = 1000
        # num_timesteps = 30
        # timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        # timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        # timesteps = timesteps + 1 
        # timesteps = timesteps[::-1]
        num_timestep = 676
        timesteps = [i for i in range(num_timestep)][::-1]

        iterator = tqdm(timesteps, desc='DDIM Sampler')

        for timestep in iterator:
            ts = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts, 
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample(self, X_t, cond, ts, unconditional_guidance_scale, unconditional_conditioning):

        e_t_uncond = self.model.apply_model(X_t, ts, unconditional_conditioning)
        e_t = self.model.apply_model(X_t, ts, cond)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        model_out = e_t 

        noise = torch.randn(X_t.shape, device=X_t.device)
        
        X_0 = (X_t - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
        u_t = torch.sqrt(self.alphas_cumprod_prev[ts])*X_0 + torch.sqrt(1.0-self.alphas_cumprod_prev[ts]-self.sigmas[ts]**2) * noise
        X_t_1 = u_t + self.sigmas[ts]*noise
        return X_t_1


class DDIMSampler_noise_noise(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model

        self.parameterization = "eps"
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
    
        linear_start = 0.00085
        linear_end = 0.0120
        ddpm_num_timestep = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, ddpm_num_timestep, dtype=torch.float64) ** 2 # beta_schedule="linear",
        betas = betas.numpy() 

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))        
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        eta = 1.0 # [0, 1] 之间
        sigmas = eta * torch.sqrt( (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * self.betas )
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))



    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    @torch.no_grad()
    def sample(self):
        batch_size = 1
        shape = (1, 4, 64, 64)
        print(f'Data shape for DDIM sampling is {shape}')
        prompt = "two cute cat"
        prompts = [prompt]
        cond = self.model.get_learned_conditioning(prompts)
        unconditional_guidance_scale = 7.5
        unconditional_conditioning = self.model.get_learned_conditioning(batch_size * [""])

        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device) # if x_T is None:

        # ddpm_num_timesteps = 1000
        # num_timesteps = 30
        # timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        # timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        # timesteps = timesteps + 1 
        # timesteps = timesteps[::-1]
        num_timestep = 676
        timesteps = [i for i in range(num_timestep)][::-1]

        iterator = tqdm(timesteps, desc='DDIM Sampler')

        for timestep in iterator:
            ts = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts, 
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample(self, X_t, cond, ts, unconditional_guidance_scale, unconditional_conditioning):

        e_t_uncond = self.model.apply_model(X_t, ts, unconditional_conditioning)
        e_t = self.model.apply_model(X_t, ts, cond)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        model_out = e_t 

        noise1 = torch.randn(X_t.shape, device=X_t.device)
        noise2 = torch.randn(X_t.shape, device=X_t.device)
        X_0 = (X_t - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
        u_t = torch.sqrt(self.alphas_cumprod_prev[ts])*X_0 + torch.sqrt(1.0-self.alphas_cumprod_prev[ts]-self.sigmas[ts]**2) * noise1
        X_t_1 = u_t + self.sigmas[ts] * noise2
        return X_t_1


