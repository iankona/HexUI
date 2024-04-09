"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class PLMSSampler:
    def __init__(self, model, schedule="linear", **kwargs):
        self.model = model
        self.schedule = schedule

        self.parameterization = "eps"
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        ddpm_num_timesteps = 1000
        plms_num_timesteps = 50
        timestep = (ddpm_num_timesteps+1) // plms_num_timesteps # 1000 // 5 == 200
        plms_timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        plms_timesteps = plms_timesteps + 1 
        # print(plms_timesteps)                                      # [1 201 401 601 801]# [1 144 287 430 573 716 859]
        plms_timesteps = np.asarray([1,201,401,601,801,851,901,951])#
        self.plms_timesteps = plms_timesteps

        # 数学准备 make ddpm betas and alphas and alphas_cumprod
        linear_start = 0.00085
        linear_end = 0.0120
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, ddpm_num_timesteps, dtype=torch.float64) ** 2 )
        betas = betas.numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]) # == alphas_cumprod[:-1].insert(index=0, object=1.0)
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32) ) # torch.nn.Module.register_buffer
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32) )
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32) )


        v_posterior = 0.0
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) + v_posterior * betas
        # above: equal to 1.0 / (1.0/ (1.0- alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32) )


        ddim_alphas = alphas_cumprod[plms_timesteps]
        ddim_alphas_prev = np.asarray([alphas_cumprod[0]] + alphas_cumprod[plms_timesteps[:-1]].tolist())
        ddim_eta = 0.0
        ddim_sigmas = ddim_eta * np.sqrt((1 - ddim_alphas_prev) / (1 - ddim_alphas) * (1 - ddim_alphas / ddim_alphas_prev))


        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt( (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * ( 1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        # print(name, attr)
        setattr(self, name, attr)



    def get_x_prev_and_pred_x0(self, x, e_t, index): # index 
        b, *_, device = *x.shape, x.device

        alphas = self.ddim_alphas # array([0.99829603, 0.7521434 , 0.42288152, 0.15981644, 0.03654652])
        alphas_prev = self.ddim_alphas_prev # array([0.99915   , 0.99829603, 0.7521434 , 0.42288152, 0.15981644])
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas # array([0.0412792 , 0.49785198, 0.75968314, 0.91661528, 0.98155666])
        sigmas = self.ddim_sigmas # array([0.0, 0.0, 0.0, 0.0, 0.0])


        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device) # tensor([[[[0.0365]]]], device='cuda:0') # index = 4
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device) # tensor([[[[0.1598]]]], device='cuda:0')
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device) # tensor([[[[0.]]]], device='cuda:0')
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device) # tensor([[[[0.9816]]]], device='cuda:0')

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() # 


        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = torch.randn(x.shape, device=x.device)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise
        return x_prev, pred_x0


    def get_model_output(self, x, c, t, unconditional_guidance_scale, unconditional_conditioning):
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, unconditional_guidance_scale=1.0, unconditional_conditioning=None, t_next=None):

        e_t = self.get_model_output(x, c, t, unconditional_guidance_scale, unconditional_conditioning)
        e_t_prime = e_t
        # # Pseudo Improved Euler (2nd order)
        # x_prev, pred_x0 = self.get_x_prev_and_pred_x0(x, e_t, index)
        # e_t_next = self.get_model_output(x_prev, c, t_next, unconditional_guidance_scale, unconditional_conditioning)
        # e_t_prime = (e_t + e_t_next) / 2
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(x, e_t_prime, index)
        return x_prev, pred_x0, e_t



    @torch.no_grad()
    def sample(self, ):
        batch_size = 1
        shape = (1, 4, 64, 64)
        print(f'Data shape for PLMS sampling is {shape}')
        prompt = "two cute cat"
        prompts = [prompt]
        conditioning = self.model.get_learned_conditioning(prompts)
        unconditional_guidance_scale = 7.5
        unconditional_conditioning = self.model.get_learned_conditioning(batch_size * [""])

        samples = self.plms_sampling(conditioning, 
                                     shape,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning,
                                     )
        return samples


    @torch.no_grad()
    def plms_sampling(self, 
                      cond, 
                      shape,
                      timesteps=None, 
                      unconditional_guidance_scale=1.0, 
                      unconditional_conditioning=None,
                      ):
        
        device = self.betas.device
        b = shape[0]
        
        img = torch.randn(shape, device=device) # if x_T is None:
        timesteps = self.plms_timesteps

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0] # 5
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # print(index) 4, 3, 2, 1, 0

            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            outs = self.p_sample_plms(img, cond, ts, index=index, 
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      t_next=ts_next)
            img, pred_x0, e_t = outs

        return img



    # @torch.no_grad()
    # def p_sample(self, ts_next, img, cond, ts, unconditional_guidance_scale, unconditional_conditioning):
    #     e_t = self.get_model_output(img, cond, ts, unconditional_guidance_scale, unconditional_conditioning)
    #     model_out = e_t
    #     noise = torch.randn(img.shape, device=img.device)
    #     X_0 = (img - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
    #     u_t = torch.sqrt(self.alphas_cumprod_prev[ts])*X_0 + torch.sqrt(1.0-self.alphas_cumprod_prev[ts]-self.sigmas[ts]**2) * noise
    #     X_t_1 = u_t + self.sigmas[ts]*noise

    #     e_t_next = self.get_model_output(X_t_1, cond, ts_next, unconditional_guidance_scale, unconditional_conditioning)
    #     model_out = e_t_next
    #     noise = torch.randn(img.shape, device=img.device)
    #     X_0 = (img - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
    #     u_t = torch.sqrt(self.alphas_cumprod_prev[ts])*X_0 + torch.sqrt(1.0-self.alphas_cumprod_prev[ts]-self.sigmas[ts]**2) * noise
    #     X_t_1 = u_t + self.sigmas[ts]*noise

    #     e_t_prime = (e_t + e_t_next) / 2
    #     model_out = e_t_prime
    #     noise = torch.randn(img.shape, device=img.device)
    #     X_0 = (img - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])
    #     u_t = torch.sqrt(self.alphas_cumprod_prev[ts])*X_0 + torch.sqrt(1.0-self.alphas_cumprod_prev[ts]-self.sigmas[ts]**2) * noise
    #     X_t_1 = u_t + self.sigmas[ts]*noise
    #     return X_t_1