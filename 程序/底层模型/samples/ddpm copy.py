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


from tqdm import tqdm





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

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / ( 1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef1', torch.tensor( betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef2', torch.tensor( (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=torch.float32))





    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        # print(name, attr)
        setattr(self, name, attr)



    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, X_t, cond, t, unconditional_guidance_scale, unconditional_conditioning): # img, pred_x0, e_t = outs 
 

        model_out = self.model.apply_model(X_t, t, cond) # 不继承torch.nn.Module，会出现not callback

        # x_in = torch.cat([X_t] * 2)
        # t_in = torch.cat([t] * 2)
        # c_in = torch.cat([unconditional_conditioning, cond])
        # e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        # e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        # model_out = e_t





        x_recon = (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, X_t.shape) * X_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, X_t.shape) * model_out
        )

        model_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, X_t.shape) * x_recon +
                extract_into_tensor(self.posterior_mean_coef2, t, X_t.shape) * X_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, X_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, X_t.shape)
        return model_mean, posterior_variance, posterior_log_variance_clipped





    @torch.no_grad()
    def p_sample(self, img, cond, ts, unconditional_guidance_scale, unconditional_conditioning, repeat_noise=False, temperature=1.0, noise_dropout=0.0, ):
        # z = unetmodel(X_t，t)
        # Xt_1 = 1/(√a_t)(X_t-(1-a_t)/(√(1-a_t))z) + 方差*z

        batch_size, *_, device = *img.shape, img.device

        outputs = self.p_mean_variance(X_t=img, cond=cond, t=ts, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
        # img, pred_x0, e_t = outs

        model_mean, _, model_log_variance = outputs
        noise = noise_like(img.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (ts == 0).float()).reshape(batch_size, *((1,) * (len(img.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise




    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        timesteps = 1000
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t ddpm', total=timesteps) # verbose=True

        for i in iterator:
            ts = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts, unconditional_guidance_scale, unconditional_conditioning)

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


