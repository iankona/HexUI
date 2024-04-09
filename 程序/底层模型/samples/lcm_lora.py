from collections import deque
import torch
import numpy as np

import tqdm


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])



class lcm_lora_sample:
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

        sigmas = torch.sqrt( (1.0 - self.alphas_cumprod) / self.alphas_cumprod )
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))
        self.register_buffer('log_sigmas', sigmas.log())


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"): attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]


    def get_sigmas(self, n=None):
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))


    def sigma_to_t(self, sigma, quantize=None):
        # quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        t = torch.round(t).long()
        return t
        # return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()




    @torch.no_grad()
    def p_sample(self, sigma, sigma_next, X_t, cond, unconditional_guidance_scale, unconditional_conditioning):
        c_in = 1.0/(sigma**2 + 1) ** 0.5
        input = X_t * c_in
        ts = self.sigma_to_t(sigma)
        e_t = self.model.apply_model(input, ts, cond)
        e_t_uncond = self.model.apply_model(input, ts, unconditional_conditioning)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        model_out = e_t 
        denoised = X_t + model_out * (-sigma)


        x = denoised
        if sigma_next > 0:
            x += sigma_next * torch.randn(x.shape, device=x.device, dtype=torch.float16)
        return x




    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]

        subseed = 1101874262
        torch.manual_seed(subseed)
        subnoise = torch.randn(shape, device=device) # if x_T is None:
        subseed_strength = 0.0


        seed = 1101874262
        torch.manual_seed(seed)
        noise = torch.randn(shape, device=device) # if x_T is None:

        # img = slerp(subseed_strength, noise, subnoise)
        img = noise

        ddpm_num_timesteps = 1000
        num_timesteps = 5
        timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        timesteps = timesteps + 1
        timesteps = list(timesteps)
        timesteps.append(999)
        timesteps = timesteps[::-1]


        img = torch.randn(shape, device=device, dtype=torch.float16) # if x_T is None:
        sigmas = self.get_sigmas(5)
        img = sigmas[0] * img

        # iterator = tqdm(timesteps, desc='Sampling t ddpm') 
        for i in tqdm.trange(len(sigmas) - 1,  desc='Sampling t ddpm'):
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            # ts = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(sigma, sigma_next, img, cond, unconditional_guidance_scale, unconditional_conditioning)

        return img

    @torch.no_grad()
    def sample(self, cond=None, shape=None, **kwargs):
        batch_size = 1
        shape = [1, 4, 48, 80] # 1 是 batch_size
        prompt = r"(best quality:1.2),(ultra highres:1.2),(delicate illustration:1.2),(hyper detailed:1.2),((1girl)),solo,masterpiece,(8k),(detailed eyes:1.3),(big head:1.2),{{{{{lower body}}}}},(True Shadows),milf,lens flare,ray tracing，hair strand,(((wet dress:1))),snow,(detailed eyes:1.1),[[[close to face]]],[[cowboy shot]],((child:1.4)),tsundere,((flowers meadows)),peach blossom，smirk，(full-face blush:1),petite, small breasts,cherry blossoms,high-waist skirt，fine fabric emphasis,off shoulder,True Shadows,Tulle material,wet dress,wet hair,tail,(fox tail),petals,flower,rose petals,wedding dress,snowfield，fox ears，blunt bangs，hair tucking，detached sleeves，perfect hands,（hime cut：1.5）,small breasts ,ribbon，gradient hair，(((pink hair))))，big hair,very long hair，comb over，widow's peak，hair over one eye，(close to face：1.2)，perfect hands, (sakura tree)，seiza，hime cut：1.5，small breasts ,babydoll，tsundere，arrogant，chiffon,fox ears,kyuubi,yukata,falling sakura leaves, highly detailed，partially submerged,((black legwear))，moonlit sky, moon in the sky, night time, ocean shore with sakura trees, falling sakura leaves, highly detailed, ears"
        negative_prompt = r"(DreamArtistBADHAND:1.2), (low quality, worst quality:1.4), <lora:EasyNegative:1.2>,(((extra fingers))), (((fewer fingers))), (((extra legs)))， (((extra hands)))，(((extra hands))), (((extra arms))), (watermark:1.3), (username:1.3), (blurry:1.3), (artist name:1.3), censored, ugly，(((Six fingers)))，(((finger))),fox,2girl,(((bad anatomy))),（Long neck），（Long thumb），(((Shoulder reflection))),"

        prompts = [prompt]
        cond = self.model.get_learned_conditioning(prompts)
        negative_prompts = [negative_prompt]
        unconditional_conditioning = self.model.get_learned_conditioning(negative_prompts)
        unconditional_guidance_scale = 7.0

        cond = cond.half()
        unconditional_conditioning = unconditional_conditioning.half()

        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img



