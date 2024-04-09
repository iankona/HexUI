
import torch
import numpy as np

from tqdm import tqdm



def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class EULERSampler:
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


    @torch.no_grad()
    def p_sample(self, tf, X_t, cond, ts, unconditional_guidance_scale, unconditional_conditioning):
        sigma = self.sigmas[ts]
        e_t = self.model.apply_model(X_t/sigma, ts, cond)
        e_t_uncond = self.model.apply_model(X_t/sigma, ts, unconditional_conditioning)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        model_out = e_t 

        denoised = X_t + model_out * (-sigma)
        d = (X_t - denoised) / self.sigmas[ts]
        dt = self.sigmas[tf] - self.sigmas[ts]
        x =  X_t  + dt*d

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

        img = slerp(subseed_strength, noise, subnoise)
        img = noise

        ddpm_num_timesteps = 1000
        num_timesteps = 20
        timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        timesteps = timesteps + 1
        timesteps = list(timesteps)
        timesteps.append(999)
        timesteps = timesteps[::-1]



        img = self.sigma_max * img

        iterator = tqdm(timesteps, desc='Sampling t ddpm') 
        for i, timestep in enumerate(iterator):
            if i== len(timesteps)-1: return img
            ts = torch.full((batch_size,), timestep, device=device, dtype=torch.long) # == timestep
            try:
                tf = timesteps[i+1]
            except:
                tf = 0

            img = self.p_sample(tf, img, cond, ts, unconditional_guidance_scale, unconditional_conditioning)

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


        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img



