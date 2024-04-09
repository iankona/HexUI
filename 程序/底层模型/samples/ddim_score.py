"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from 程序.底层模型.clip_encoder import prompt_parser
















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

        eta = 1.0 # [0, 1] 之间
        sigmas = eta * torch.sqrt( (1.0 - self.alphas_cumprod_prev.clone()) / (1.0 - self.alphas_cumprod.clone()) * self.betas.clone() )
        self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))



    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    @torch.no_grad()
    def sample(self):
        batch_size = 1
        shape = (1, 4, 48, 80)


        print()

        prompt = r"(best quality:1.2),(ultra highres:1.2),(delicate illustration:1.2),(hyper detailed:1.2),((1girl)),solo,masterpiece,(8k),(detailed eyes:1.3),(big head:1.2),{{{{{lower body}}}}},(True Shadows),milf,lens flare,ray tracing，hair strand,(((wet dress:1))),snow,(detailed eyes:1.1),[[[close to face]]],[[cowboy shot]],((child:1.4)),tsundere,((flowers meadows)),peach blossom，smirk，(full-face blush:1),petite, small breasts,cherry blossoms,high-waist skirt，fine fabric emphasis,off shoulder,True Shadows,Tulle material,wet dress,wet hair,tail,(fox tail),petals,flower,rose petals,wedding dress,snowfield，fox ears，blunt bangs，hair tucking，detached sleeves，perfect hands,（hime cut：1.5）,small breasts ,ribbon，gradient hair，(((pink hair))))，big hair,very long hair，comb over，widow's peak，hair over one eye，(close to face：1.2)，perfect hands, (sakura tree)，seiza，hime cut：1.5，small breasts ,babydoll，tsundere，arrogant，chiffon,fox ears,kyuubi,yukata,falling sakura leaves, highly detailed，partially submerged,((black legwear))，moonlit sky, moon in the sky, night time, ocean shore with sakura trees, falling sakura leaves, highly detailed, ears"
        # # prompts = [prompt]
        # # cond = self.model.get_learned_conditioning(prompts)
        # # print(cond.shape)
        negative_prompt = r"(DreamArtistBADHAND:1.2), (low quality, worst quality:1.4), <lora:EasyNegative:1.2>,(((extra fingers))), (((fewer fingers))), (((extra legs)))， (((extra hands)))，(((extra hands))), (((extra arms))), (watermark:1.3), (username:1.3), (blurry:1.3), (artist name:1.3), censored, ugly，(((Six fingers)))，(((finger))),fox,2girl,(((bad anatomy))),（Long neck），（Long thumb），(((Shoulder reflection))),"
        
        # uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, p.steps, cached_uc)
        # c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, cached_c)

        # x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)


        # conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        # unconditional_conditioning = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, self.step)
        # prompt = r"a cute girl"
        # negative_prompt = r""

        steps = 20
        prompts = [prompt]
        negative_prompts = [negative_prompt]

        # cond = self.model.get_learned_conditioning(prompts)
        # un_cond = self.model.get_learned_conditioning(negative_prompts)

        conds = prompt_parser.get_multicond_learned_conditioning(self.model, prompts, steps)
        un_conds = prompt_parser.get_learned_conditioning(self.model, negative_prompts, steps)
        # a = conds.batch[0][0] # ComposableScheduledPromptConditioning
        # b = a.schedules[0]
        # cond = b.cond
        # print(cond.shape) # torch.Size([308, 768])


        # a = un_conds.batch[0][0] # ComposableScheduledPromptConditioning
        # b = a.schedules[0]
        # un_cond = b.cond
        # print(un_cond.shape) # torch.Size([154, 768])

        unconditional_guidance_scale = 7.0


        # cond = conds["c_crossattn"][0]
        # unconditional_conditioning = un_conds["c_crossattn"][0]
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(conds, steps)
        unconditional_conditioning = prompt_parser.reconstruct_cond_batch(un_conds, steps)
        cond = tensor


        # cond = cond.reshape([1, cond.shape[0], cond.shape[1]])
        # unconditional_conditioning = un_cond.reshape([1, un_cond.shape[0], un_cond.shape[1]])
        if unconditional_conditioning.shape[1] < cond.shape[1]:
            last_vector = unconditional_conditioning[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
            unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
        elif unconditional_conditioning.shape[1] > cond.shape[1]:
            unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]




        img = self.p_sample_loop(cond, shape, unconditional_guidance_scale, unconditional_conditioning)
        return img

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, unconditional_guidance_scale, unconditional_conditioning):
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device) # if x_T is None:

        ddpm_num_timesteps = 1000
        num_timesteps = 20
        timestep = ddpm_num_timesteps // num_timesteps # 1000 // 5 == 200
        timesteps = np.asarray(list(range(0, ddpm_num_timesteps, timestep))) # [0 200 400 600 800] 
        timesteps = timesteps + 1 
        timesteps = timesteps[::-1]
        # num_timestep = 676
        # timesteps = [i for i in range(num_timestep)][::-1]

        # cond = conds[0][0].cond.reshape([1, 77, 768])
        # print(cond.shape)
        # cond = cond # torch.Size([77, 768])  torch.Size([77, 320])
        # unconditional_conditioning = unconditional_conditionings[0][0].cond.reshape([1, 77, 768]) # 和什么东西一样
        iterator = tqdm(timesteps, desc='DDIM Sampler')

        for i, timestep, in enumerate(iterator):
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


        alphas_cumprod_t = self.alphas_cumprod[ts] # tensor([[[[0.0365]]]], device='cuda:0') # index = 4
        alphas_cumprod_next = self.alphas_cumprod[tf] # tensor([[[[0.1598]]]], device='cuda:0')
        sigma_t = self.sigmas[ts]  # tensor([[[[0.]]]], device='cuda:0')



    # def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    #     # select alphas for computing the variance schedule
    #     alphas = alphacums[ddim_timesteps]
    #     alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    #     # according the the formula provided in https://arxiv.org/abs/2010.02502
    #     sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    #     if verbose:
    #         print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
    #         print(f'For the chosen value of eta, which is {eta}, '
    #               f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    #     return sigmas, alphas, alphas_prev



        # alphas =  self.ddim_alphas
        # alphas_prev = self.ddim_alphas_prev
        # sqrt_one_minus_alphas =  self.ddim_sqrt_one_minus_alphas
        # sigmas =  self.ddim_sigmas
        # # select parameters corresponding to the currently considered timestep
        # a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        # a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        # sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() # 
        # dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # noise = torch.randn(x.shape, device=x.device)
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise

        # noise = torch.randn(X_t.shape, device=X_t.device)
        # X_0 = (X_t - torch.sqrt(1.0-alphas_cumprod_t)*model_out) / torch.sqrt(alphas_cumprod_t)
        # u_t_f = torch.sqrt(alphas_cumprod_next)*X_0 + torch.sqrt(1.0-alphas_cumprod_next-sigma_t**2) * noise
        # X_t_f = u_t_f + sigma_t*noise
        # noise = torch.randn(X_t.shape, device=X_t.device)
        # if ts[0] == 1: return X_t
        X_0 = (X_t - torch.sqrt(1.0-self.alphas_cumprod[ts])*model_out) / torch.sqrt(self.alphas_cumprod[ts])

        d_x = torch.sqrt(1.0-self.alphas_cumprod[tf]-self.sigmas[ts]**2) * model_out

        noise = torch.randn(X_t.shape, device=X_t.device)
        u_t_f = torch.sqrt(self.alphas_cumprod[tf])*X_0 + torch.sqrt(1.0-self.alphas_cumprod[tf]-self.sigmas[ts]**2) * model_out
        X_t_f = u_t_f + self.sigmas[ts]*noise


        return X_t_f










# def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
#     eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
#     xs = []

#     # if we have multiple seeds, this means we are working with batch size>1; this then
#     # enables the generation of additional tensors with noise that the sampler will use during its processing.
#     # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
#     # produce the same images as with two batches [100], [101].
#     if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or eta_noise_seed_delta > 0):
#         sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
#     else:
#         sampler_noises = None

#     for i, seed in enumerate(seeds):
#         noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

#         subnoise = None
#         if subseeds is not None:
#             subseed = 0 if i >= len(subseeds) else subseeds[i]

#             subnoise = devices.randn(subseed, noise_shape)

#         # randn results depend on device; gpu and cpu get different results for same seed;
#         # the way I see it, it's better to do this on CPU, so that everyone gets same result;
#         # but the original script had it like this, so I do not dare change it for now because
#         # it will break everyone's seeds.
#         noise = devices.randn(seed, noise_shape)

#         if subnoise is not None:
#             noise = slerp(subseed_strength, noise, subnoise)

#         if noise_shape != shape:
#             x = devices.randn(seed, shape)
#             dx = (shape[2] - noise_shape[2]) // 2
#             dy = (shape[1] - noise_shape[1]) // 2
#             w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
#             h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
#             tx = 0 if dx < 0 else dx
#             ty = 0 if dy < 0 else dy
#             dx = max(-dx, 0)
#             dy = max(-dy, 0)

#             x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
#             noise = x

#         if sampler_noises is not None:
#             cnt = p.sampler.number_of_needed_noises(p)

#             if eta_noise_seed_delta > 0:
#                 torch.manual_seed(seed + eta_noise_seed_delta)

#             for j in range(cnt):
#                 sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

#         xs.append(noise)

#     if sampler_noises is not None:
#         p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]

#     x = torch.stack(xs).to(shared.device)
#     return x


