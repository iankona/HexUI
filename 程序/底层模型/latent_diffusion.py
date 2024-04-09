

import numpy as np
from einops import rearrange, repeat

import torch
from 程序.底层模型.auto_encoder.distributions import DiagonalGaussianDistribution
from 程序.底层模型.denoise_model.unet import UNetModel
from 程序.底层模型.auto_encoder.AutoencoderKL import AutoencoderKL
from 程序.底层模型.clip_encoder.modules import FrozenCLIPEmbedder, FrozenCLIPEmbedderWithCustomWords

# from 程序.底层模型.sd_hijack.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords


class 类(torch.nn.Module):
    """main class"""
    def __init__(self,
                 vae_config=None, # AutoencoderKL
                 clip_config=None, # FrozenCLIPEmbedder
                 unet_config=None, # Unet
                 num_timesteps_cond=None, # 1
                 cond_stage_key="image", # txt
                 cond_stage_trainable=False, # false
                 concat_mode=True, # True
                 cond_stage_forward=None, # None
                 conditioning_key=None, #  'crossattn'
                 scale_factor=1.0, # 0.18215
                 scale_by_std=False, # False
                 *args, # 
                 **kwargs, #  {'linear_start': 0.00085, 'linear_end': 0.012, 'log_every_t': 200, 'timesteps': 1000, 'first_stage_key': 'jpg', 'image_size': 64, 'channels': 4, 'monitor': 'val/loss_simple_ema', 'use_ema': False, 'scheduler_config': {'target': 'diffusion.lr_scheduler.LambdaLinearScheduler', 'params': {'warm_up_steps': [10000], 'cycle_lengths': [10000000000000], 'f_start': [1e-06], 'f_max': [1.0], 'f_min': [1.0]}}, 'unet_config': {'target': 'DenoiseModel.unet.UNetModel', 'params': {'image_size': 32, 'in_channels': 4, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_heads': 8, 'use_spatial_transformer': True, 'transformer_depth': 1, 'context_dim': 768, 'use_checkpoint': True, 'legacy': False}}}
                ):
        self.device = torch.device("cuda")
        super().__init__()
  
        # for backwards compatibility after implementation of UnetWrapper

        self.scale_factor = scale_factor

        self.first_stage_model = self.instantiate_vae_stage(vae_config)
        self.cond_stage_model = self.instantiate_clip_stage(clip_config)
        self.model = self.instantiate_unet_stage(unet_config)
        self.cond_stage_model


    def init_from_ckpt(self, filepath):
        filepath = r"D:\losd\Gf_style2.ckpt"
        state_dict_2 = torch.load(filepath, map_location="cpu")

        filepath = r"D:\lora\GuoFeng3.ckpt"
        state_dict_3 = torch.load(filepath, map_location="cpu")

        state_dict = state_dict_2
        # 模型融合
        # state_dict = {}
        # for name in state_dict_2:
        #     tensor3 = state_dict_3.get(name, None)
        #     if tensor3 == None: continue
        #     tensor2 = state_dict_2.get(name, None)
        #     tensor = tensor2 + tensor3*0.01
        #     state_dict[name] = tensor
 
        vae_state_dict = {}
        clip_state_dict = {}
        for name, value in state_dict.items():
            if "first_stage_model." in name: vae_state_dict[name] = value
            if "cond_stage_model.transformer." in name: clip_state_dict[name] = value
        for name in vae_state_dict: state_dict.pop(name)
        for name in clip_state_dict: state_dict.pop(name)
        self.load_state_dict(state_dict, strict=False) # 注释掉，只会生成噪音图 # load_state_dict是torch.nn.Module的函数

        state_dict = {} 
        for name, value in  vae_state_dict.items(): state_dict[name[18:]] = value
        self.first_stage_model.load_state_dict(state_dict, strict=False)

        state_dict = {} 
        for name, value in  clip_state_dict.items(): state_dict[name[29:]] = value

        # self.cond_stage_model.transformer.load_state_dict(state_dict, strict=False) 
        self.cond_stage_model.wrapped.transformer.load_state_dict(state_dict, strict=False) 

        self.first_stage_model = self.first_stage_model.half()
        self.cond_stage_model = self.cond_stage_model.half()
        self.model = self.model.half()


    def instantiate_vae_stage(self, config):
        kwargs = config.get("params", dict())
        model = AutoencoderKL(**kwargs)
        model.to("cpu")
        model.eval() 
        for param in model.parameters():
            param.requires_grad = False
        return model.half()

    def instantiate_clip_stage(self, config):
        kwargs = config.get("params", dict())
        model = FrozenCLIPEmbedder(**kwargs)
        model.eval()
        model = FrozenCLIPEmbedderWithCustomWords(model) # 保证和webui效果一致
        model.to("cpu")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model.half()

    def instantiate_unet_stage(self, config):
        model = UnetWrapper(config, conditioning_key='crossattn')
        return model.half()


    def get_learned_conditioning(self, c):
        c = self.cond_stage_model(c)
        # c = self.cond_stage_model.encode(c)
        return c


    def get_vae_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z



    @torch.no_grad()
    def decode_vae(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_vae(self, x):
        return self.first_stage_model.encode(x)


    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        x_recon = self.model(x_noisy, t, **cond) # 不继承torch.nn.Module，会出现not callback

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


class UnetWrapper(torch.nn.Module):
    def __init__(self, unet_config, conditioning_key):
        super().__init__()
        kwargs = unet_config.get("params", dict())
        model = UNetModel(**kwargs)       # Unet
        self.diffusion_model = model
        self.conditioning_key = conditioning_key        # crossattn
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out






























# def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
#                 # linear 1000 0.00085 0.012 0.008

    
#     if schedule == "linear":
#         betas = (
#                 torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
#         )

#     elif schedule == "cosine":
#         timesteps = (
#                 torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
#         )
#         alphas = timesteps / (1 + cosine_s) * np.pi / 2
#         alphas = torch.cos(alphas).pow(2)
#         alphas = alphas / alphas[0]
#         betas = 1 - alphas[1:] / alphas[:-1]
#         betas = np.clip(betas, a_min=0, a_max=0.999)

#     elif schedule == "sqrt_linear":
#         betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
#     elif schedule == "sqrt":
#         betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
#     else:
#         raise ValueError(f"schedule '{schedule}' unknown.")
#     return betas.numpy()







# class DDPM:
#     # classic DDPM with Gaussian diffusion, in image space
#     def __init__(self,
#                  unet_config=None,
#                  scheduler_config=None,
#                  timesteps=1000, # 1000
#                  beta_schedule="linear", # "linear"
#                  loss_type="l2", # "l2"
#                  monitor="val/loss", #  "val/loss_simple_ema"
#                  use_ema=True, # False
#                  first_stage_key="image", # "jpeg"
#                  image_size=256, # 64
#                  channels=3, # 4
#                  log_every_t=100, # 200
#                  clip_denoised=True, # true
#                  linear_start=1e-4, #  0.00085
#                  linear_end=2e-2, #  0.012
#                  cosine_s=8e-3, #  0.008
#                  given_betas=None, # None
#                  original_elbo_weight=0.0, # 0.0
#                  v_posterior=0.0,  # 0.0 # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
#                  l_simple_weight=1.0, # 1.0
#                  conditioning_key=None, # “crossattn”
#                  parameterization="eps", # eps # all assuming fixed variance schedules # assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
#                  use_positional_encodings=False, # false
#                  learn_logvar=False, # false
#                  logvar_init=0.0, # 0.0
#                  ):
#         super().__init__()
        
#         self.parameterization = parameterization
#         print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
#         self.cond_stage_model = None
#         self.clip_denoised = clip_denoised # True
#         self.log_every_t = log_every_t # 100
#         self.first_stage_key = first_stage_key # jpg
#         self.image_size = image_size  # 64
#         self.channels = channels    # 4
#         self.use_positional_encodings = use_positional_encodings # false
#         self.model = UnetWrapper(unet_config, conditioning_key) # conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

#         self.use_ema = use_ema # false

#         self.use_scheduler = scheduler_config is not None 
#         if self.use_scheduler: # 有运行
#             self.scheduler_config = scheduler_config

#         self.v_posterior = v_posterior # 0.0
#         self.original_elbo_weight = original_elbo_weight # 0.0
#         self.l_simple_weight = l_simple_weight # 1.0

#         if monitor is not None: # "val/loss_simple_ema"
#             self.monitor = monitor # "val/loss_simple_ema"





#         # linear 1000 0.00085 0.012 0.008
#         # if beta_schedule == "linear":
#         betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2 )
#         betas = betas.numpy()
#         alphas = 1.0 - betas
#         alphas_cumprod = np.cumprod(alphas, axis=0)
#         alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

#         timesteps, = betas.shape
#         self.num_timesteps = int(timesteps)
#         self.linear_start = linear_start
#         self.linear_end = linear_end
#         assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        
#         self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32) ) # torch.nn.Module.register_buffer
#         self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32) )
#         self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32) )



#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) + self.v_posterior * betas
#         # above: equal to 1.0 / (1.0/ (1.0- alpha_cumprod_tm1) + alpha_t / beta_t)
#         self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32) )


#         lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * torch.tensor(alphas, dtype=torch.float32)  * (1 - self.alphas_cumprod))

#         # TODO how to choose this term
#         lvlb_weights[0] = lvlb_weights[1]
#         self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
#         assert not torch.isnan(self.lvlb_weights).all()


#         self.loss_type = loss_type # "l2"

#         self.learn_logvar = learn_logvar # False
#         self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))# 0.0, 1000



# class _DeviceDtypeModuleMixin:
    # __jit_unused_properties__: List[str] = ["device", "dtype"]

    # def __init__(self) -> None:
    #     super().__init__()
    #     self._dtype: Union[str, torch.dtype] = torch.get_default_dtype()
    #     self._device = torch.device("cpu")

    # @property
    # def dtype(self) -> Union[str, torch.dtype]:
    #     return self._dtype

    # @dtype.setter
    # def dtype(self, new_dtype: Union[str, torch.dtype]) -> None:
    #     # necessary to avoid infinite recursion
    #     raise RuntimeError("Cannot set the dtype explicitly. Please use module.to(new_dtype).")

    # @property
    # def device(self) -> torch.device:
    #     device = self._device

    #     # make this more explicit to always include the index
    #     if device.type == "cuda" and device.index is None:
    #         return torch.device(f"cuda:{torch.cuda.current_device()}")

    #     return device

    # def to(self, *args: Any, **kwargs: Any) -> Self:
    #     """See :meth:`torch.nn.Module.to`."""
    #     # this converts `str` device to `torch.device`
    #     device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
    #     self.__update_properties(device=device, dtype=dtype)
    #     return super().to(*args, **kwargs)

    # def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
    #     if device is None:
    #         device = torch.device("cuda", torch.cuda.current_device())
    #     elif isinstance(device, int):
    #         device = torch.device("cuda", index=device)
    #     self.__update_properties(device=device)
    #     return super().cuda(device=device)

    # def cpu(self) -> Self:
    #     """See :meth:`torch.nn.Module.cpu`."""
    #     self.__update_properties(device=torch.device("cpu"))
    #     return super().cpu()

    # def type(self, dst_type: Union[str, torch.dtype]) -> Self:
    #     """See :meth:`torch.nn.Module.type`."""
    #     self.__update_properties(dtype=dst_type)
    #     return super().type(dst_type=dst_type)

    # def float(self) -> Self:
    #     """See :meth:`torch.nn.Module.float`."""
    #     self.__update_properties(dtype=torch.float)
    #     return super().float()

    # def double(self) -> Self:
    #     """See :meth:`torch.nn.Module.double`."""
    #     self.__update_properties(dtype=torch.double)
    #     return super().double()

    # def half(self) -> Self:
    #     """See :meth:`torch.nn.Module.half`."""
    #     self.__update_properties(dtype=torch.half)
    #     return super().half()
    # def __update_properties(
    #     self, device: Optional[torch.device] = None, dtype: Optional[Union[str, torch.dtype]] = None
    # ) -> None:
    #     def apply_fn(module: Union[_DeviceDtypeModuleMixin, torch.nn.Module]) -> None:
    #         if not isinstance(module, _DeviceDtypeModuleMixin):
    #             return
    #         if device is not None:
    #             module._device = device
    #         if dtype is not None:
    #             module._dtype = dtype

    #     self.apply(apply_fn)