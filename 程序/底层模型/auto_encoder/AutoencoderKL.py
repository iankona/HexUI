import torch

from 程序.底层模型.auto_encoder.model_block import Encoder, Decoder
from 程序.底层模型.auto_encoder.distributions import DiagonalGaussianDistribution
from util import instantiate_from_config



class AutoencoderKL(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor


    def init_from_ckpt(self, path="", ignore_keys=list()):
        ckpt_path = r"D:\Program_python\stable-diffusion-sample\lora\vae-ft-mse-840000-ema-pruned-laion.ckpt"
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior




if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load(r"../autoencoder/autoencoder.yaml")
    # print(config)
    # ddconfig = config["ddconfig"]
    # lossconfig = config["lossconfig"]

    autoencoder = instantiate_from_config(config)
    autoencoder.eval()
    x = torch.randn(1, 3, 256, 256)
    dec, posterior = autoencoder(x)
    print(dec.shape)
    print(posterior.mode().shape)
    print(autoencoder)