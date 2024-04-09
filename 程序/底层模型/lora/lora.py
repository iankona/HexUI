import os
import re
import torch


from typing import Union



import safetensors
import 程序.底层模型.shared as shared


def __替换__forward__函数__():
    if not hasattr(torch.nn, 'Linear_forward_before_lora'):
        torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward
    if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

    torch.nn.Linear.forward = __lora__Linear__forward__
    torch.nn.Conv2d.forward = __lora__Conv2d__forward__

def __还原__forward__函数__():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora


def __lora__Linear__forward__(self, input):
    output = torch.nn.Linear_forward_before_lora(self, input)
    return __lora__forward__(self, input, output)


def __lora__Conv2d__forward__(self, input):
    output = torch.nn.Conv2d_forward_before_lora(self, input)
    return __lora__forward__(self, input, output)


def __lora__forward__(module, input, res):
    if len(shared.loras) == 0: return res

    lora_module_name = getattr(module, 'lora_module_name', None) # 'transformer_text_model_encoder_layers_0_self_attn_q_proj'
    for lora in shared.loras:
        module = lora.modules.get(lora_module_name, None)
        if module == None: continue
        # if shared.opts.lora_apply_to_outputs and res.shape == input.shape: # webui, shared.opts.lora_apply_to_outputs == False
        #     res = res + module.up(module.down(res)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)
        # else:
        #     res = res + module.up(module.down(input)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)
        res = res + module.up(module.down(input)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

    return res







# def __还原__forward__函数__():
#     torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
#     torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
#     torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
#     torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
#     torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
#     torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora



# def __替换__forward__函数__():
#     if not hasattr(torch.nn, 'Linear_forward_before_lora'):
#         torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

#     if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
#         torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

#     if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
#         torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

#     if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
#         torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

#     if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
#         torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

#     if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
#         torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

#     torch.nn.Linear.forward = lora_Linear_forward
#     torch.nn.Linear._load_from_state_dict = lora_Linear_load_state_dict
#     torch.nn.Conv2d.forward = lora_Conv2d_forward
#     torch.nn.Conv2d._load_from_state_dict = lora_Conv2d_load_state_dict
#     torch.nn.MultiheadAttention.forward = lora_MultiheadAttention_forward
#     torch.nn.MultiheadAttention._load_from_state_dict = lora_MultiheadAttention_load_state_dict















# def lora_calc_updown(lora, module, target):
#     with torch.no_grad():
#         up = module.up.weight.to(target.device, dtype=target.dtype)
#         down = module.down.weight.to(target.device, dtype=target.dtype)

#         if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
#             updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
#         elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
#             updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
#         else:
#             updown = up @ down

#         updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

#         return updown


# def lora_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
#     weights_backup = getattr(self, "lora_weights_backup", None)

#     if weights_backup is None:
#         return

#     if isinstance(self, torch.nn.MultiheadAttention):
#         self.in_proj_weight.copy_(weights_backup[0])
#         self.out_proj.weight.copy_(weights_backup[1])
#     else:
#         self.weight.copy_(weights_backup)


# def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
#     """
#     Applies the currently selected set of Loras to the weights of torch layer self.
#     If weights already have this particular set of loras applied, does nothing.
#     If not, restores orginal weights from backup and alters weights according to loras.
#     """

#     lora_layer_name = getattr(self, 'lora_layer_name', None)
#     if lora_layer_name is None:
#         return

#     current_names = getattr(self, "lora_current_names", ())
#     wanted_names = tuple((x.name, x.multiplier) for x in shared.loras)

#     weights_backup = getattr(self, "lora_weights_backup", None)
#     if weights_backup is None:
#         if isinstance(self, torch.nn.MultiheadAttention):
#             weights_backup = (self.in_proj_weight.to("cpu", copy=True), self.out_proj.weight.to("cpu", copy=True))
#         else:
#             weights_backup = self.weight.to("cpu", copy=True)

#         self.lora_weights_backup = weights_backup

#     if current_names != wanted_names:
#         lora_restore_weights_from_backup(self)

#         for lora in shared.loras:
#             module = lora.modules.get(lora_layer_name, None)
#             if module is not None and hasattr(self, 'weight'):
#                 self.weight += lora_calc_updown(lora, module, self.weight)
#                 continue

#             module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
#             module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
#             module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
#             module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

#             if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
#                 updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
#                 updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
#                 updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
#                 updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

#                 self.in_proj_weight += updown_qkv
#                 self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
#                 continue

#             if module is None:
#                 continue

#             print(f'failed to calculate lora weights for layer {lora_layer_name}')

#         self.lora_current_names = wanted_names






# def lora_forward(module, input, original_forward):
#     """
#     Old way of applying Lora by executing operations during layer's forward.
#     Stacking many loras this way results in big performance degradation.
#     """
#     # torch.device("cuda")
#     if len(shared.loras) == 0:
#         return original_forward(module, input)

#     input = input.to(dtype=torch.float16)

#     lora_restore_weights_from_backup(module)
#     lora_reset_cached_weight(module)

#     res = original_forward(module, input)

#     lora_layer_name = getattr(module, 'lora_layer_name', None)
#     for lora in shared.loras:
#         module = lora.modules.get(lora_layer_name, None)
#         if module is None:
#             continue

#         module.up.to(device="cuda:0")
#         module.down.to(device="cuda:0")

#         res = res + module.up(module.down(input)) * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

#     return res


# def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
#     self.lora_current_names = ()
#     self.lora_weights_backup = None


# def lora_Linear_forward(self, input):
#     # if shared.opts.lora_functional:
#     #     return lora_forward(self, input, torch.nn.Linear_forward_before_lora)

#     lora_apply_weights(self)

#     return torch.nn.Linear_forward_before_lora(self, input)


# def lora_Linear_load_state_dict(self, *args, **kwargs):
#     lora_reset_cached_weight(self)

#     return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


# def lora_Conv2d_forward(self, input):
#     # if shared.opts.lora_functional:
#     #     return lora_forward(self, input, torch.nn.Conv2d_forward_before_lora)

#     lora_apply_weights(self)

#     return torch.nn.Conv2d_forward_before_lora(self, input)


# def lora_Conv2d_load_state_dict(self, *args, **kwargs):
#     lora_reset_cached_weight(self)

#     return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


# def lora_MultiheadAttention_forward(self, *args, **kwargs):
#     lora_apply_weights(self)
#     return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


# def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
#     lora_reset_cached_weight(self)
#     return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)











class Lora:
    def __init__(self):
        self.path = None
        self.basename = None
        self.hash = None
        self.state_dict = {}
        self.modules = {}
        self.multiplier = 1.0


    def filepath(self, filepath:str):
        self.path = filepath
        self.basename = os.path.basename(filepath)
        self.hash = None
        return self








class LoraModule:
    def __init__(self):
        self.alpha = None
        self.up = None
        self.down = None
        self.multiplier = 1.0








def register_lora_module_mapping_to_latent_diffusion(sd_model):
    # sd_model = shared.latent_model
    lora_module_mapping = {}

    for name, module in sd_model.cond_stage_model.wrapped.named_modules(): # 'transformer.text_model.encoder.layers.0.self_attn.v_proj'
        lora_name = name.replace(".", "_")
        lora_module_mapping[lora_name] = module
        module.lora_module_name = lora_name

    for name, module in sd_model.model.named_modules(): # 'diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k'
        lora_name = name.replace(".", "_")
        lora_module_mapping[lora_name] = module
        module.lora_module_name = lora_name

    sd_model.lora_module_mapping = lora_module_mapping

    # file = open("lora_layout_name.py", mode="w")
    # for name in lora_module_mapping:
    #     file.write(f"\"{name}\"\n")
    # file.close()


def load_loras():
    shared.loras.clear()
    filepaths = [
                #  r"D:\lora\lcm_lora\lcm-lora-sdv1-5.safetensors",
                # [r"D:\lora\GuoFeng3.ckpt", 0.4],
                [r"D:\lora\chunmomoLora_v11.safetensors", 0.3],
                [r"D:\lora\Elegant_hanfu_ruqun_style_v10.safetensors", 0.3],
                [r"D:\lora\Senchan_25_Twitter_v10.safetensors", 0.35],
    ]
    for filepath, multiplier in filepaths:
        # if filepath.endswith(".ckpt"): lora = read_ckpt(filepath)
        if filepath.endswith(".safetensors"): lora = read_lora(filepath)
        if lora is None: continue
        lora.multiplier = multiplier
        shared.loras.append(lora)




def read_ckpt(filepath):
    if filepath == "": return None
    if not os.path.isfile(filepath): return None
    lora = Lora()
    lora.state_dict = torch.load(filepath, map_location="cpu")
    read_lora_modules(lora, shared.latent_model)
    return lora



def read_lora(filepath):
    if filepath == "": return None
    if not os.path.isfile(filepath): return None
    lora = Lora()
    lora.state_dict = safetensors.torch.load_file(filepath, device="cpu")
    read_lora_modules(lora, shared.latent_model)
    return lora








def read_lora_modules(lora, latent_model):
    for name, weight in lora.state_dict.items():
        lora_name, lora_key = name.split(".")[0:2]
        try:
            lora_module_name = get_module_name_from_lora_name(lora_name)
        except:
            continue
        if lora_module_name not in lora.modules: lora.modules[lora_module_name] = LoraModule()
        lora_module = lora.modules[lora_module_name]

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        latent_module = latent_model.lora_module_mapping.get(lora_module_name, None)
        if type(latent_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(latent_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(latent_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(latent_module) == torch.nn.Conv2d and weight.shape[2:] == (1, 1):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif type(latent_module) == torch.nn.Conv2d and weight.shape[2:] == (3, 3):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (3, 3), bias=False)
        else:
            raise ValueError()


        with torch.no_grad(): module.weight.copy_(weight) # copy_ 要求 no_grad
        # module.to(device=torch.device("cuda") , dtype=torch.float32) # Linear(in_features=768, out_features=32, bias=False)
        module.to(device=torch.device("cuda") , dtype=torch.float16) # Linear(in_features=768, out_features=32, bias=False)

        if lora_key == "lora_up":
            lora_module.up = module
        elif lora_key == "lora_down":
            lora_module.down = module
        else:
            raise ValueError()

    # if len(keys_failed_to_match) > 0:
    #     print(f"Failed to match keys when loading Lora : {keys_failed_to_match}")

  



def get_module_name_from_lora_name(name):
    m = re.search(r"_text_model_encoder_layers_(\d+)_(.+)", name) # 'lora_te_text_model_encoder_layers_0_mlp_fc1'
    if m: 
        lora_module_name = f"transformer_{name[8:]}"
    
    m = re.search(r"_down_blocks_(\d+)_attentions_(\d+)_(.+)", name)
    if m: 
        number = 1 + int(m.group(1))*3 + int(m.group(2)) # 12, 34,78
        lora_module_name = f"diffusion_model_input_blocks_{number}_1_{m.group(3)}"

    m = re.search(r"_mid_block_attentions_(\d+)_(.+)", name)
    if m: 
        lora_module_name = f"diffusion_model_middle_block_1_{m.group(2)}"

    m = re.search(r"_up_blocks_(\d+)_attentions_(\d+)_(.+)", name)
    if m:
        number = int(m.group(1)) * 3 + int(m.group(2))     # 3,4,5, 6,7,8, 9,10,11
        lora_module_name = f"diffusion_model_output_blocks_{number}_1_{m.group(3)}"



    m = re.search(r"_down_blocks_(\d+)_resnets_(\d+)_(.+)", name)
    if m:
        du_suffix = m.group(3)
        cv_suffix = {
            'conv1': 'in_layers_2',
            'conv2': 'out_layers_3',
            'time_emb_proj': 'emb_layers_1',
            'conv_shortcut': 'skip_connection'
        }[du_suffix]

        number = 1 + int(m.group(1)) * 3 + int(m.group(2))      # 1,2, 4,5, 7,8
        lora_module_name = f"diffusion_model_input_blocks_{number}_0_{cv_suffix}"


    m = re.search(r"_down_blocks_(\d+)_downsamplers_0_conv", name)
    if m:
        number = 3 + int(m.group(1)) * 3
        lora_module_name = f"diffusion_model_input_blocks_{number}_0_op"
        

    m = re.search(r"_mid_block_resnets_(\d+)_(.+)", name)
    if m:
        du_suffix = m.group(2)
        cv_suffix = {
            'conv1': 'in_layers_2',
            'conv2': 'out_layers_3',
            'time_emb_proj': 'emb_layers_1',
            'conv_shortcut': 'skip_connection'
        }[du_suffix]
        number = int(m.group(1)) * 2
        lora_module_name = f"diffusion_model_middle_block_{number}_{cv_suffix}"
    

    m = re.search(r"_up_blocks_(\d+)_resnets_(\d+)_(.+)", name)
    if m:
        du_suffix = m.group(3)
        cv_suffix = {
            'conv1': 'in_layers_2',
            'conv2': 'out_layers_3',
            'time_emb_proj': 'emb_layers_1',
            'conv_shortcut': 'skip_connection'
        }[du_suffix]

        number = int(m.group(1)) * 3 + int(m.group(2))      # 1,2, 4,5, 7,8
        lora_module_name = f"diffusion_model_output_blocks_{number}_0_{cv_suffix}"
    

    m = re.search(r"_up_blocks_(\d+)_upsamplers_0_conv", name)
    if m:
        block_index = int(m.group(1))
        cv_index = block_index * 3 + 2
        lora_module_name = f"diffusion_model_output_blocks_{cv_index}_{bool(block_index)+1}_conv"
    
    return lora_module_name









# def get_module_name_from_ckpt_name(name):
#     m = re.search(r"transformer.text_model(.+)", name) # 'lora_te_text_model_encoder_layers_0_mlp_fc1'
#     if m: 
#         a = m.group(0)
#         b = m.group(1)
#         lora_module_name = f"transformer.text_model{m.group(0)}"
#         lora_module_name = lora_module_name.replace(".", "_")
    
#     m = re.search(r".diffusion_model.(.+)", name)
#     if m: 
#         a = m.group(0)
#         b = m.group(1)
#         lora_module_name = f"diffusion_model.{m.group(0)}"
#         lora_module_name = lora_module_name.replace(".", "_")

    
    # return lora_module_name



















