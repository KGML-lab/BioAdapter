import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
ip_contrib_maps = {}
ip_contrib_counts = {}
step_ip_contrib_maps = {}
step_ip_contrib_counts = {}

def clear_attn_maps():
    attn_maps.clear()
    ip_contrib_maps.clear()
    ip_contrib_counts.clear()
    clear_step_ip_contrib_maps()


def clear_step_ip_contrib_maps():
    step_ip_contrib_maps.clear()
    step_ip_contrib_counts.clear()


def accumulate_contrib_map(store, counts, name, contrib_map):
    if name in store:
        store[name] = store[name] + contrib_map
        counts[name] += 1
    else:
        store[name] = contrib_map.clone()
        counts[name] = 1


def normalize_map(attn_map):
    attn_map = attn_map.to(dtype=torch.float32)
    attn_min = attn_map.min()
    attn_max = attn_map.max()
    if attn_max <= attn_min:
        return torch.zeros_like(attn_map, dtype=torch.float32)
    return (attn_map - attn_min) / (attn_max - attn_min)

def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map
        if hasattr(module.processor, "ip_contrib_map"):
            contrib_map = module.processor.ip_contrib_map.detach()
            accumulate_contrib_map(ip_contrib_maps, ip_contrib_counts, name, contrib_map)
            accumulate_contrib_map(step_ip_contrib_maps, step_ip_contrib_counts, name, contrib_map)
            del module.processor.ip_contrib_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map
def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps


def get_net_attn_maps_per_sample(image_size, num_samples, detach=True):
    net_attn_maps = []

    for _, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map

        if attn_map.shape[0] == 2 * num_samples:
            attn_map = attn_map[num_samples:]
        elif attn_map.shape[0] != num_samples:
            raise ValueError(
                f"Expected attention batch dimension {num_samples} or {2 * num_samples}, got {attn_map.shape[0]}"
            )

        sample_maps = []
        for sample_idx in range(num_samples):
            sample_maps.append(upscale(attn_map[sample_idx], image_size))
        net_attn_maps.append(torch.stack(sample_maps, dim=0))

    if not net_attn_maps:
        raise ValueError("No attention maps were captured. Did you call register_cross_attention_hook?")

    return torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)


def upscale_spatial_map(spatial_map, target_size, normalize=True):
    spatial_map = spatial_map.to(dtype=torch.float32)
    temp_size = None

    for i in range(0, 5):
        scale = 2 ** i
        if (target_size[0] // scale) * (target_size[1] // scale) == spatial_map.shape[0] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    spatial_map = spatial_map.view(*temp_size)
    spatial_map = F.interpolate(
        spatial_map.unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )[0, 0]

    if normalize:
        return normalize_map(spatial_map)
    return spatial_map


def build_ip_contrib_maps_per_sample(contrib_maps, contrib_counts, image_size, num_samples, detach=True):
    net_maps = []

    for name, contrib_map in contrib_maps.items():
        contrib_map = contrib_map.cpu() if detach else contrib_map
        contrib_map = contrib_map / max(contrib_counts.get(name, 1), 1)

        if contrib_map.shape[0] == 2 * num_samples:
            contrib_map = contrib_map[num_samples:]
        elif contrib_map.shape[0] != num_samples:
            raise ValueError(
                f"Expected contribution batch dimension {num_samples} or {2 * num_samples}, got {contrib_map.shape[0]}"
            )

        sample_maps = []
        for sample_idx in range(num_samples):
            sample_maps.append(upscale_spatial_map(contrib_map[sample_idx], image_size, normalize=False))
        net_maps.append(torch.stack(sample_maps, dim=0))

    if not net_maps:
        raise ValueError("No IP contribution maps were captured. Did you call register_cross_attention_hook?")

    mean_map = torch.mean(torch.stack(net_maps, dim=0), dim=0)
    normalized_maps = [normalize_map(mean_map[sample_idx]) for sample_idx in range(num_samples)]
    return torch.stack(normalized_maps, dim=0)


def get_net_ip_contrib_maps_per_sample(image_size, num_samples, detach=True):
    return build_ip_contrib_maps_per_sample(
        ip_contrib_maps,
        ip_contrib_counts,
        image_size=image_size,
        num_samples=num_samples,
        detach=detach,
    )


def get_step_ip_contrib_maps_per_sample(image_size, num_samples, detach=True):
    return build_ip_contrib_maps_per_sample(
        step_ip_contrib_maps,
        step_ip_contrib_counts,
        image_size=image_size,
        num_samples=num_samples,
        detach=detach,
    )

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
