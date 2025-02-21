import torch
import requests
from tqdm import tqdm
import os
from collections import OrderedDict

def download_lpips_linear_weights(save_path='lpips_linear_weights.pth'):
    """Download the LPIPS linear weights."""
    if os.path.exists(save_path):
        print(f"Linear weights already exist at {save_path}")
        return save_path
    
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth'
    
    print(f"Downloading LPIPS linear weights...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    return save_path

def prepare_lpips_weights(output_path='lpips_complete_weights.pth'):
    """Prepare complete LPIPS weights including VGG backbone and linear layers."""
    # First, get a pretrained VGG model
    import torchvision.models as models
    vgg = models.vgg16(pretrained=True)
    vgg_state_dict = vgg.state_dict()
    
    # Download linear weights
    linear_weights_path = download_lpips_linear_weights()
    linear_state_dict = torch.load(linear_weights_path)
    
    # Create the complete state dict
    complete_state_dict = OrderedDict()
    
    # Add scaling layer weights (these are fixed values from the original implementation)
    complete_state_dict['scaling_layer.shift'] = torch.tensor([-.030, -.088, -.188])[None, :, None, None]
    complete_state_dict['scaling_layer.scale'] = torch.tensor([.458, .448, .450])[None, :, None, None]
    
    # Add VGG backbone weights
    for key, value in vgg_state_dict.items():
        if 'features' in key:
            # Convert VGG features keys to LPIPS net.slice format
            layer_num = int(key.split('.')[1])
            if layer_num <= 3:
                new_key = f'net.slice1.{layer_num}.{key.split(".")[-1]}'
            elif layer_num <= 8:
                new_key = f'net.slice2.{layer_num}.{key.split(".")[-1]}'
            elif layer_num <= 15:
                new_key = f'net.slice3.{layer_num}.{key.split(".")[-1]}'
            elif layer_num <= 22:
                new_key = f'net.slice4.{layer_num}.{key.split(".")[-1]}'
            elif layer_num <= 29:
                new_key = f'net.slice5.{layer_num}.{key.split(".")[-1]}'
            else:
                continue
            complete_state_dict[new_key] = value
    
    # Add linear layer weights
    for key, value in linear_state_dict.items():
        complete_state_dict[key] = value
    
    # Save the complete weights
    torch.save(complete_state_dict, output_path)
    print(f"Saved complete LPIPS weights to {output_path}")
    return output_path

if __name__ == "__main__":
    weights_path = prepare_lpips_weights()
    print(f"LPIPS weights ready at: {weights_path}")
    
    # Verify the weights contain all necessary keys
    state_dict = torch.load(weights_path)
    print("\nVerifying state dict keys...")
    expected_keys = [
        "scaling_layer.shift", "scaling_layer.scale",
        "net.slice1.0.weight", "net.slice1.0.bias",
        "lin0.model.1.weight", "lin1.model.1.weight",
        "lin2.model.1.weight", "lin3.model.1.weight",
        "lin4.model.1.weight"
    ]
    
    for key in expected_keys:
        print(f"Found {key}: {'✓' if key in state_dict else '✗'}")