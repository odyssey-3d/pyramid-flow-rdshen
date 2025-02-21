import os
import torch
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
from einops import rearrange

import json
import jsonlines
from tqdm import tqdm
from concurrent import futures
from torch.utils.data import DataLoader, DistributedSampler

from trainer_misc import init_distributed_mode
from pyramid_dit import (
    SD3TextEncoderWithMask,
    FluxTextEncoderWithMask,
)


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--anno_file', type=str, default='', help="The video annotation file")
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--model_name', default='pyramid_flux', type=str, help="The Model Architecture Name", choices=["pyramid_flux", "pyramid_mmdit"])
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--jsonl_out_file', type=str, default='', help="The jsonl file to save the output")
    return parser.parse_args()


class VideoTextDataset(Dataset):
    def __init__(self, anno_file):
        super().__init__()

        self.annotation = []
        with jsonlines.open(anno_file, 'r') as reader:
            for item in tqdm(reader):
                self.annotation.append(item)   # The item is a dict that has key_name: text, text_fea

    def __getitem__(self, index):
        try:
            anno = self.annotation[index]
            video_path = anno['video']
            latent_path = anno['latent']
            text = anno['text']
            text_fea_path = video_path.replace(f'.mp4', '-text.pt')
            if 'text_fea' in anno:
                text_fea_path = anno['text_fea']    # The text feature saving path
            text_fea_save_dir = os.path.split(text_fea_path)[0]
            if not os.path.exists(text_fea_save_dir):
                os.makedirs(text_fea_save_dir, exist_ok=True)
            return text, text_fea_path, video_path, latent_path
        except Exception as e:
            print(f'Error with {e}')
            return None, None
    
    def __len__(self):
        return len(self.annotation)


def build_data_loader(args):

    def collate_fn(batch):
        text_list = []
        output_path_list = []
        video_path_list = []
        latent_path_list = []
        for text, text_fea_path, video_path, latent_path in batch:
            if text is not None:
                text_list.append(text)
                output_path_list.append(text_fea_path)
                video_path_list.append(video_path)
                latent_path_list.append(latent_path)
                
        return {'text': text_list, 'output': output_path_list, 'video': video_path_list, 'latent': latent_path_list}

    dataset = VideoTextDataset(args.anno_file)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, 
        sampler=sampler, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return loader


def build_model(args):
    model_dtype = args.model_dtype
    model_name = args.model_name
    model_path = args.model_path

    if model_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif model_dtype == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if model_name == "pyramid_flux":
        text_encoder = FluxTextEncoderWithMask(model_path, torch_dtype=torch_dtype)
    elif model_name == "pyramid_mmdit":
        text_encoder = SD3TextEncoderWithMask(model_path, torch_dtype=torch_dtype)
    else:
        raise NotImplementedError

    return text_encoder


def save_output(prompt_embed, prompt_attention_mask, pooled_prompt_embed, output_path):
    try:
        output_dict = {
            'prompt_embed': prompt_embed.unsqueeze(0).cpu().clone(),
            'prompt_attention_mask': prompt_attention_mask.unsqueeze(0).cpu().clone(),
            'pooled_prompt_embed': pooled_prompt_embed.unsqueeze(0).cpu().clone(),
        }
        torch.save(output_dict, output_path)
    except Exception as e:
        pass


def main():
    args = get_args()
    
    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank

    model = build_model(args)
    model.to(device)

    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16 
    elif args.model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    data_loader = build_data_loader(args)
    torch.distributed.barrier()

    task_queue = []
    processed_videos = []
    
    with futures.ThreadPoolExecutor(max_workers=16) as executor:

        for sample in tqdm(data_loader):
            texts = sample['text']
            outputs = sample['output']
            video_paths = sample['video']
            latent_paths = sample['latent']
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                prompt_embeds, prompt_attention_masks, pooled_prompt_embeds = model(texts, device)

                for video_path, output_path, latent_path, prompt_embed, prompt_attention_mask, pooled_prompt_embed, text in \
                    zip(video_paths, outputs, latent_paths, prompt_embeds, prompt_attention_masks, pooled_prompt_embeds, texts):
                    
                    task_queue.append(
                        executor.submit(
                            save_output, prompt_embed, prompt_attention_mask, pooled_prompt_embed, output_path
                        )
                    )
                    processed_videos.append({'video': video_path, 'text': text, 'latent': latent_path, 'text_fea': output_path})
                    
        for future in futures.as_completed(task_queue):
            res = future.result()

    torch.distributed.barrier()
    
    # Only let the main process write the output file
    if args.rank == 0:
        with jsonlines.open(args.jsonl_out_file, 'w') as writer:
            for video in processed_videos:
                writer.write(video)

    torch.distributed.barrier()


if __name__ == '__main__':
    main()