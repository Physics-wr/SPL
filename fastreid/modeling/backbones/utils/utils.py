# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

def pos_embed_downscale(
    pos_embed: torch.Tensor, 
    output_size: tuple=(384, 192),
    ) -> torch.Tensor:
    '''
    Downscale the pos_embed to the output size
    '''
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed, 
        size=output_size, 
        # mode="bicubic",
        mode="bilinear",
        align_corners=False,
        )
    pos_embed = pos_embed.permute(0, 2, 3, 1)
    return pos_embed


def rel_pos_embed_downscale(
    pos_embed: torch.Tensor, 
    output_size: tuple,
    ) -> torch.Tensor:
    '''
    Downscale the pos_embed to the output size
    '''
    pos_embed = torch.nn.functional.interpolate(
        pos_embed.unsqueeze(0).unsqueeze(0),
        size=output_size, 
        # mode="bicubic",
        mode="bilinear",
        align_corners=False,
        )
    return pos_embed.squeeze(0).squeeze(0)