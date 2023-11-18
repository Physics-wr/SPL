# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from functools import partial

from fastreid.layers import *
from fastreid.utils import comm
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from .sam.transformer import *
from .sam.mask_decoder import MaskDecoder
from .sam.image_encoder import ImageEncoderViT
from .sam.prompt_encoder import PromptEncoder
from .utils.utils import rel_pos_embed_downscale, pos_embed_downscale
from .sam.utils import *
import random

from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        cfg,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.train_set_num=len(cfg.DATASETS.NAMES)
        self.cfg=cfg
        self.num_blocks=cfg.MODEL.LOSSES.HINT.NUM_BLOCKS

        self.bottleneck=nn.ModuleList()
        self.classifier=nn.ModuleList()
        for i in range(len(self.image_encoder.blocks)):
            bottleneck = nn.BatchNorm1d(768) 
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
            self.bottleneck.append(bottleneck)
            classifier = nn.ModuleList([nn.Linear(768, i, bias=False) for i in cfg.DATASETS.NUM_PIDS])
            for clsf in classifier:
                clsf.apply(weights_init_classifier)
            self.classifier.append(classifier)    

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        data_name:int,
        masks: torch.Tensor = None,
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings, list_out = self.image_encoder(batched_input,masks)
        encoder_feat=list_out[-1][...,None,None]

        if self.training:
            global_feat=list_out[-1]
            feat=self.bottleneck[-1](global_feat)
            cls_scores=[]
            for i in range(len(self.classifier)):
                feat=self.bottleneck[i](list_out[i])
                cls_scores=self.classifier[i][data_name](feat)
            output={'feat':global_feat,'score':cls_scores[-1]}
            random_block_idx=random.sample(range(len(self.classifier)-1),self.num_blocks)
            output_rbs=[]
            for n in random_block_idx:
                output_rbs.append({'feat':list_out[n],'score':cls_scores[n]})               
    
        if self.cfg.MODEL.BACKBONE.POINT_PROMPT:
            points_coords = torch.as_tensor([self.cfg.INPUT.SIZE_TRAIN[1]//2,self.cfg.INPUT.SIZE_TRAIN[0]//2], dtype=torch.float32,device=image_embeddings.device)[None, None, :] #B*N*2
            labels_coords = torch.as_tensor([1], dtype=torch.int,device=image_embeddings.device)[None, :] #B*N
            points = (torch.repeat_interleave(points_coords,batched_input.size(0),0), torch.repeat_interleave(labels_coords,batched_input.size(0),0))
        else:
            points = None   

        if self.cfg.MODEL.BACKBONE.BOX_PROMPT:
            boxes = torch.as_tensor([[0, 0, self.cfg.INPUT.SIZE_TRAIN[1]//2,self.cfg.INPUT.SIZE_TRAIN[0]//2]], dtype=torch.float32,device=image_embeddings.device)[None, :, :] #B*N*4
            boxes = torch.repeat_interleave(boxes,batched_input.size(0),0)
        else:
            boxes = None
            
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None,
        )

        cls_token = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
        )
        
        if self.training:
            return cls_token, encoder_feat, output, output_rbs
        else:
            return cls_token, encoder_feat, None, None
    

@BACKBONE_REGISTRY.register()
def build_sam_reid(cfg):
    """
    Create a ResNeXt instance from config.
    Returns:
        ResNeXt: a :class:`ResNeXt` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    img_size      = cfg.INPUT.SIZE_TRAIN
    layer_filter  = cfg.MODEL.BACKBONE.LAYER_FILTER,
    weight_lambda = cfg.MODEL.BACKBONE.WEIGHT_LAMBDA,
    # fmt: on

    num_blocks_per_stage = {
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
        '152x': [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 3, 0]}[depth]
    
    sam_model = Sam(
        cfg,
        image_encoder=ImageEncoderViT(
            layer_filter=layer_filter,
            weight_lambda=weight_lambda,
            depth=12,
            embed_dim=768,
            img_size=img_size, #(384, 192),
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=(16, 16),
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        ),
        prompt_encoder = PromptEncoder(
                embed_dim=256,
                image_embedding_size=(img_size[0]//16, img_size[1]//16),
                input_image_size=img_size,
                mask_in_chans=16,
                activation=nn.GELU,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )
    
    
    
    ## Load pretrained model
    if pretrain:
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key
        
        ## rescale the position embedding    
        state_dict['image_encoder.pos_embed'] = pos_embed_downscale(
            state_dict['image_encoder.pos_embed'],
            output_size=(cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16),
            )
        
        ## rescale the relative position embedding in global attention layers
        for i in [2, 5, 8, 11]:
            new_pos_h = rel_pos_embed_downscale(state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_h'], (47, 64))
            new_pos_w = rel_pos_embed_downscale(state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_w'], (23, 64))
            state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_h'] = new_pos_h
            state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_w'] = new_pos_w

        incompatible = sam_model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return sam_model

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)










        # ## Domain prompt
        # if is_test: #is adapter
        #     prompt=None
        #     x, _, cls_token = self.mask_decoder(
        #             # image_embeddings=curr_embedding.unsqueeze(0),
        #             image_embeddings=image_embeddings,
        #             image_pe=self.prompt_encoder.get_dense_pe(),
        #             sparse_prompt_embeddings=sparse_embeddings,
        #             dense_prompt_embeddings=dense_embeddings,
        #             domain_prompt=prompt,
        #             cls_token_input=None,
        #             multimask_output=multimask_output,
        #     )
        #     prompt_weight = self.adaptor(cls_token)# weighted sum
        #     for j in range(self.train_set_num):
        #         current_prompt=self.prompt[j]*prompt_weight[0][j]
        #         if prompt==None:
        #             prompt=current_prompt
        #         else:
        #             prompt+=current_prompt
        # else:
        #     prompt = []
        #     for data_name in data_names:
        #         prompt.append(self.prompt[data_name].unsqueeze(0))
        #     prompt = torch.cat(prompt, dim=0) # TODO: check if this is correct
        #     cls_token = None