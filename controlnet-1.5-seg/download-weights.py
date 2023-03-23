#!/usr/bin/env python

import os
import shutil

import settings

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

import torch
from diffusers import ControlNetModel
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small", cache_dir=settings.MODEL_CACHE)
UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small", cache_dir=settings.MODEL_CACHE)

ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    cache_dir=settings.MODEL_CACHE,
)
