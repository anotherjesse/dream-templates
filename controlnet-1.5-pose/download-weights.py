#!/usr/bin/env python

import os
import shutil

import settings

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

import torch
from diffusers import ControlNetModel
from huggingface_hub import hf_hub_download


# https://github.com/patrickvonplaten/controlnet_aux/pull/4
hf_hub_download(
    "lllyasviel/ControlNet",
    "annotator/ckpts/body_pose_model.pth",
    cache_dir=settings.MODEL_CACHE,
)

ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    cache_dir=settings.MODEL_CACHE,
)
