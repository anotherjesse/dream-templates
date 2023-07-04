#!/usr/bin/env python

import os
import shutil
import torch

from diffusers import ControlNetModel
import settings

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

TMP_CACHE = './tmp'

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

c = ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE)

c.save_pretrained(settings.MODEL_CACHE + '/controlnet')

shutil.rmtree(TMP_CACHE)
