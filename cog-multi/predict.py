import os
import shutil
import subprocess
from typing import Iterator
import time


import torch
from cog import BasePredictor, Input, Path
import io
from compel import Compel
import hashlib
import requests
import tarfile
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

import settings

from functools import lru_cache


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    @lru_cache(maxsize=10)
    def get_weights(self, weights: str):
        destination_path = self.weights_path(weights)
        if not os.path.exists(destination_path):
            self.download_weights(weights)

        return self.load_weights(destination_path)

    def load_weights(self, path):
        print(f"Loading txt2img... from {path}")
        return StableDiffusionPipeline.from_pretrained(
            path,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    def weights_path(self, weights: str):
        if not os.path.exists("/src/weights"):
            os.makedirs("/src/weights")

        hashed_url = hashlib.sha256(weights.encode()).hexdigest()
        short_hash = hashed_url[:16]
        return os.path.join("/src/weights", short_hash)

    def download_weights(self, weights: str):
        print(f"Downloading weights for {weights}...")
        
        dest = self.weights_path(weights)

        weights = weights.replace('https://replicate.delivery/pbxt/', 'https://storage.googleapis.com/replicate-files/')
        print("using url", weights)
        cmd = ['/src/pget', '-x', weights, dest]
        print(" ".join(cmd))
        start = time.time()
        output = subprocess.check_output(cmd)
        print("downloaded in", time.time() - start)

    def load_image(self, image_path: Path):
        if image_path is None:
            return None
        # not sure why I have to copy the image, but it fails otherwise
        # seems like a bug in cog
        if os.path.exists("img.png"):
            os.unlink("img.png")
        shutil.copy(image_path, "img.png")
        return load_image("img.png")

    def get_pipeline(self, pipe, kind):
        if kind == "txt2img":
            return pipe

        if kind == "img2img":
            return StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )

        if kind == "inpaint":
            return StableDiffusionInpaintPipelineLegacy(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )

    @torch.inference_mode()
    def predict(
        self,
        weights: str = Input(
            description="URL with stablediffusion weights tar to use",
        ),
        image: Path = Input(
            description="Optional Image to use for img2img guidance", default=None
        ),
        mask: Path = Input(
            description="Optional Mask to use for legacy inpainting", default=None
        ),
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=32,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
            ],
            description="Choose a scheduler.",
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        info: bool = Input(
            description="log extra information about the run", default=False
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""

        if info:
            os.system("nvidia-smi")
            os.system("df -h")
            os.system("free -h")
            print(self.get_weights.cache_info())

        start = time.time()
        pipe = self.get_weights(weights)
        print("loading weights took: %0.2f" % (time.time() - start))

        start = time.time()
        if image:
            image = self.load_image(image)
        if mask:
            mask = self.load_image(mask)
        print("loading images took: %0.2f" % (time.time() - start))

        start = time.time()
        if image and mask:
            print("Using inpaint pipeline")
            pipe = self.get_pipeline(pipe, "inpaint")
            # FIXME(ja): prompt/negative_prompt are sent to the inpainting pipeline
            # because it doesn't support prompt_embeds/negative_prompt_embeds
            extra_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "mask_image": mask,
                "strength": prompt_strength,
            }
        elif image:
            print("Using img2img pipeline")
            pipe = self.get_pipeline(pipe, "img2img")
            extra_kwargs = {
                "image": image,
                "strength": prompt_strength,
            }
        else:
            print("Using txt2img pipeline")
            pipe = self.get_pipeline(pipe, "txt2img")
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        print("loading pipeline took: %0.2f" % (time.time() - start))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        # FIXME(ja): we shouldn't need to do this multiple times
        # or perhaps we create the object each time?
        print("Loading compel...")
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
        )

        if prompt:
            print("parsed prompt:", compel.parse_prompt_string(prompt))
            prompt_embeds = compel(prompt)
        else:
            prompt_embeds = None

        if negative_prompt:
            print(
                "parsed negative prompt:",
                compel.parse_prompt_string(negative_prompt),
            )
            negative_prompt_embeds = compel(negative_prompt)
        else:
            negative_prompt_embeds = None

        # if disable_safety_check:
        #     pipe.safety_checker = None
        # else:
        #     pipe.safety_checker = self.safety_checker

        result_count = 0
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)
            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue

            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            yield Path(output_path)
            result_count += 1

        if result_count == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )


def make_scheduler(name, config):
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "HeunDiscrete": HeunDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "UniPCMultistep": UniPCMultistepScheduler.from_config(config),
    }[name]
