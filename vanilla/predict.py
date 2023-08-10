import os
import shutil
from typing import Iterator

import torch
from cog import BasePredictor, Input, Path
from compel import Compel
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
from PIL import Image
import numpy as np


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        if not os.path.exists(settings.BASE_MODEL_PATH):
            self.real = False
            return

        print("Loading txt2img...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            settings.BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            cache_dir=settings.MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        
        self.safety_checker = self.txt2img_pipe.safety_checker

        print("Loading img2img...")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        )

        print("Loading inpaint...")
        self.inpainting_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        )

        print("Loading compel...")
        self.compel = Compel(
            tokenizer=self.txt2img_pipe.tokenizer,
            text_encoder=self.txt2img_pipe.text_encoder,
        )

        self.real = True

    def load_image(self, image_path: Path):
        if image_path is None:
            return None
        # not sure why I have to copy the image, but it fails otherwise
        # seems like a bug in cog
        if os.path.exists("img.png"):
            os.unlink("img.png")
        shutil.copy(image_path, "img.png")
        return load_image("img.png")

    @torch.inference_mode()
    def predict(
        self,
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
            le=10,
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
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""

        if not self.real:
            raise RuntimeError("This is a template, not a real model - add weights")

        if image:
            image = self.load_image(image)
        if mask:
            mask = self.load_image(mask)

        if image and mask:
            print("Using inpaint pipeline")
            pipe = self.inpainting_pipe
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
            pipe = self.img2img_pipe
            extra_kwargs = {
                "image": image,
                "strength": prompt_strength,
            }
        else:
            print("Using txt2img pipeline")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        if prompt:
            print("parsed prompt:", self.compel.parse_prompt_string(prompt))
            prompt_embeds = self.compel(prompt)
        else:
            prompt_embeds = None

        if negative_prompt:
            print(
                "parsed negative prompt:",
                self.compel.parse_prompt_string(negative_prompt),
            )
            negative_prompt_embeds = self.compel(negative_prompt)
        else:
            negative_prompt_embeds = None

        if disable_safety_check:
            pipe.safety_checker = None
        else:
            pipe.safety_checker = self.safety_checker

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
