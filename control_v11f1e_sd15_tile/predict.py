import os
import shutil
from typing import Iterator

import torch
from cog import BasePredictor, Input, Path
from compel import Compel
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import numpy as np
import settings
from PIL import Image

class KerrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KerrasDPM": KerrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        print("Loading pipelines...")

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

        print("Loading controlnet...")
        controlnet = ControlNetModel.from_pretrained(
            settings.MODEL_CACHE + '/controlnet',
            torch_dtype=torch.float16,
            local_files_only=True,
        )

        self.cnet_pipe = StableDiffusionControlNetPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        ).to("cuda")

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

    def resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img


    @torch.inference_mode()
    def predict(
        self,
        control_image: Path = Input(
            description="Optional Image to use for guidance based on tile",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale of conditioning for controlnet",
            default=1.0,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        strength: float = Input(
            description="Strength", ge=0.0, le=1.0, default=1.0
        ),
        resolution: int = Input(
            description="Resolution of output image",
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
        scheduler: str = Input(
            default="KerrasDPM",
            choices=list(SCHEDULERS.keys()),
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

        if control_image:
            print("Using ControlNet tile")
            
            control_image = self.load_image(control_image)
            processed_control_image = self.resize_for_condition_image(control_image, resolution)

            width = processed_control_image.size[0]
            height = processed_control_image.size[1]

            pipe = self.cnet_pipe
            extra_kwargs = {
                "image": processed_control_image,
                # "controlnet_conditioning_image": processed_control_image,
                "width": width,
                "height": height,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
            }
            
        else:
            print("Using txt2img pipeline")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": resolution,
                "height": resolution,
            }

        pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

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
