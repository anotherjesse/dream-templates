import shutil
import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils import load_image

import settings

from controlnet_aux import OpenposeDetector
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...")

        if not os.path.exists(settings.BASE_MODEL_PATH):
            self.real = False
            return

        print("Loading pose...")
        self.openpose = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet",
            cache_dir=settings.MODEL_CACHE,
        )

        print("Loading txt2img...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            settings.BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            cache_dir=settings.MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

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

        print("Loading controlnet...")
        controlnet = ControlNetModel.from_pretrained(
            settings.CONTROLNET_MODEL,
            torch_dtype=torch.float16,
            cache_dir=settings.MODEL_CACHE,
            local_files_only=True,
        )

        print("Loading controlnet txt2img...")
        self.cnet_txt2img_pipe = StableDiffusionControlNetPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        ).to("cuda")

        print("Loading controlnet img2img...")
        self.cnet_img2img_pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
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

    def process_control(self, control_image):
        if control_image is None:
            return None

        return self.openpose(control_image)

    @torch.inference_mode()
    def predict(
        self,
        control_image: Path = Input(
            description="Optional Image to use for guidance based on posenet",
            default=None,
        ),
        image: Path = Input(
            description="Optional Image to use for img2img guidance", default=None
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
            le=4,
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
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if not self.real:
            raise RuntimeError("This is a template, not a real model - add weights")

        if image:
            image = self.load_image(image)
        if control_image:
            control_image = self.load_image(control_image)
            control_image = self.process_control(control_image)

        if control_image and image:
            print("Using ControlNet img2img")
            pipe = self.cnet_img2img_pipe
            extra_kwargs = {
                "controlnet_conditioning_image": control_image,
                "image": image,
                "strength": prompt_strength,
            }
        elif control_image:
            print("Using ControlNet txt2img")
            pipe = self.cnet_txt2img_pipe
            extra_kwargs = {
                "image": control_image,
                "width": width,
                "height": height,
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

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
