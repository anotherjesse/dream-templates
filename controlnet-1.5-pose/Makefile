all: pose


pose:
	sudo rm -rf diffusers-cache
	cog run python download-weights.py
	cog push r8.im/anotherjesse/controlnet-1.5-pose-template