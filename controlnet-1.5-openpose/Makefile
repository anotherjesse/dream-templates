all: pose

download:
	sudo rm -rf diffusers-cache
	cog run python download-weights.py

push:
	if ! test -d diffusers-cache; then \
		echo "Controlnet weights do not exist, make download"; \
		exit 1; \
	fi
	cog push r8.im/anotherjesse/controlnet-1.5-pose-template

samples:
	if ! test -d weights; then \
		echo "Directory weights does not exist"; \
		exit 1; \
	fi
	cog predict -i prompt="portrait of cjw by van gogh" \
		-i control_image=@../images/human_512x512.png \
		-i seed=42
	sudo mv output.0.png output.controlnet_txt2img.png
	cog predict -i prompt="portrait of cjw by van gogh" \
		-i seed=42
	sudo mv output.0.png output.txt2img.png
	cog predict -i prompt="portrait of cjw by van gogh" \
		-i control_image=@../images/human_512x512.png \
		-i image=@../images/human_512x512.png \
		-i seed=42
	sudo mv output.0.png output.controlnet_img2img.png
	cog predict -i prompt="portrait of cjw by van gogh" \
		-i image=@../images/human_512x512.png \
		-i seed=42
	sudo mv output.0.png output.img2img.png
	cog predict -i prompt="portrait of cjw by van gogh" \
		-i image=@../images/dog.png \
		-i mask=@../images/mask.png \
		-i seed=42
	sudo mv output.0.png output.inpainting.png

