all: canny


scibble: 
	sudo rm -rf diffusers-cache
	cp settings-scribble.py settings.py
	cog run python download-weights.py
	cog push r8.im/anotherjesse/controlnet-1.5-scribble-template

canny:
	sudo rm -rf diffusers-cache
	cp settings-canny.py settings.py
	cog run python download-weights.py
	cog push r8.im/anotherjesse/controlnet-1.5-canny-template