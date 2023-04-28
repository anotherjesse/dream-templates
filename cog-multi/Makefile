all: samples

download:
	sudo rm -rf diffusers-cache
	cog run python download-weights.py

push:
	if ! test -d diffusers-cache; then \
		echo "Controlnet weights do not exist, make download"; \
		exit 1; \
	fi
	cog push r8.im/anotherjesse/controlnet-1.5-depth-template

serve:
	cog run -p 5000 python -m cog.server.http

samples:
	python samples.py
