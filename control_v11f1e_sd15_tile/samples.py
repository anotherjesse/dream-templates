import base64
import requests
import sys


def gen(output_fn, **kwargs):
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.normal.png",
        prompt="best quality",
        negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
        control_image="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png",
        resolution=512,
        seed=42,
    )
    gen(
        "sample.lesser.png",
        prompt="best quality",
        negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
        control_image="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png",
        controlnet_conditioning_scale=0.35,
        resolution=512,
        seed=42,
    )
    gen(
        "sample.bigger.png",
        prompt="best quality",
        negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
        control_image="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/images/original.png",
        resolution=1024,
        seed=42,
    )
    gen(
        "sample.txt2img.png",
        prompt="a handsome man with ray-ban sunglasses",
        seed=42,
    )


if __name__ == "__main__":
    main()
