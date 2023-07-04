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
        "sample.mask.png",
        prompt="a handsome man with ray-ban sunglasses",
        mask="https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png",
        control_image="https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png",
        seed=42,
    )
    gen(
        "sample.txt2img.png",
        prompt="a handsome man with ray-ban sunglasses",
        seed=42,
    )


if __name__ == "__main__":
    main()
