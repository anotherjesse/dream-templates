# Templates?

This repository is meant to build a fully functional Cog models for replicate that are missing their Stable Diffusion weights.

The weights are then added during the Build step of part of the Dreambooth API.

## Inside the Replicate Dreambooth API

In [Train and deploy a DreamBooth model on Replicate](https://replicate.ai/blog/dreambooth/), we learn how to train a model using the Replicate Dreambooth API.

This is an experimental API, so it is currently documented only in the blog post.  Since the blog post was written, we've added a few new features, so let's document them here.

### What does the API do?

Replicate added an experimental endpoint https://dreambooth-api-experimental.replicate.com/v1/trainings that performs the following steps:

1. Train: Runs a training job on the Replicate platform as a normal "[prediction](https://replicate.com/docs/reference/http#predictions.get)"
2. Build: When the training job is complete, it downloads the resulting weights and builds a [Cog](https://github.com/replicate/cog) model.
3. Push: Pushes the Cog model to the Replicate registry, which enables you to run it on the Replicate platform.
4. Webhook: Sends a webhook when your model is ready for use.

### What are the inputs to the API?

Let's take a deep dive into the JSON payload and add a couple of new fields not documented in the blog post: `notes` and `template_version`. 

```json
{
  "input": {
    "instance_prompt": "a photo of a cjw person",
    "class_prompt": "a photo of a person",
    "instance_data": "https://example.com/person.zip",
    "max_train_steps": 2000
  },
  "trainer_version": "cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa",
  "template_version": "0f5cfc3e2a0e86dbd141057501ba5196c7dbea94c45dab4894e6ff7d6a2cc324",
  "model": "yourusername/yourmodel",
  "notes": "notes about this dreambooth training",
  "webhook_completed": "https://example.com/dreambooth-webhook"
}
```

Let's break down the steps used in the API one by one and look at the JSON payload used for each step.

#### Step 1: Train

The fields used for training are: `input` and `trainer_version`.

The `input` are the fields needed by the trainer you specify.  In the blog post, we used the `instance_prompt` and `class_prompt` fields, but you can use any fields defined by the trainer.

Looking at the [replicate/dreambooth versions](https://replicate.com/replicate/dreambooth/versions), here are links to API docs for fields used by a given trainer:

- [9c41656f8ae2e3d2af4c1b46913d7467cd891f2c1c5f3d97f1142e876e63ed7a](https://replicate.com/replicate/dreambooth/versions/9c41656f8ae2e3d2af4c1b46913d7467cd891f2c1c5f3d97f1142e876e63ed7a/api#inputs) supports a `ckpt_base` - a file that start training from an existing checkpoint.
- [cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa](https://replicate.com/replicate/dreambooth/versions/cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa/api#inputs) - Dreambooth for SD1.5

The Replicate trainers are opensource [github.com/replicate/dreambooth](https://github.com/replicate/dreambooth).  Which means you can help improve the trainers by submitting a pull request.

But you don't have to wait for a new trainer version to be released to use it.  You can specify a model version you created and uploaded by setting the `trainer_version` field.  This allows you to test out new trainers before they are released, or add custom functionality to the trainer for your own use.

The only requirement is that the trainer returns an archive of the weights with the name `output.zip`.  The archive can contain any files you want, but need to be supported by the Cog model you specify in the Build step.  For the models we have built, the archive contains the result of `pipeline.save_pretrained` using the [diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers) library.

**tl;dr**: You can use any trainer you want, as long as it returns an archive of the weights with the name `output.zip`.  The trainer runs as a normal prediction on the Replicate platform, so you can use the [Replicate API](https://replicate.com/docs/reference/http) to see your training predictions.

#### Step 2: Build

Your training prediction produced a weights archive zip.  Now we need to build a Cog model that can load the weights and run inference.

To make that step easier, you can now specify a `template_version` the same way you specified the trainer.

`template_version` are models that are missing the weights.  They expect the stable diffusion weights to be added during the build step.

You can think of the build step as:

1. Download the weights archive
2. unzip the weights archive to a directory called `weights`
3. Create a Dockerfile that copies the weights to the Cog model based on the `template_version`

```dockerfile
FROM r8.im/anotherjesse/controlnet-1.5-pose-template@sha256:0f5cfc3e2a0e86dbd141057501ba5196c7dbea94c45dab4894e6ff7d6a2cc324

COPY weights /src/weights
```

4. docker build

Except you don't have to do anything!  Replicate does all of that for you.

Some templates that exist on replicate.com:

- [vanilla (diffusers)](https://replicate.com/anotherjesse/vanilla-template)
- [controlnet-1.5-canny-template](https://replicate.com/anotherjesse/controlnet-1.5-canny-template)
- [controlnet-1.5-depth-template](https://replicate.com/anotherjesse/controlnet-1.5-depth-template)
- [controlnet-1.5-hed-template](https://replicate.com/anotherjesse/controlnet-1.5-hed-template)
- [controlnet-1.5-mlsd-template](https://replicate.com/anotherjesse/controlnet-1.5-mlsd-template)
- [controlnet-1.5-normal-template](https://replicate.com/anotherjesse/controlnet-1.5-normal-template)
- [controlnet-1.5-pose-template](https://replicate.com/anotherjesse/controlnet-1.5-pose-template)
- [controlnet-1.5-scribble-template](https://replicate.com/anotherjesse/controlnet-1.5-scribble-template)

To use a template, find the version number on the versions page.  For example, I find that the version on the [controlnet-1.5-pose-template version page](https://replicate.com/anotherjesse/controlnet-1.5-pose-template/versions) is `0f5cfc3e2a0e86dbd141057501ba5196c7dbea94c45dab4894e6ff7d6a2cc324`.

#### Step 3: Push

Now we are ready to push to your `model` and add any `notes`.  

This step takes a minute or so to run, after which you can use the model in your own code.

After the push is complete you will have a new version of your model.  

If you pass notes, those will be added to the notes section for versions of the model (see [Stable Diffusion Versions](https://replicate.com/stability-ai/stable-diffusion/versions) for an example).

#### Step 4: Webhook

After the build is complete, if you requested a webhook, it will be sent.

You can include query parameters in the webhook URL, allowing you to pass information about the training back to your own code.

Included in the json payload is a lot of data, including:

- `id` - the id of the training prediction
- `status` - the status of the training: `succeeded` or `failed`
- `metrics.predict_time` - the time it took to run the training prediction
- `model` - the model name of the model you pushed to
- `version` - the version of the model this training created

```json
{
  "created_at": "2023-03-20T15:39:05.186935Z",
  "error": null,
  "id": "xxxxxxxxxxxxxxxxxxxxxx",
  "input": {
    "instance_prompt": "a photo of a cjw person",
    "class_prompt": "a photo of a person",
    "instance_data": "https://example.com/person.zip",
    "max_train_steps": 2000
  },
  "logs": null,
  "metrics": {
    "predict_time": 431.063869
  },
  "model": "yourusername/yourmodel",
  "notes": "notes about this dreambooth training",
  "status": "succeeded",
  "webhook_completed": "https://xxxxxxxxxx/xxxxxxxxxx",
  "version": "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### Putting it all together

Let's look at this request again:

```json
{
  "input": {
    "instance_prompt": "a photo of a cjw person",
    "class_prompt": "a photo of a person",
    "instance_data": "https://example.com/person.zip",
    "max_train_steps": 2000
  },
  "trainer_version": "cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa",
  "template_version": "0f5cfc3e2a0e86dbd141057501ba5196c7dbea94c45dab4894e6ff7d6a2cc324",
  "model": "yourusername/yourmodel",
  "notes": "notes about this dreambooth training",
  "webhook_completed": "https://example.com/dreambooth-webhook"
}
```

This says:

1. Training with SD1.5
  - use the `cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa` as `trainer_version`
  - `input` is sent to the trainer as `instance_prompt`, `class_prompt`, and `instance_data` (along with any other fields that the specific trainer expects)
  - a prediction using trainer version, input is visible there, the output will contain a weights archive
2. Build with ControlNet 1.5 OpenPose Template
  - use the `0f5cfc3e2a0e86dbd141057501ba5196c7dbea94c45dab4894e6ff7d6a2cc324` as `template_version`
  - `weights` are added from the training prediction output ontop of the template
3. Push to `model` 
  - in this example `yourusername/yourmodel`, but yours should be your actual username and model name...
  - `notes` are added to the version page for the model
4. Webhook
  - `webhook_completed` is sent when the build is complete
  - webhook contains the training duration as well as the model version that was created


## TODO

- [ ] documentation
- [x] controlnet 1.5 canny
- [x] controlnet 1.5 depth
- [x] controlnet 1.5 hed
- [x] controlnet 1.5 mlsd
- [x] controlnet 1.5 normal
- [x] controlnet 1.5 openpose
- [x] controlnet 1.5 scribble
- [ ] controlnet 1.5 seg
- [ ] controlnet + img2img - [community img2img support](https://github.com/huggingface/diffusers/pull/2584)
- [ ] add [weighted prompts](https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts#how-to-do-promptweighting-in-diffusers) to everything!
- [x] vanilla stable diffusion (txt2img, img2img)
- [ ] inpainting support (for which models?)
- [ ] t2i on top (for which models)?
- [ ] LoRA inference on top (for all models?)
