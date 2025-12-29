from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "runwayml/stable-diffusion-v1-5"

current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

print("Saving images to:", output_dir)

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")


prompt = "A futuristic robot working on a laptop"


image = pipe(prompt).images[0]

image_path = os.path.join(output_dir, "generated_image.png")
image.save(image_path)

print(" Image generated successfully at:", image_path)
