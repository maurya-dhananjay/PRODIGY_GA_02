from diffusers import StableDiffusionPipeline
import torch


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


prompt = "A futuristic robot working on a laptop"


image = pipe(prompt).images[0]


image.save("outputs/generated_image.png")

print("Image generated successfully!")
