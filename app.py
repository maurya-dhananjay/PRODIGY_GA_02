from diffusers import StableDiffusionPipeline
import torch

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Text prompt
prompt = "A futuristic robot working on a laptop"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("outputs/generated_image.png")

print("Image generated successfully!")
