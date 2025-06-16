import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Prompt input
prompt = input("Enter your text prompt: ")

# Set device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
)
pipe = pipe.to(device)

# Generate the image
print(f"Generating image for prompt: {prompt}")
image = pipe(prompt).images[0]

# Save the image
output_path = "generated_image.png"
image.save(output_path)

print(f"Image saved at: {os.path.abspath(output_path)}")
image.show()
