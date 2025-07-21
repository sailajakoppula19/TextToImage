# text_to_image_gui.py (Upgraded for Performance)
import gradio as gr
import torch
from transformers import BertTokenizer
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline
import numpy as np

# Load BERT tokenizer for Task 1
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Simulated CGAN Generator for Task 3
def generate_cgan_shape(text):
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if text == "circle":
        draw.ellipse((16, 16, 48, 48), fill=(0, 0, 0))
    elif text == "square":
        draw.rectangle((16, 16, 48, 48), fill=(0, 0, 0))
    elif text == "triangle":
        draw.polygon([(32, 16), (16, 48), (48, 48)], fill=(0, 0, 0))
    return img

# Load Stable Diffusion model for Tasks 4 & 5
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

# Optional: Disable safety checker
pipe.safety_checker = lambda images, clip_input: (images, False)

# Enable performance optimizations
pipe = pipe.to(device)
pipe.enable_attention_slicing()
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

# Stable Diffusion generation function
def generate_stable_diffusion(prompt):
    image = pipe(
        prompt,
        num_inference_steps=25,   # Reduced from 30 for speed
        guidance_scale=7.0        # Balanced generation
    ).images[0]
    return image

# Task 1: Tokenize and Encode Text
def tokenize_encode_text(text):
    tokens = tokenizer.tokenize(text)
    encoding = tokenizer.encode(text)
    return f"Tokens:\n{tokens}\n\nEncoding:\n{encoding}"

# Gradio Interfaces
task1_interface = gr.Interface(
    fn=tokenize_encode_text,
    inputs=gr.Textbox(label="Enter Text", lines=2, placeholder="Type here..."),
    outputs="text",
    title="Task 1: BERT Tokenization & Encoding"
)

task3_interface = gr.Interface(
    fn=generate_cgan_shape,
    inputs=gr.Dropdown(["circle", "square", "triangle"], label="Choose Shape"),
    outputs="image",
    title="Task 3: CGAN (Simulated Shape Generator)"
)

task4_interface = gr.Interface(
    fn=generate_stable_diffusion,
    inputs=gr.Textbox(label="Enter Prompt for Image", lines=2, placeholder="e.g., A fantasy landscape"),
    outputs="image",
    title="Tasks 4 & 5: Stable Diffusion Text-to-Image"
)

# Combine all tasks in tabs
demo = gr.TabbedInterface(
    [task1_interface, task3_interface, task4_interface],
    tab_names=["Task 1: BERT", "Task 3: CGAN", "Task 4 & 5: Stable Diffusion"]
)

# Launch Gradio App with share=True if needed
demo.launch()
