
import gradio as gr
from unsloth import FastLanguageModel
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# === Configuration ===
MODEL_NAME = "skapl/lora_model"  # Replace with the actual fine-tuned model name
MAX_SEQ_LENGTH = 512  # Adjust based on your model's requirements
DTYPE = torch.float16  # Use float32 or float16 depending on your setup
LOAD_IN_4BIT = False  # Set to True if using reduced precision

# === Load Models ===
# Load fine-tuned language model (Lore Model)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)  # Optimize the model for inference

# Load BLIP image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === Define Functionality ===
def describe_image(image: Image):
    """
    Generate a description for an uploaded image.
    Combines BLIP-generated caption with a detailed response from the fine-tuned language model.
    """
    # Step 1: Generate a basic caption using BLIP
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = image_model.generate(**inputs)
    blip_caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Step 2: Enhance the caption using the fine-tuned language model
    prompt = f"Describe this image: {blip_caption}"
    model_inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are an image description assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate a detailed description
    generated_ids = model.generate(
        input_ids=model_inputs,
        max_new_tokens=150,
        temperature=0.8
    )
    detailed_description = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return detailed_description

# === Gradio Interface ===
gr.Interface(
    fn=describe_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Image Description"),
    title="Image Description Assistant",
    description="Upload an image, and the bot will describe it with rich details.",
).launch()
