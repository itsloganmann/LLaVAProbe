from analysis.llava_runner import LlavaRunner
from PIL import Image
import requests
import torch

# Initialize the actual LLaVA model
# Note: Use "llava-hf/llava-1.5-7b-hf" for HuggingFace format
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

runner = LlavaRunner(
    model_id="llava-hf/llava-1.5-7b-hf",
    device=device,
    quantization=None  # Set to "4bit" or "8bit" for memory-efficient inference
)

# Load a real image
print("Loading image...")
image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)

# Run your full attention tracking on a real question
print("Running inference with attention tracking...")
output = runner.run(
    image=image,
    prompt="What do you see in this image?",
    prefix="This image shows",
)

# Print results
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Predicted answer: {output.predicted_answer}")
print(f"Token confidence: {output.token_confidence:.4f}")
print(f"Critical attention head delta: {output.head_delta:.6f}")
print(f"Attention map shape: {output.attention_map.shape}")
print(f"Number of generated tokens: {len(output.predicted_token_ids)}")
print(f"Token strings: {output.token_strings}")
print("="*50)
