from PIL import Image
import requests
import torch

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Load image
print("Loading image...")
image = Image.open(requests.get(
    "https://llava-vl.github.io/static/images/view.jpg",
    stream=True
).raw)

# Always load LlavaRunner (never use preloaded model)
print("Loading model via LlavaRunner...")

from analysis.llava_runner import LlavaRunner

runner = LlavaRunner(
    model_id="llava-hf/llava-1.5-7b-hf",
    device="cuda" if torch.cuda.is_available() else "cpu",
    quantization=None
)

print("Running inference with full attention + layer evolution...")

output = runner.run(
    image=image,
    prompt="What do you see in this image?",
    prefix="This image shows",
)

print("\n" + "="*50)
print("RESULTS (Full Attention Tracking + Layer Evolution)")
print("="*50)
print(f"Predicted answer: {output.predicted_answer}")
print(f"Token confidence: {output.token_confidence:.4f}")
print(f"Head delta: {output.head_delta:.6f}")
print(f"Attention map shape: {output.attention_map.shape}")
print(f"Generated tokens: {len(output.predicted_token_ids)}")
print(f"Tokens: {output.token_strings}")
print("="*50)
