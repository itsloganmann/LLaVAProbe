from PIL import Image
import requests
import torch

print(f"Using device: cuda" if torch.cuda.is_available() else "cpu")

# Load a real image
print("Loading image...")
image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)

# OPTION 1: Use already-loaded model from your previous cell
# If you already have 'model' and 'processor' loaded in memory, skip LlavaRunner
try:
    # Check if model already exists in namespace (from previous Colab cell)
    model
    processor
    print("✅ Using already-loaded model from previous cell!")
    use_existing_model = True
except NameError:
    # Model not loaded yet, use LlavaRunner
    print("Loading model via LlavaRunner...")
    from analysis.llava_runner import LlavaRunner
    
    runner = LlavaRunner(
        model_id="llava-hf/llava-1.5-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        quantization=None
    )
    use_existing_model = False

# Run inference with attention tracking
print("Running inference with attention tracking...")

if use_existing_model:
    # Use the already-loaded model and processor directly
    prompt = "USER: <image>\nWhat do you see in this image?\nASSISTANT: This image shows"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
            output_hidden_states=True
        )
    
    # Decode the output
    generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    predicted_answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("\n" + "="*50)
    print("RESULTS (Direct Model Inference)")
    print("="*50)
    print(f"Predicted answer: {predicted_answer}")
    print(f"Generated tokens: {len(generated_ids)}")
    print("="*50)
    print("\n⚠️ Note: Full attention tracking requires LlavaRunner.")
    print("   This is a basic inference test only.")
    
else:
    # Use LlavaRunner for full attention tracking
    output = runner.run(
        image=image,
        prompt="What do you see in this image?",
        prefix="This image shows",
    )
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS (Full Attention Tracking)")
    print("="*50)
    print(f"Predicted answer: {output.predicted_answer}")
    print(f"Token confidence: {output.token_confidence:.4f}")
    print(f"Critical attention head delta: {output.head_delta:.6f}")
    print(f"Attention map shape: {output.attention_map.shape}")
    print(f"Number of generated tokens: {len(output.predicted_token_ids)}")
    print(f"Token strings: {output.token_strings}")
    print("="*50)
