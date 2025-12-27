import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms import ToTensor, Resize, Compose

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROCESSED_DATA_PATH = "data_processing/data/processed/filtered_vqa_with_links.json"

# ---------------------------------------------------------
# Load processed VQA data
# ---------------------------------------------------------
try:
    with open(PROCESSED_DATA_PATH, "r") as f:
        dataset = json.load(f)

    # (Optional) limit to first N samples while debugging
    # comment this out later for full run
    # dataset = dataset[:20]

    print(f"✅ Successfully loaded dataset from {PROCESSED_DATA_PATH}")
except FileNotFoundError:
    print(f"❌ Error: File not found at {PROCESSED_DATA_PATH}")
    print(
        "➡️ Run the preprocessing script in data_processing/ that creates "
        "`filtered_vqa_with_links.json` before running the analysis."
    )
    sys.exit(1)  # ✅ use sys.exit instead of bare exit()

# ---------------------------------------------------------
# Inspect dataset structure (first 3 entries)
# ---------------------------------------------------------
print("\n🔍 First 3 dataset entries:")
for i in range(min(3, len(dataset))):
    print(json.dumps(dataset[i], indent=4))


# ---------------------------------------------------------
# Dummy inference (only used when running this file directly)
# ---------------------------------------------------------
def inference(question, image):
    """Simulate inference returning an answer and a 24x24 matrix."""
    answer = "dummy_answer"  # Replace with actual model output
    matrix = np.random.rand(24, 24)  # Random 24x24 matrix for demonstration
    return answer, matrix


# ---------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------
class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        question_text = sample.get("question_text", "[MISSING QUESTION TEXT]")
        image_url = sample.get("image_url", None)
        category = sample.get("category", "[MISSING CATEGORY]")
        answer = sample.get("answer", "[MISSING ANSWER]")

        image = torch.zeros(3, 224, 224)  # default if loading fails
        if image_url:
            try:
                response = requests.get(image_url, timeout=5)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = self.transform(image)
            except Exception as e:
                print(f"❌ Error loading image from {image_url}: {e}")

        return question_text, image, category, answer


# ---------------------------------------------------------
# DataLoader (this is what you import as `dataloader`)
# ---------------------------------------------------------
batch_size = 16
vqa_dataset = VQADataset(dataset)
dataloader = DataLoader(
    vqa_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: (
        [item[0] for item in batch],           # questions
        torch.stack([item[1] for item in batch]),  # images
        [item[2] for item in batch],           # categories
        [item[3] for item in batch],           # answers
    ),
)

# Make it importable as: from dataloader import dataloader
__all__ = ["dataloader"]


# ---------------------------------------------------------
# Optional: debug run when you call `python dataloader.py`
# ---------------------------------------------------------
def compute_entropy(probabilities):
    probabilities = np.array(probabilities)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


if __name__ == "__main__":
    # Simple sanity check loop
    for batch in dataloader:
        questions, images, categories, answers = batch

        answers_out = []
        matrices = []
        for q, img in zip(questions, images):
            ans, matrix = inference(q, img)
            answers_out.append(ans)
            matrices.append(matrix)

        answer_probs = np.random.dirichlet(np.ones(len(answers_out)), size=1)[0]
        entropy = compute_entropy(answer_probs)

        print(f"✅ Batch Processed | Entropy: {entropy:.4f} | Categories: {set(categories)}")

    print(f"\n📦 Total Samples in Dataset: {len(vqa_dataset)}")
