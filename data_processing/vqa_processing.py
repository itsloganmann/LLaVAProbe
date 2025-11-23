import json
import os

# Paths
annotations_path = "./data/raw/v2_mscoco_train2014_annotations.json"
questions_path = "./data/raw/v2_OpenEnded_mscoco_train2014_questions.json"
processed_data_path = "./data/processed/filtered_vqa_with_links.json"

# Create directories
os.makedirs("./data/processed", exist_ok=True)

# Category Mapping
CATEGORY_MAPPING = {
    "what is this": "object identification",
    "what color": "color recognition",
    "how many": "counting",
    "what kind": "classification",
    "where": "location",
    "who": "person identification",
    "when": "time-related",
    "why": "reasoning",
    "which": "comparison",
    "other": "other",
}

def categorize_question(question_text, answer):
    """Assigns a category based on question text."""
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in question_text.lower():
            return category
    return "yes/no" if answer.lower() in ["yes", "no"] else "other"

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Load questions
with open(questions_path, "r") as f:
    questions = json.load(f)

# Create a mapping from question_id to its text
question_text_map = {q["question_id"]: q["question"] for q in questions["questions"]}

# Extract (image, question, answer) + generate image links
filtered_data = []
for ann in annotations["annotations"][:5000]:
    question_id = ann["question_id"]
    image_id = str(ann["image_id"]).zfill(12)

    # Get the question text
    question_text = question_text_map.get(question_id, "Unknown question")
    
    # Generate image link
    image_url = f"http://images.cocodataset.org/train2014/COCO_train2014_{image_id}.jpg"
    
    entry = {
        "image_id": image_id,
        "image_url": image_url,
        "question_id": question_id,
        "question_text": question_text,
        "category": categorize_question(question_text, ann["multiple_choice_answer"]),
        "answer": ann["multiple_choice_answer"]
    }
    filtered_data.append(entry)

# Store in processed folder
with open(processed_data_path, "w") as f:
    json.dump(filtered_data, f, indent=4)

print(f"✅ Processed {len(filtered_data)} entries and saved to {processed_data_path}")