import os
import random
import csv
import json
import torch
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict

import openai
from dataloader import dataloader
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ————————————————————————————————————————————————————————————
# 1) Configuration
openai.api_key = 'api-key-here'  # replace with your OpenAI API key
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ————————————————————————————————————————————————————————————
# 2) ChatGPT-based prefix generator
_prefix_cache = {}
def generate_prefix(question: str) -> str:
    if question in _prefix_cache:
        return _prefix_cache[question]
    system_msg = {
        "role": "system",
        "content": (
            "You are a prefix making assistant. Do NOT answer the question. "
            "Given a single question, output only a short prefix that naturally "
            "precedes the one-word or one-phrase answer to the question."
        )
    }
    user_msg = {"role": "user", "content": f"Question: \"{question}\"\nPrefix:"}
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_msg, user_msg],
            max_tokens=8,
            temperature=0.0,
            stop=["\n"]
        )
        prefix = resp.choices[0].message.content.strip().rstrip(".") + ":"
    except Exception:
        prefix = "Answer:"
    _prefix_cache[question] = prefix
    return prefix

# ————————————————————————————————————————————————————————————
# 3) Build dynamic instruction via ChatGPT with few-shot examples
_instruction_cache = {}
def build_instruction(question: str, qtype: str) -> str:
    key = f"{qtype}||{question}"
    if key in _instruction_cache:
        return _instruction_cache[key]

    system_msg = {
        "role": "system",
        "content": """
You are an assistant that crafts clear, concise instruction prefixes for visual question answering.
Below are examples:

Type: Object Identification
Question: “What is the large animal in the background?”
Instruction: Answer in one word which animal is largest:

Type: Color Recognition
Question: “What color is the umbrella?”
Instruction: Respond with exactly one color the color of the umbrella

Type: Yes/No
Question: “Is there a stop sign visible?”
Instruction: Answer yes if the sign is visible and no if it is not

Type: Counting
Question: “How many bicycles are parked?”
Instruction: only output a number and nothing else.

Now, given a question type and text, output the single best instruction.
"""
    }
    user_msg = {
        "role": "user",
        "content": f"Type: {qtype}\nQuestion: \"{question}\"\n\nInstruction:"
    }
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_msg, user_msg],
            max_tokens=30,
            temperature=0.0,
            stop=["\n"]
        )
        instruction = resp.choices[0].message.content.strip()
    except Exception:
        instruction = f"Answer with one word: {question}"
    _instruction_cache[key] = instruction
    return instruction

# ————————————————————————————————————————————————————————————
# 4) LLaVA wrapper
@dataclass
class LlavaWrapper:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    device: torch.device = DEVICE

    def __post_init__(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            revision="a272c74"
        ).to(self.device)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(self.model_id, revision="a272c74")

    def infer_top4(self, pil_img, instruction: str):
        full = f"USER: <image>\n{instruction}"
        inputs = self.proc(text=full, images=pil_img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        logits = out.logits[0, -1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idxs = torch.argsort(probs, descending=True)[:10]
        raw = [self.proc.tokenizer.decode(i.item(), skip_special_tokens=True).strip() for i in idxs]

        bad = {"a","the","</s>","",".",",","?","on","in","and","of","to","for"}
        filtered = [t for t in raw if t.lower() not in bad and len(t) > 2]
        top4 = filtered[:4]
        if len(top4) < 4:
            for t in raw:
                if t not in top4:
                    top4.append(t)
                if len(top4) == 4:
                    break
        return top4[0], top4

# ————————————————————————————————————————————————————————————
# 5) Group question indices by type
raw = dataloader.dataset.data   # this is your JSON list with image_url field
type_to_idxs = defaultdict(list)
for i, sample in enumerate(raw):
    type_to_idxs[sample.get("category", "<None>")].append(i)

# ————————————————————————————————————————————————————————————
# 6) Run model & write CSV for all samples (including image_url)
csv_path = "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow([
        "question_type",
        "question",
        "prefix",
        "prompt",
        "ground_truth",
        "top_4_tokens",
        "model_answer",
        "image_url",
    ])
    llava = LlavaWrapper()
    total = 0

    # iterate every sample
    for qtype, idxs in type_to_idxs.items():
        for idx in idxs:
            question, img_t, _, gt = dataloader.dataset[idx]
            # pull the URL from raw JSON
            img_url = raw[idx].get("image_url", "")

            # skip non-RGB
            if img_t.dim() != 3 or img_t.shape[0] != 3:
                continue

            # to PIL
            pil = Image.fromarray(
                (img_t.permute(1,2,0).cpu().numpy() * 255).astype("uint8")
            )

            # build and run
            prefix = generate_prefix(question)
            instr  = build_instruction(question, qtype)
            prompt = f"{instr} {question} {prefix}"
            answer, top4 = llava.infer_top4(pil, prompt)

            # write row
            writer.writerow([
                qtype,
                question,
                prefix,
                prompt,
                gt,
                json.dumps(top4),
                answer,
                img_url,
            ])
            total += 1

print(f"✅ Done! Wrote {total} rows to {csv_path}")

