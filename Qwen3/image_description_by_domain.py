import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Literal

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
#)

def parse_args():
    parser = argparse.ArgumentParser(description="I2T inference of Qwen3 VL")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default=None, 
        help="Directory to read the images from."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True, 
        help="Directory to save the text result to."
    )
    
    args = parser.parse_args()
        
    return args

def process_vlm_response(text_response: str) -> Literal[0, 1]:
    cleaned_response = text_response.strip().lower()

    if cleaned_response.startswith('yes'):
        return 1

    if cleaned_response.startswith('no'):
        return 0

    raise ValueError(
        f"VLM output '{text_response}' is ambiguous or not a binary answer. "
        "Expected the response to start with 'yes' or 'no'."
    )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen3-VL-8B-Instruct",
     dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

args = parse_args()
models = ['bagel', 'emu', 'janus', 'mmada', 'showo2', 'gpt', 'gemini']
domains = ['animals', 'arts_and_works', 'buildings', 'clothing', 'food_and_drinks', 'household_items', 'interior_spaces', 'landscapes', 'people', 'vehicles']
splits = ['train', 'val']

matrix = {d1: {d2: None for d2 in domains} for d1 in domains}

for domain in domains:
    for m in models:

        for question_domain in domains:
            question = f"In the image, do you see {question_domain}? Answer the question with just yes or no."
            yes_count = 0
            total = 0
            for split in splits:
                input_dir = os.path.join(args.image_dir, domain, m, split)

                for img in os.listdir(input_dir):
                    full_path = os.path.join(input_dir, img)

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": full_path,
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ]

                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    inputs = inputs.to(model.device)

                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    print(f"{full_path}: {question} -> {output_text}")
                    
                    binary = process_vlm_response(output_text[0])
                        
                    yes_count += binary
                    total += 1

            matrix[domain][question_domain] = yes_count / total * 100 # make it into %
    
df = pd.DataFrame(matrix).T

plt.figure(figsize=(14, 10))

df_heat = df.astype(float)

vmin = df_heat.min().min()
vmax = df_heat.max().max()

def text_color(value):
    midpoint = vmin + 0.5 * (vmax - vmin)
    return "white" if value < midpoint else "black"

ax = sns.heatmap(
    df_heat,
    cmap="viridis",
    linewidths=0.7,
    linecolor="black",
    cbar_kws={'label': 'Domain frequency (%)'},
    annot=False
)

for y in range(df_heat.shape[0]):
    for x in range(df_heat.shape[1]):
        val = df_heat.iloc[y, x]
        ax.text(
            x + 0.5, y + 0.5,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=text_color(val)
        )

plt.title(
    "Domain Frequency Heatmap (D1 = rows, D2 = columns)",
    fontsize=22, pad=20, fontweight="bold"
)

plt.xlabel("Domain Question Asked (D2)", fontsize=16)
plt.ylabel("Original Domain (D1)", fontsize=16)

plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(args.save_dir, "domain_frequency.png"), dpi=300)

print("Saved heatmap")


