import argparse
import os
import pandas as pd
import csv
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
        "--save_file", 
        type=str, 
        required=True, 
        help="Directory to save the text result to."
    )
    
    args = parser.parse_args()
        
    return args

def append_csv_data(filename, new_data_rows):
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.exists(filename)
    
    # Open the file in append mode ('a')
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Optionally write a header only if the file did not exist before
        # Note: You need to know your header structure to use this
        if not file_exists:
            # Example Header:
            header = ['Response']
            writer.writerow(header)
        
        # Write the new rows of data
        writer.writerows(new_data_rows)
        
    print(f"Successfully appended {len(new_data_rows)} rows to {filename}.")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen3-VL-8B-Instruct",
     dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     device_map="auto",
 )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

args = parse_args()
splits = ['train', 'val']

for split in splits:
    input_dir = os.path.join(args.image_dir, split)
    for img in os.listdir(input_dir):
        full_path = os.path.join(input_dir, img)
        question = """Perform an in-depth visual analysis of the image. Structure your response using the following three headings, ensuring high detail in each section:

Composition & Aesthetics: Describe the shot type (e.g., wide, close-up), the color palette, the use of foreground/background, and the primary light source.

Narrative & Subject: Identify all key subjects and describe the central activity or narrative moment captured. What story does the image attempt to tell?

Potential Use & Context: Based on the style, suggest where this image might typically be used (e.g., editorial, commercial, documentary) and which audience it is targeting."""

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
        print(f"{full_path}: {output_text}")
        append_csv_data(args.save_file, output_text[0])
