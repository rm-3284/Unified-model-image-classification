import csv
import os

def fix_fragmented_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        fragments = [line.strip('\r\n') for line in f]
    
    fixed_text = "".join(fragments)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_text)
        
    print(f"Repaired file saved to: {output_path}")

base_dir = "/n/fs/vision-mix/rm4411/Qwen3/image_descriptions"
models = ['BAGEL', 'Emu3.5', 'Janus-Pro-7B', 'MMaDA', 'show-o2']
for model in models:
    filename = f"{model}.csv"
    fixed = f"{model}-fixed.csv"
    fix_fragmented_csv(os.path.join(base_dir, filename), os.path.join(base_dir, fixed))

