import os
import pandas as pd

save_dir = "/n/fs/vision-mix/rm4411/prompts_with_categories/MMaDA"

prompt_file = "/n/fs/vision-mix/rm4411/prompts_with_categories.csv"
df = pd.read_csv(prompt_file)
start_idx = 0
end_idx = len(df)

missing = False

for i in range(start_idx, end_idx):
    image_id = df.loc[i, "image_id"]
    file_path = os.path.join(save_dir, f"{image_id}.png")
    if os.path.exists(file_path):
        if missing:
            print(f"image {i} exists")
            missing = False
        continue
    else:
        #print(f"image {i} missing")
        if not missing:
            print(f"image {i} missing start")
            missing = True
