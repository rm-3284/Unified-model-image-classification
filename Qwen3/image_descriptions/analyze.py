import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re
from collections import Counter

STYLE_DICT = {
    'Medium': ['watercolor', 'oil painting', 'pencil sketch', '3d render', 'digital art', 'photograph', 'illustration'],
    'Lighting': ['cinematic', 'volumetric', 'golden hour', 'soft lighting', 'neon', 'high contrast', 'backlit'],
    'Composition': ['wide-angle', 'close-up', 'bokeh', 'fisheye', 'symmetrical', 'birds eye view', 'macro'],
    'Color': ['vibrant', 'pastel', 'monochrome', 'sepia', 'saturated', 'earthy tones', 'muted']
}

def safe_load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip()
    
    if content.count('"') % 2 != 0:
        content += '"'
    
    from io import StringIO
    try:
        return pd.read_csv(StringIO(content), on_bad_lines='skip')
    except Exception:
        return pd.DataFrame({'Response': [content]})

def compare_models(directory_path, description_column='Response'):
    all_model_stats = []
    
    files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    for file_path in files:
        model_name = os.path.basename(file_path).replace(".csv", "")
        print(f"Processing model: {model_name}...")
        
        try:
            df = safe_load_csv(file_path)
            if df.empty:
                continue
            
            all_text = " ".join(df[description_column].astype(str)).lower()
            
            stats = {'Model': model_name}
            for category, keywords in STYLE_DICT.items():
                total_cat_hits = sum(len(re.findall(rf'\b{word}\b', all_text)) for word in keywords)
                stats[category] = total_cat_hits
            
            all_model_stats.append(stats)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    comparison_df = pd.DataFrame(all_model_stats).set_index('Model')

    row_sums = comparison_df.sum(axis=1)
    normalized_df = comparison_df.div(row_sums, axis=0) * 100

    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Style Prevalence (%)'})
    plt.title('Stylistic Profile Comparison Across Models', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('model_comparison_heatmap.png')
    
    comparison_df.to_csv('master_style_comparison.csv')
    print("\nComparison saved to 'master_style_comparison.csv' and 'model_comparison_heatmap.png'")

compare_models("/n/fs/vision-mix/rm4411/Qwen3/image_descriptions/fixed_descriptions")
