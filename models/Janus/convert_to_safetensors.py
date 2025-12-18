import torch
from safetensors.torch import save_file
import os

MODEL_PATH = "deepseek-ai/Janus-Pro-7B"

# 1. Download the PyTorch weights locally first (e.g., using huggingface_hub)
#    (You might already have these files in your local cache)

# 2. Iterate and Convert
def convert_to_safetensors(model_path):
    # Find all PyTorch checkpoint files (e.g., pytorch_model-00001-of-0000X.bin)
    pt_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
    
    for pt_file in pt_files:
        pt_path = os.path.join(model_path, pt_file)
        sf_path = pt_path.replace('.bin', '.safetensors')
        
        print(f"Converting {pt_file}...")
        
        # Load the state dict (assuming you can do this outside the restricted loader)
        state_dict = torch.load(pt_path, map_location="cpu")
        
        # Save the file using the safetensors format
        save_file(state_dict, sf_path, metadata={"format": "pt"})
        print(f"Saved to {sf_path}")

# Run the conversion (Adjust MODEL_PATH if necessary)
convert_to_safetensors(MODEL_PATH)
