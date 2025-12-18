import argparse
import os
import PIL.Image
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# Import necessary components from the provided code structure
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# --- 1. CONFIGURATION ---

# Specify the path to the model
MODEL_PATH = "deepseek-ai/Janus-Pro-7B"

# --- 2. GENERATION FUNCTION (MODIFIED FOR OUTPUT) ---

@torch.inference_mode()
def generate_images(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    save_dir: str,
    prompt_index,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    """
    Generates images from a single prompt using the provided VLM.
    
    The function executes the CFG-guided, token-by-token generation process
    and saves the resulting images to the specified directory.
    """
    start_time = time.time()
    # --- Prepare Input Tokens ---
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # Prepare batch for CFG: [Cond_1, Uncond_1, Cond_2, Uncond_2, ...]
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            # Unconditional prompt: often achieved by masking the main content
            # The original code uses vl_chat_processor.pad_id for unconditional context.
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    # Get input embeddings
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    
    # Initialize past_key_values for fast generation
    outputs = None 
    
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    print(f"Starting T2I generation for {parallel_size} samples...")

    # --- Token-by-Token Generation Loop ---
    for i in range(image_token_num_per_image):
        # 1. Forward pass (uses cache for speed)
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        
        hidden_states = outputs.last_hidden_state
        
        # 2. Get logits and separate Cond/Uncond
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]    # Conditional logits (from prompt)
        logit_uncond = logits[1::2, :]  # Unconditional logits (from masked prompt)
        
        # 3. Apply Classifier-Free Guidance (CFG)
        # logit_cfg = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        logits_cfg = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits_cfg / temperature, dim=-1)

        # 4. Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # 5. Prepare input embeddings for the next step (auto-regressive input)
        next_token_input = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    print("Generation complete. Starting VAE decoding...")

    # --- 3. VAE Decoding and Saving ---
    
    # Decode: latent tokens -> image pixels
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), 
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    # Normalize and clamp pixels to [0, 255]
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # Save images
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(parallel_size):
        # Use prompt_index to ensure unique filenames when reading from a file
        save_path = os.path.join(save_dir, f"{prompt_index}.png")
        PIL.Image.fromarray(visual_img[i]).save(save_path)
    
    print(f"Successfully saved {parallel_size} images to {save_dir}")
    end_time = time.time()
    print(f"time {end_time - start_time:.4f}")
    return 

# --- 4. MAIN EXECUTION BLOCK ---

def parse_args():
    parser = argparse.ArgumentParser(description="T2I inference script for Janus-Pro model.")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None, 
        help="Single text prompt for image generation."
    )
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default=None, 
        help="Path to a text file containing multiple prompts (one per line)."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True, 
        help="Directory to save the generated images."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=16, 
        help="Number of parallel samples to generate per prompt."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="The final pixel resolution of the generated image (e.g., 384x384)."
    )
    # Add other parameters from the generate function for flexibility
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="Classifier-Free Guidance weight.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    
    args = parser.parse_args()
    
    if args.prompt is None and args.prompt_file is None:
        parser.error("Must provide either --prompt or --prompt_file.")
        
    return args

def main():
    args = parse_args()
    
    # --- Model Loading (Only once) ---
    print(f"Loading model from {MODEL_PATH}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_PATH)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print("Model loaded successfully.")
    
    # --- Prepare Prompt List ---
    prompts_to_run: List[str] = []
    
    if args.prompt:
        prompts_to_run.append(args.prompt)
    df = prompts_to_run
    
    if args.prompt_file:
        df = pd.read_csv(args.prompt_file)

    start_idx = 0
    end_idx = len(df)

    # --- Run Generation Loop ---
    for i in range(start_idx, end_idx):
        # Apply the required SFT (Supervised Fine-Tuning) template
        raw_prompt = df.loc[i, "prompt"]
        image_id = df.loc[i, "image_id"]
        
        if os.path.exists(os.path.join(args.save_dir, f"{image_id}.png")):
            print(f"{image_id} alredy exists")
            continue
        conversation = [
            {"role": "<|User|>", "content": str(raw_prompt)},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        final_prompt = sft_format + vl_chat_processor.image_start_tag
        
        print(f"\n--- Generating for Prompt {i+1}/{len(prompts_to_run)}: {raw_prompt[:50]}... ---")

        # Call the modified generation function
        generate_images(
            vl_gpt,
            vl_chat_processor,
            final_prompt,
            save_dir=args.save_dir,
            prompt_index=image_id,
            temperature=args.temperature,
            parallel_size=args.num_samples,
            cfg_weight=args.cfg_weight,
            # Pass fixed model params (could be added to args if needed)
            image_token_num_per_image=576, # 384/16 = 24. 24*24 = 576
            img_size=args.img_size,
            patch_size=16,
        )
        print(f"image_{i} done")

if __name__ == '__main__':
    main()
