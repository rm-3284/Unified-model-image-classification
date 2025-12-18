# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
from datetime import datetime # ADDED: For creating unique folder names

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
import torch
# REMOVED: import wandb (and related imports like wandb.util)
from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn.functional as F

def resize_vocab(model, config):
    print(f"Resizing token embeddings to {config.new_vocab_size}")
    model.resize_token_embeddings(config.new_vocab_size)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    # --- REMOVED WANDB INITIALIZATION LOGIC ---
    # The following lines related to wandb run ID generation are removed.
    # resume_wandb_run = config.wandb.resume
    # run_id = config.wandb.get("run_id", None)
    # if run_id is None:
    #     resume_wandb_run = False
    #     run_id = wandb.util.generate_id()
    #     config.wandb.run_id = run_id
    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
    # wandb.init(project="demo", name=config.experiment.name + '_t2i', config=wandb_config)
    
    # --- ADDED LOCAL OUTPUT DIRECTORY SETUP ---
    # Create a unique output folder name based on experiment name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.get("save_dir", None) is None:
        OUTPUT_DIR = f"generated_images/{config.experiment.name}_{timestamp}"
    else:
        OUTPUT_DIR = config.get("save_dir", None)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving generated images to: {OUTPUT_DIR}")
    # --- END LOCAL OUTPUT SETUP ---


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)


    model.to(device)

    mask_token_id = model.config.mask_token_id
    #if config.get("validation_prompts_file", None) is not None:
    #    config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = 1

    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    #with open(config.dataset.params.validation_prompts_file, "r") as f:
    #    validation_prompts = f.read().splitlines()
    #validation_prompts = [config.get("prompt")]
    prompt_file = config.get("prompt_file")
    import pandas as pd
    df = pd.read_csv(prompt_file)
    validation_prompts = df
    #validation_prompts = ["Create an image of Mt. Fuji"]

    # Initialize a counter for unique image filenames
    global_image_counter = 0 
    start_step = 0
    end_step = len(df)
    for step in tqdm(range(start_step, end_step, config.training.batch_size)):
        start_time = time.time()
        prompts = validation_prompts.loc[step:step + config.training.batch_size, "prompt"].tolist()
        ids = validation_prompts.loc[step:step + config.training.batch_size, "image_id"].tolist()

        if os.path.exists(os.path.join(OUTPUT_DIR, f"{ids[0]}.png")):
            print(f"{ids[0]}.png already exists")
            continue
        

        #prompts = validation_prompts
        print(prompts, flush=True)
        image_tokens = torch.ones((len(prompts), config.model.mmada.num_vq_tokens),
                                    dtype=torch.long, device=device) * mask_token_id
        input_ids, attention_mask = uni_prompting((prompts, image_tokens), 't2i_gen')
        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_schedule(schedule, **args)
        else:
            mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        end_time = time.time()
        print(f"resolution 1024, time {end_time - start_time: .4f}", flush=True)
        print(f"{len(pil_images)} images generated")

        # --- REPLACED WANDB LOGGING WITH LOCAL SAVE LOGIC ---
        #image = pil_images[0]
        #image_filename = os.path.join(OUTPUT_DIR, "test2.png")
        #image.save(image_filename)
        for i, image in enumerate(pil_images):
            # Create a unique filename based on the global counter
            id = ids[i]
            image_filename = os.path.join(OUTPUT_DIR, f"{id}.png")
            print(f"{image_filename} saved")
            image.save(image_filename)
            global_image_counter += 1
            
            # Optional: Log the prompt alongside the saved file name
            # print(f"Saved: {image_filename} (Prompt: {prompts[i][:50]}...)")
        print(f"step {step} done")
        # --- END LOCAL SAVE LOGIC ---
        
        # REMOVED: wandb.log({"generated_images": wandb_images}, step=step)
