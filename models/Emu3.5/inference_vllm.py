# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import torch
import time

import importlib as imp
import os.path as osp

from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.model_utils import build_emu3p5_vllm
from src.utils.vllm_generation_utils import generate
from src.utils.generation_utils import multimodal_decode, decode_image
from src.utils.painting_utils import ProtoWriter
from src.utils.input_utils import build_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompts", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--cfg", default="", type=str)
    parser.add_argument("--tensor-parallel-size", default=2, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--seed", default=6666, type=int)
    args = parser.parse_args()
    return args


def inference(
    args,
    cfg,
    model,
    tokenizer,
    vq_model,
):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    #os.makedirs(save_path, exist_ok=True)
    #os.makedirs(f"{save_path}/proto", exist_ok=True)
    #proto_writer = ProtoWriter()

    prompt_file = args.prompts
    df = pd.read_csv(prompt_file)
    start_idx = args.start
    end_idx = len(df)
    print(f"read prompt file {prompt_file}", flush=True)

    #for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
    #    if osp.exists(f"{save_path}/proto/{name}.pb"):
    #        print(f"[WARNING] Result already exists, skipping {name}", flush=True)
    #        continue
    for i in range(start_idx, end_idx):
        image_id = df.loc[i, "image_id"]
        file_name = osp.join(save_path, f"{image_id}.png")
        if osp.exists(file_name):
            print(f"[WARNING] {file_name} already exists at step {i}")
            continue

        torch.cuda.empty_cache()

        question = df.loc[i, "prompt"]

        reference_image = None
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                print(f"[INFO] {len(question['reference_image'])} reference images are provided")
                reference_image = []
                for img in question["reference_image"]:
                    reference_image.append(Image.open(img).convert("RGB"))
            else:
                print (f"[INFO] 1 reference image is provided")
                reference_image = Image.open(question["reference_image"]).convert("RGB")
            question = question["prompt"]
        else:
            print(f"[INFO] No reference image is provided")

        #proto_writer.clear()
        #proto_writer.extend([["question", question]])
        #if reference_image is not None:
        #    if isinstance(reference_image, list):
        #        for idx, img in enumerate(reference_image):
        #            proto_writer.extend([[f"reference_image", img]])
        #    else:
        #        proto_writer.extend([["reference_image", reference_image]])

        success = True
        #prompt = cfg.template.format(question=question)
        prompt = question
        print(f"[INFO] Handling prompt: {prompt}")
        start_time = time.time()
        if reference_image is not None:
            if isinstance(reference_image, list):
                image_str = ""
                for img in reference_image:
                    image_str += build_image(img, cfg, tokenizer, vq_model)
            else:
                image_str = build_image(reference_image, cfg, tokenizer, vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.tensor([[cfg.special_token_ids["BOS"]]], dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        for result_tokens in generate(cfg, model, tokenizer, input_ids, unconditional_ids):
            try:
                print(f"{result_tokens.shape=}")
                result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, tokenizer, vq_model)
                #proto_writer.extend(mm_out)
                id = df.loc[i, 'image_id']
                full_path = os.path.join(save_path, f"{id}.png")
                for mode, content in mm_out:
                    if mode == 'image':
                        #id = 'test5'
                        content.save(full_path)
                        end_time = time.time()
                        print(f"image {i} finished, time {end_time - start_time:.4f}", flush=True)
                    else:
                        print(mode, content)
                        image = decode_image(content, tokenizer, vq_model)
                        if image is not None:
                            image.save(full_path)
                        continue
            except Exception as e:
                success = False
                print(f"[ERROR] Failed to generate token sequence: {e}")
                break

        if not success:
            continue

        #proto_writer.save(f"{save_path}/proto/{name}.pb")


def main():
    args = parse_args()
    cfg_name = Path(args.cfg).stem
    cfg_package = Path(args.cfg).parent.__str__().replace("/", ".")
    cfg = imp.import_module(f".{cfg_name}", package=cfg_package)

    #if isinstance(cfg.prompts, dict):
    #    cfg.prompts = [(n, p) for n, p in cfg.prompts.items()]
    #else:
    #    cfg.prompts = [(f"{idx:03d}", p) for idx, p in enumerate(cfg.prompts)]

    #cfg.prompts = [(n, p) for n, p in cfg.prompts if not osp.exists(f"{cfg.save_path}/proto/{n}.pb")]
    #cfg.num_prompts = len(cfg.prompts)

    model, tokenizer, vq_model = build_emu3p5_vllm(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        vq_device=cfg.vq_device,
        seed=cfg.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )
    print(f"[INFO] Model loaded successfully")
    cfg.special_token_ids = {}
    for k, v in cfg.special_tokens.items():
        cfg.special_token_ids[k] = tokenizer.encode(v)[0]

    inference(
        args=args,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        vq_model=vq_model,
    )
    print(f"[INFO] Inference finished")


if __name__ == "__main__":
    main()
