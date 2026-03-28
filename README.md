# Implementation details

## models

This folder includes the 5-open source models we used (BAGEL, Emu3.5, Janus, MMaDA, show-o2) with the scripts to download the checkpoints from huggingface and modified image generation scripts for our purpose.

## prompts

This folder contains the prompts we used for the experiments.

## Qwen3

This folder contains the scripts we used for analyzing the OOD images.

## transformation-examples

This folder contains the pictures for tranformed images.

## understand_bias

This folder contains our scripts for the training of Conv-Next. The code base is based on [understand_bias](https://github.com/boyazeng/understand_bias/tree/main). We added a resize script before training and modified original scripts for the corruption experiments. We mainly modified main.py, dataset.py, data_path.py and engine.py for the corruption and transformation experiments. We newly created resize_img.py, visualize_images.py for this project. The folder named figures includes all the heatmaps generated for the analysis purpose.
