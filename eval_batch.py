#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
from tqdm import tqdm
from PIL import Image

from grounded_instruct_pix2pix_wrapper import GroundedInstructPixtoPix


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)

    # -----------------------------
    # Configuration
    # -----------------------------
    device = "cuda:0"
    num_timesteps = 100
    image_guidance_scale = 1.5
    text_guidance_scale = 7.5
    seed = 42
    blending_range = [100, 1]
    verbose = False

    edit_instruction = "Turn the table into a white marble table"
    

    input_folder = "/home/ubuntu/spjain/Qwen-SAM-Instruct-Pix2Pix/src/garden"
    output_folder = "/home/ubuntu/spjain/Qwen-SAM-Instruct-Pix2Pix/src/edited/garden_dino_marble_100iters_bb100"
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # Initialize model wrapper
    # -----------------------------
    editor = GroundedInstructPixtoPix(
        num_timesteps=num_timesteps,
        device=device,
        image_guidance_scale=image_guidance_scale,
        text_guidance_scale=text_guidance_scale,
        start_blending_at_tstep=blending_range[0],
        end_blending_at_tstep=blending_range[1],
        prompt=edit_instruction,
        seed=seed,
        verbose=verbose,
        debug = True, mode='dino'
    )

    # -----------------------------
    # Collect images
    # -----------------------------
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"Found {len(image_paths)} images.")

    # -----------------------------
    # Batch editing
    # -----------------------------
    i = 0
    for img_path in tqdm(image_paths, desc="Editing Images"):
        if(i>10):
            break
        i+=1
        # Load and resize
        img_pil = editor.load_pil_image(img_path)

        # Run editing
        edited_pil = editor.edit_image(img_pil, img_path)

        # Save result
        save_path = os.path.join(output_folder, os.path.basename(img_path))
        edited_pil.resize(img_pil.size).save(save_path)

    print("✨ Batch editing completed!")
