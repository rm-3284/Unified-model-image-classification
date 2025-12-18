import numpy as np
import os
from torchvision import transforms
from PIL import Image, ImageFilter

def get_transformation(cj, noise, blur, size):
    if cj != 0:
        cj = (int(cj),) * 3
        added_transform = transforms.ColorJitter(*cj)
    elif noise != 0:
        def add_noise(sample):
            normalized_image = np.array(sample) / 255.0  # Normalize to [0, 1]
            sample = Image.fromarray(np.clip((normalized_image + np.random.normal(0, noise, normalized_image.shape)) * 255, 0, 255).astype(np.uint8))
            return sample
        added_transform = add_noise
    elif blur != 0:
        def blur_f(sample):
            return sample.filter(ImageFilter.GaussianBlur(radius=blur))
        added_transform = blur_f
    else:
        def resize(sample):
            original_width, original_height = sample.size
            aspect_ratio = original_width / original_height
            if original_width < original_height:  # Width is the shortest side
                new_width = size
                new_height = int(new_width / aspect_ratio)
            else:  # Height is the shortest side
                new_height = size
                new_width = int(new_height * aspect_ratio)
            return sample.resize((new_width, new_height), Image.Resampling.BICUBIC)
        added_transform = resize
    return added_transform

def process_and_save_image(input_path: str, output_path: str, cj: float, noise: float, blur: float, size: int):
    """
    Loads an image, applies a single transformation, and saves the result.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Loading image from: {input_path}")
    
    try:
        img = Image.open(input_path).convert("RGB")
        transform_func = get_transformation(cj, noise, blur, size)
        
        if isinstance(transform_func, transforms.ColorJitter):
            transform_name = 'ColorJitter'
        else:
            transform_name = transform_func.__name__
            
        print(f"Applying transformation: {transform_name}")

        transformed_img = transform_func(img)
        transformed_img.save(output_path)
        
        print("-" * 40)
        print(f"Successfully saved transformed image to: {output_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    img_path = "/n/fs/vision-mix/rm4411/resized_images/Emu3.5/train/0a2ff3532afdb6825f431aa33eb9ac7190ba3f09.png"

    output_dir = "/n/fs/vision-mix/rm4411/transformation-examples"
    os.makedirs(output_dir, exist_ok=True)

    for i in [1, 2]:
        file_name = f'cj_{i}.png'
        process_and_save_image(img_path, os.path.join(output_dir, file_name), i, 0, 0, 244)
    
    for i in [3, 5]:
        file_name = f'blur_{i}.png'
        process_and_save_image(img_path, os.path.join(output_dir, file_name), 0, 0, i, 244)

    for i in [0.2, 0.3]:
        file_name = f'noise_{i}.png'
        process_and_save_image(img_path, os.path.join(output_dir, file_name), 0, i, 0, 244)

    for i in [64, 32]:
        file_name = f'resize_{i}.png'
        process_and_save_image(img_path, os.path.join(output_dir, file_name), 0, 0, 0, i)
