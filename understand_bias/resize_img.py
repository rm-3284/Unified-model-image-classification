import os
import argparse
from PIL import Image

def resize_images_pipeline(input_dir: str, output_dir: str, size: int, model: str):
    #train_dir = os.path.join(output_dir, model, 'train')
    #val_dir = os.path.join(output_dir, model, 'val')
    #os.makedirs(train_dir, exist_ok=True)
    #os.makedirs(val_dir, exist_ok=True)

    #bagel_train_dir = os.path.join(output_dir, 'bagel', 'train')
    #bagel_val_dir = os.path.join(output_dir, 'bagel', 'val')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting image resizing process...")
    print(f"Target size (shorter edge): {size} pixels")
    print("-" * 40)
    
    processed_count = 0

    for filename in os.listdir(input_dir):
        #output_dir_ = os.path.join(output_dir, model, check_train_or_val(bagel_train_dir, bagel_val_dir, filename))

        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)

        #if not os.path.isfile(input_filepath):
        #    continue
            
        try:
            img = Image.open(input_filepath).convert("RGB")
            resized_img = img.resize((size, size), Image.Resampling.BICUBIC)
            resized_img.save(output_filepath)
            processed_count += 1
            
        except FileNotFoundError:
            print(f"Warning: File not found {filename}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")

    print("-" * 40)
    print(f"Successfully processed {processed_count} images.")
    print(f"Files saved to: {output_dir}")

def check_train_or_val(train_dir: str, val_dir: str, image_name: str):
    train_files = set(os.listdir(train_dir))
    val_files = set(os.listdir(val_dir))
    if image_name in train_files:
        return 'train'
    else:
        return 'val'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch resize images while maintaining aspect ratio.")
    
    parser.add_argument("input_dir", type=str, 
                        help="Path to the directory containing source images.")
    parser.add_argument("output_dir", type=str, 
                        help="Path to the directory where resized images will be saved.")
    parser.add_argument("--model", type=str)
    parser.add_argument("--size", type=int, default=224,
                        help="Target size for the shorter edge (e.g., 256).")
    
    args = parser.parse_args()

    langs = ['en', 'es', 'ja', 'tr', 'zh']


    for lang in langs:
        input_dir = os.path.join(args.input_dir, lang)
        output_dir = os.path.join(args.output_dir, lang)

        resize_images_pipeline(
            input_dir,
            output_dir,
            args.size,
            args.model
        )
