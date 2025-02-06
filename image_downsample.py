import os
from PIL import Image
import shutil

def downsample_images(root_folder, output_root):
    # Resolution constraints
    MIN_RES_LOWER = 300
    MIN_RES_UPPER = 512
    MAX_RES = 1024
    RATIO_THRESHOLD = 2
    
    # Create the output root folder if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # Walk through all folders and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Get the relative path from root folder
        rel_path = os.path.relpath(dirpath, root_folder)
        
        # Create corresponding output directory
        output_dir = os.path.join(output_root, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_dir, filename)
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        aspect_ratio = max(width/height, height/width)
                        max_dimension = max(width, height)
                        min_dimension = min(width, height)
                        
                        needs_resize = False
                        
                        # Different processing based on aspect ratio
                        if aspect_ratio > RATIO_THRESHOLD:
                            # For images with ratio > 2, only check max resolution
                            if max_dimension > MAX_RES:
                                needs_resize = True
                                scale_factor = MAX_RES / max_dimension
                                new_width = int(width * scale_factor)
                                new_height = int(height * scale_factor)
                        else:
                            # For images with ratio <= 2, check both conditions
                            if max_dimension > MAX_RES or min_dimension > MIN_RES_UPPER:
                                needs_resize = True
                                
                                # First handle max resolution if needed
                                if max_dimension > MAX_RES:
                                    scale_factor = MAX_RES / max_dimension
                                    new_width = int(width * scale_factor)
                                    new_height = int(height * scale_factor)
                                    
                                    # Check if this scaling satisfies the minimum dimension requirement
                                    if min(new_width, new_height) > MIN_RES_UPPER:
                                        # Need to scale down further
                                        min_dim = min(new_width, new_height)
                                        additional_scale = MIN_RES_UPPER / min_dim
                                        new_width = int(new_width * additional_scale)
                                        new_height = int(new_height * additional_scale)
                                else:
                                    # Only need to handle minimum dimension
                                    scale_factor = MIN_RES_UPPER / min_dimension
                                    new_width = int(width * scale_factor)
                                    new_height = int(height * scale_factor)
                                
                                # Verify the new dimensions meet minimum criteria
                                if min(new_width, new_height) < MIN_RES_LOWER:
                                    needs_resize = False
                        
                        if needs_resize:
                            # Resize image
                            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            resized_img.save(output_path, quality=95, optimize=True)
                            print(f"Resized {img_path} to {new_width}x{new_height} (ratio: {aspect_ratio:.2f})")
                        else:
                            # Copy original if no resize needed or if constraints can't be met
                            shutil.copy2(img_path, output_path)
                            print(f"Copied {img_path} (ratio: {aspect_ratio:.2f})")
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

# Path to your root folder
root_folder = "/Users/kunqueen/Desktop/LLMbias_papers/Dataset/PARA/imgs"
output_root = "/Users/kunqueen/Desktop/LLMbias_papers/Dataset/PARA/imgs_downsampled"

# Run the downsampling
downsample_images(root_folder, output_root)
