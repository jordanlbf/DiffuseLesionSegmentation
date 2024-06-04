import numpy as np
from fastai.vision.all import PILImage
from PIL import Image
from pathlib import Path
from fastai.vision.all import *

def preprocess_mask(mask_path):
    """Convert mask pixel values from 127, 255 to 0, 1."""
    with Image.open(mask_path) as img:
        mask_array = np.array(img)
        mask_array[mask_array == 127] = 0  # Skin
        mask_array[mask_array == 255] = 1  # Lesion
        return PILImage.create(mask_array)

def rename_to_uppercase_jpg(image_path):
    """Renames the file extension to uppercase .JPG if it is .jpg."""
    if image_path.suffix.lower() == '.jpg':
        upper_jpg_path = image_path.with_suffix('.JPG')
        image_path.rename(upper_jpg_path)
        return upper_jpg_path
    return image_path

def get_y_fn(x):
    """Adjust path to the mask for a given image to ensure the extension and naming convention."""
    image_path = rename_to_uppercase_jpg(Path(x))
    mask_path = image_path.parent.parent / 'masks' / (image_path.stem + '.png')
    if not mask_path.exists():
        print(f"Expected mask not found: {mask_path}")
        return None
    return preprocess_mask(mask_path)

def get_aug_pipeline(size_tform):
    return  [
        *aug_transforms(
            size=size_tform,
            min_scale=0.75,
            max_rotate=15,
            do_flip=True,
            max_lighting=0.5,
            p_lighting=0.8,
            flip_vert=True,
            max_warp=0.2,  # Warp magnitude
            p_affine=0.75,  # Probability of applying affine transformations
            max_zoom=1.1,
        ),
        Normalize.from_stats(*imagenet_stats)
    ]
