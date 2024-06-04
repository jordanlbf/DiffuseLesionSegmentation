from torchvision.transforms import functional as TF
import random
from torchvision import transforms
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image

class JointTransform:
    def __init__(self, size):
        self.size = size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats

    def __call__(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Random rescaling
        scale = random.uniform(0.75, 1.25)
        image = TF.resize(image, [int(self.size[0] * scale), int(self.size[1] * scale)])
        mask = TF.resize(mask, [int(self.size[0] * scale), int(self.size[1] * scale)], interpolation=Image.NEAREST)

        # Random color jitter (brightness, contrast)
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        image = color_jitter(image)

        # Random affine transformations
        if random.random() > 0.75:
            image = TF.affine(image, angle=0, translate=(10, 10), scale=1, shear=(10, 10))
            mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1, shear=(10, 10))

        # Resize and convert to tensor
        image = TF.resize(image, self.size)
        image = TF.to_tensor(image)

        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask)

        # Normalize the image
        image = self.normalize(image)

        # Ensure mask values are only 0 or 1
        mask = mask_transform(mask)

        return image, mask

def mask_transform(mask):
    new_mask = torch.zeros_like(mask, dtype=torch.long)
    new_mask[mask == 127] = 0  # Map skin to class 0
    new_mask[mask == 255] = 1  # Map lesion to class 1
    return new_mask

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.joint_transform = joint_transform

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Ensure that we only keep pairs that exist in both folders
        self.images = [img for img in self.images if os.path.splitext(img)[0] + '.png' in self.masks]
        self.masks = [mask for mask in self.masks if mask in [os.path.splitext(img)[0] + '.png' for img in self.images]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'  # Adjust for mask naming convention

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        return image, mask
